from collections import defaultdict

import torch
import torch.nn.functional as F

# 加VGGT
import sys
import os



# 获取当前文件(train.py)的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录(train/)
current_dir = os.path.dirname(current_file_path)
# 获取上一级目录，也就是项目根目录
project_root = os.path.dirname(current_dir)

# 关键修改：将 VGGT2RGT 目录添加到 sys.path，而不是 project_root
vggt_root = os.path.join(project_root, 'VGGT2RGT')
if vggt_root not in sys.path:
    sys.path.insert(0, vggt_root)
from VGGT2RGT.vggt.distillation import create_student_vggt
from VGGT2RGT.vggt.models.vggt import VGGT


import torch.nn.functional as F
def extract_vggt_features_with_original_patches(
    images: torch.Tensor,
    original_size: int = 384,
    device: str = "cuda",
    target_size: int = 518,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    B, frames, C, H, W = images.shape
    images_flat = images.view(B*frames,C,H,W).to(device)
    images_resized = F.interpolate(
        images_flat,
        size=(target_size, target_size),
        mode='bilinear',  # 对应 PIL 的 BICUBIC
        align_corners=False
    ).view(B, frames, C, 518, 518)

    if images_resized.max() > 1.0:
        images_resized = images_resized / 255.0
    
    # 移动到设备
    images_resized = images_resized.to(device, dtype=dtype)
    
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        features = _VGGT_model.get_last_block_features(images_resized, target_size=original_size)
    # print(f"shape of VGGT output:{features.shape}")
    output = features.view(B,frames*729, -1).to(device=device, dtype=dtype)
    return output

@torch.no_grad()
def log_sample_res(
    text_encoder,
    vision_encoder,
    rdt,
    args,
    accelerator,
    weight_dtype,
    dataset_id2name,
    dataloader,
    logger,
):
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logger.info(f"Running sampling for {args.num_sample_batches} batches...")
        #0427新增VGGT定义
        global _VGGT_model
        model_path = "/media/a/9ee7cc3d-b146-4042-a066-a581a75a3537/huggingface/pretrained_model/VGGT"
        _VGGT_model = VGGT.from_pretrained(model_path).to(device=accelerator.device).eval()
        # student_path = '/media/a/9ee7cc3d-b146-4042-a066-a581a75a3537/huggingface/pretrained_model/VGGT_align/best.pth'
        # _VGGT_model = create_student_vggt(embed_dim=1024, depth=2).to(device=accelerator.device).eval()
        # checkpoint = torch.load(student_path,map_location='cpu')
        # _VGGT_model.load_state_dict(checkpoint['student_state_dict'])
        rdt.eval()

        loss_for_log = defaultdict(float)
        loss_counter = defaultdict(int)
        for step, batch in enumerate(dataloader):
            if step >= args.num_sample_batches:
                break

            data_indices = batch["data_indices"]
            ctrl_freqs = batch["ctrl_freqs"]
            state_norm = batch["state_norm"].to(dtype=weight_dtype)
            images = batch["images"].to(dtype=weight_dtype)
            states = batch["states"].to(dtype=weight_dtype)
            # We only use the last state as input
            states = states[:, -1:, :]
            actions = batch["actions"].to(dtype=weight_dtype)
            state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)

            batch_size, _, C, H, W = images.shape
            image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
            image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))
            #0427VGGT融合
            VGGT_output_embeds = extract_vggt_features_with_original_patches(
                images = images, original_size = H, device = accelerator.device, target_size=518, dtype=weight_dtype)
            # 0523新增
            geo_tokens=VGGT_output_embeds
            
            lang_attn_mask = batch["lang_attn_mask"]
            text_embeds = (batch["lang_embeds"].to(dtype=weight_dtype) if args.precomp_lang_embed else text_encoder(
                input_ids=batch["input_ids"], attention_mask=lang_attn_mask)["last_hidden_state"].detach())

            pred_actions = rdt.predict_action(
                lang_tokens=text_embeds,
                lang_attn_mask=lang_attn_mask,
                img_tokens=image_embeds,
                geo_tokens=geo_tokens,
                state_tokens=states,
                action_mask=state_elem_mask.unsqueeze(1),
                ctrl_freqs=ctrl_freqs
            )

            num_steps = pred_actions.shape[1]
            expanded_state_elem_mask = (state_elem_mask.unsqueeze(1).tile((1, num_steps, 1)).float())
            expanded_state_norm = (state_norm.unsqueeze(1).tile((1, num_steps, 1)).float())

            loss = F.mse_loss(pred_actions, actions, reduction="none").float()

            mse_loss_per_entry = (loss * expanded_state_elem_mask).reshape(
                (batch_size, -1)).sum(1) / expanded_state_elem_mask.reshape((batch_size, -1)).sum(1)
            l2_loss_per_entry = loss.sqrt() / (expanded_state_norm + 1e-3)
            l2_loss_per_entry = (l2_loss_per_entry * expanded_state_elem_mask).reshape(
                (batch_size, -1)).sum(1) / expanded_state_elem_mask.reshape((batch_size, -1)).sum(1)

            dataset_indices, mse_losses, l2_losses = accelerator.gather_for_metrics((
                torch.LongTensor(data_indices).to(device=pred_actions.device),
                mse_loss_per_entry,
                l2_loss_per_entry,
            ), )
            dataset_indices = dataset_indices.tolist()
            if accelerator.is_main_process:
                for loss_suffix, losses in zip(["_sample_mse", "_sample_l2err"], [mse_losses, l2_losses]):
                    for dataset_idx, loss_tensor in zip(dataset_indices, losses):
                        loss_name = dataset_id2name[dataset_idx] + loss_suffix
                        loss_for_log[loss_name] += loss_tensor.item()
                        loss_counter[loss_name] += 1

            mse_loss = (loss * expanded_state_elem_mask).sum() / expanded_state_elem_mask.sum()
            mse_loss_scaler = accelerator.gather(mse_loss).mean().item()
            loss_for_log["overall_avg_sample_mse"] += mse_loss_scaler

            l2_loss = loss.sqrt() / (expanded_state_norm + 1e-3)
            l2_loss = (l2_loss * expanded_state_elem_mask).sum() / expanded_state_elem_mask.sum()
            l2_loss_scaler = accelerator.gather(l2_loss).mean().item()
            loss_for_log["overall_avg_sample_l2err"] += l2_loss_scaler

        for name in loss_for_log:
            if name in ["overall_avg_sample_mse", "overall_avg_sample_l2err"]:
                loss_scaler = loss_for_log[name]
                loss_for_log[name] = round(loss_scaler / (args.num_sample_batches), 4)
            else:
                loss_for_log[name] = round(loss_for_log[name] / loss_counter[name], 4)

        rdt.train()
        torch.cuda.empty_cache()

        return dict(loss_for_log)

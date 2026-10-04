import argparse
import json
import torch.nn as nn

from dataset import trainloader, valloader, TRAIN_PATH
import datetime
import time
import logging
import os
import glob
from logger import setup_logger
from loss_v3 import fusion_loss_mef
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
from rgb2ycbcr import RGB2YCrCb, YCrCb2RGB
import random

from metric import (
    VIF_function,
    Qabf_function,
    MI_function,
    SCD_function,
    SSIM_function,
)
from metafusion_net import FusionNet, _weights_init
from Ufuser import Ufuser


import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

# 在 TensorBoard 预览中展示 Mbloom 与 Mhalo（仅当损失模块提供 mask 接口时生效）
PREVIEW_WITH_MASK = True
# 固定预览样本（来自 TRAIN_PATH，和 train/val 切分无关）
PREVIEW_IMAGE_IDS = {"00917N", "01185N", "01154N", "00326D", "00328D"}

def seed_everything(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_preview_tensor(image_vis, image_ir, fused_y, image_vis_ycrcb=None,
                        Mbloom=None, Mhalo=None):
    """
    Create preview tensor with RGB images: [VIS | IR | FUSED_RGB]
    
    Args:
        image_vis: RGB visible image [B, 3, H, W]
        image_ir: IR image [B, 1, H, W]
        fused_y: Fused Y channel [B, 1, H, W]
        image_vis_ycrcb: Optional YCrCb visible image [B, 3, H, W]
    """
    vis = image_vis[0].detach().cpu().clamp(0, 1)  # [3, H, W]
    ir = image_ir[0].detach().cpu().clamp(0, 1)    # [1, H, W]
    
    # Convert IR to RGB for display (灰度复制到 3 通道)
    if ir.shape[0] == 1:
        ir = ir.repeat(3, 1, 1)
    
    # Convert fused Y to RGB using original CrCb channels
    if image_vis_ycrcb is not None:
        fused_ycrcb = torch.cat([
            fused_y[0:1].detach().cpu().clamp(0, 1),
            image_vis_ycrcb[0:1, 1:2].detach().cpu(),  # Cr
            image_vis_ycrcb[0:1, 2:3].detach().cpu()   # Cb
        ], dim=1)  # 在通道维度拼接: [1, 3, H, W]
        fused_rgb = YCrCb2RGB(fused_ycrcb).squeeze(0).clamp(0, 1)
    else:
        # Fallback: convert grayscale to RGB
        fused_rgb = fused_y[0].detach().cpu().clamp(0, 1)
        if fused_rgb.shape[0] == 1:
            fused_rgb = fused_rgb.repeat(3, 1, 1)

    # 掩膜可视化：单通道复制成 3 通道
    def mask_to_rgb(mask):
        # mask: [B,1,H,W]
        m = mask[0].detach().cpu().clamp(0, 1)  # [1, H, W]
        if m.shape[0] == 1:
            m = m.repeat(3, 1, 1)               # [3, H, W]
        return m

    if (Mbloom is not None) and (Mhalo is not None):
        Mbloom_rgb = mask_to_rgb(Mbloom)
        Mhalo_rgb = mask_to_rgb(Mhalo)
        # 按顺序拼接：IR | VIS | Mbloom | Mhalo | FusedRGB
        preview = torch.cat([ir, vis, Mbloom_rgb, Mhalo_rgb, fused_rgb], dim=2)
    else:
        # 兼容旧逻辑：VIS | IR | FusedRGB
        preview = torch.cat([vis, ir, fused_rgb], dim=2)

    return preview


def build_fixed_previews_from_train_path(train_path, preview_ids):
    """直接从 TRAIN_PATH 读取固定预览样本。"""
    samples = []
    for sample_id in sorted(preview_ids):
        ir_candidates = sorted(glob.glob(os.path.join(train_path, "ir", f"{sample_id}.*")))
        vi_candidates = sorted(glob.glob(os.path.join(train_path, "vi", f"{sample_id}.*")))
        if not ir_candidates or not vi_candidates:
            continue
        ir_path, vi_path = ir_candidates[0], vi_candidates[0]
        image_ir = np.array(Image.open(ir_path).convert('L')).astype(np.float32) / 255.0
        image_vi = np.array(Image.open(vi_path).convert('RGB')).astype(np.float32) / 255.0
        ir_tensor = torch.from_numpy(image_ir).unsqueeze(0)                  # [1,H,W]
        vi_tensor = torch.from_numpy(image_vi).permute(2, 0, 1).contiguous() # [3,H,W]
        samples.append((sample_id, ir_tensor, vi_tensor))
    return samples


def build_train_model(backbone="ufuser"):
    """Build the fusion network. Paper main results use U-fuser (EMMA)."""
    if backbone == "ufuser":
        return Ufuser()

    if backbone != "metafusion":
        raise ValueError(f"Unknown backbone: {backbone}")

    class MetaFusionTrainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fusion = FusionNet(block_num=3, feature_out=False)
            self.fusion.apply(_weights_init)

        def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
            x = torch.cat([ir, vi, torch.abs(ir - vi), torch.max(ir, vi)], dim=1)
            _, weights = self.fusion(x)  # [B,2,H,W], sigmoid output
            weights = torch.softmax(weights, dim=1)
            fused = weights[:, 0:1, :, :] * ir + weights[:, 1:2, :, :] * vi
            return fused.clamp(0.0, 1.0)

    return MetaFusionTrainModel()


def net_trainable_params(train_model, backbone):
    if backbone == "metafusion":
        # MetaConv2d weights are exposed via fusion.params(), not model.parameters().
        return list(train_model.fusion.params())
    return list(train_model.parameters())


def forward_fused_y(model, image_ir, image_vis_y, ufuser_call="named"):
    """Fusion forward. ``named`` is (IR, VIS_Y); ``paper`` matches 151334: (VIS_Y, IR)."""
    if ufuser_call == "paper":
        out = model(image_vis_y, image_ir)
    elif ufuser_call == "named":
        out = model(image_ir, image_vis_y)
    else:
        raise ValueError(f"Unknown ufuser_call: {ufuser_call}")
    return out.clamp(0.0, 1.0)


def snapshot_loss_hparams(train_loss):
    return {
        "q_bright": float(train_loss.q_bright),
        "delta": float(train_loss.delta),
        "w_l1": float(train_loss.w_l1),
        "w_grad": float(train_loss.w_grad),
        "lambda_halo": float(train_loss.lambda_halo),
        "lambda_bloom": float(train_loss.lambda_bloom),
    }


def train(logger, exp_name=None, tb_root='./logs/tensorboard', tb_image_every=1,
          backbone="ufuser", ufuser_call="named", lambda_halo=1.0, lambda_bloom=0.5,
          q_bright=0.90):

    lr_start = 5e-4
    model_path = './model'
    model_path = os.path.join(model_path)
    os.makedirs(model_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_model = build_train_model(backbone)
    train_model.to(device)
    # init_weights(train_model)
    train_model.train()

    if backbone != "ufuser" and ufuser_call == "paper":
        logger.warning("ufuser_call=paper is ignored for backbone=%s", backbone)
        ufuser_call = "named"

    train_loss = fusion_loss_mef(
        lambda_halo=lambda_halo, lambda_bloom=lambda_bloom, q_bright=q_bright
    )
    train_loss.to(device)
    net_params = net_trainable_params(train_model, backbone)

    optimizer = torch.optim.Adam(net_params, lr=lr_start)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.75, patience=2, min_lr=1e-6
    )
    epoch = 200

    st = glob_st = time.time()
    val_best_score = 0.0
    patience_max = 20
    patience = 0
    
    # 生成实验ID（用于统一日志、tensorboard 与模型命名）
    if exp_name is None:
        exp_id = time.strftime("%Y%m%d-%H%M%S")
    else:
        exp_id = exp_name
    best_model_path = None
    tb_log_dir = os.path.join(os.path.abspath(tb_root), exp_id)
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f'Train start! Experiment: {exp_id}')
    logger.info("backbone=%s, train_batch=%s, ufuser_call=%s",
                backbone, getattr(trainloader, "batch_size", None), ufuser_call)
    if backbone == "ufuser" and ufuser_call == "paper":
        logger.info("U-fuser call is (VIS_Y, IR), matching checkpoint 20260403-151334.")
    elif backbone == "ufuser":
        logger.info("U-fuser call is (IR, VIS_Y): I_en←IR, V_en←VIS.")
    logger.info(
        "loss_v3 => additive max-int/max-grad, no SSIM, fixed lambdas. "
        "w_l1=%.2f, w_grad=%.2f, lambda_halo=%.3f, lambda_bloom=%.3f, "
        "q_bright=%.2f, delta=%.3f. "
        "Halo mask: VIS>=q AND VIS>IR. Washout mask: IR>=q AND IR>VIS. "
        "Val selection: Qabf only.",
        train_loss.w_l1,
        train_loss.w_grad,
        train_loss.lambda_halo,
        train_loss.lambda_bloom,
        train_loss.q_bright,
        train_loss.delta,
    )
    cov = max(1.0 - train_loss.q_bright, 1e-6)
    logger.info(
        "Per-pixel pull vs intensity on candidate pixels is about "
        "halo=%.2fx, washout=%.2fx  (lambda / (w_l1 * (1-q)), q=%.2f).",
        train_loss.lambda_halo / (train_loss.w_l1 * cov),
        train_loss.lambda_bloom / (train_loss.w_l1 * cov),
        train_loss.q_bright,
    )

    fixed_preview_samples = build_fixed_previews_from_train_path(TRAIN_PATH, PREVIEW_IMAGE_IDS)
    found_ids = {sid for sid, _, _ in fixed_preview_samples}
    missing_ids = PREVIEW_IMAGE_IDS - found_ids
    logger.info(f"Fixed TRAIN_PATH previews found: {sorted(found_ids)}")
    if missing_ids:
        logger.info(f"Fixed TRAIN_PATH previews missing: {sorted(missing_ids)}")

    try:
        for epo in range(epoch):
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_losses = []
            epoch_loss_dict = {
                'loss_grad': [],
                'loss_l1': [],
                'loss_reg': [],
                'mask_cov': [],
            }
            
            for it, (image_ir, image_vis) in enumerate(trainloader):
                
                train_model.train()

                image_vis = image_vis.to(device)
                image_ir = image_ir.to(device)
                image_vis_ycrcb = RGB2YCrCb(image_vis)

                logits = forward_fused_y(
                    train_model,
                    image_ir,
                    image_vis_ycrcb[:, 0:1, :, :],
                    ufuser_call=ufuser_call,
                )

                if it == 0:
                    with torch.no_grad():
                        fused_min = logits.min().item()
                        fused_max = logits.max().item()
                        fused_mean = logits.mean().item()
                        ir_mean = image_ir.mean().item()
                        vis_mean = image_vis_ycrcb[:, 0:1, :, :].mean().item()
                    writer.add_scalar('train/fused_min', fused_min, epo + 1)
                    writer.add_scalar('train/fused_max', fused_max, epo + 1)
                    writer.add_scalar('train/fused_mean', fused_mean, epo + 1)
                    writer.add_scalar('train/ir_mean', ir_mean, epo + 1)
                    writer.add_scalar('train/vis_mean', vis_mean, epo + 1)
                    if fused_max < 1e-3:
                        logger.warning(
                            f"Epoch {epo+1}: fused output near zero (min={fused_min:.4g}, "
                            f"max={fused_max:.4g}, mean={fused_mean:.4g})"
                        )
                
                # 生成对抗样本
                # image_vis_adv, image_ir_adv = attack(image_vis, image_ir, train_model, train_loss)

                # logits_adv = train_model(image_vis_adv, image_ir_adv)
                

                optimizer.zero_grad()


                loss_total, loss_grad, loss_l1, loss_reg, mask_cov = train_loss(
                    image_ir, image_vis_ycrcb[:, 0:1, :, :], logits
                )
                # loss_total_adv, loss_mse_adv, loss_ssim_adv = train_loss(logits_adv, image_gt_ycbcr)

                # loss = loss_total + loss_total_adv
                loss = loss_total
                
                # 检查损失是否为 NaN 或 Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Loss is NaN or Inf at epoch {epo}, iter {it}, skipping...")
                    continue
                
                # 累积loss用于epoch平均
                epoch_losses.append(loss.item())
                epoch_loss_dict['loss_grad'].append(float(loss_grad.detach().cpu()))
                epoch_loss_dict['loss_l1'].append(float(loss_l1.detach().cpu()))
                epoch_loss_dict['loss_reg'].append(float(loss_reg.detach().cpu()))
                epoch_loss_dict['mask_cov'].append(float(mask_cov.detach().cpu()))
                
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    net_params,
                    max_norm=1.0,
                )
                
                optimizer.step()
                
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = len(trainloader) * epo + it + 1
                eta = int((len(trainloader) * epoch - now_it)
                          * (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                
                # 每50个iteration输出一次进度（可选，用于监控训练进度）
                if now_it % 50 == 0:
                    logger.info(f"Epoch {epo+1}/{epoch}, Iter {it+1}/{len(trainloader)}, "
                              f"loss: {loss.item():.4f}, eta: {eta}")
                st = ed
            
            # 每个epoch结束时输出平均训练loss
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            avg_loss_dict = {key: sum(vals) / len(vals) if vals else 0.0 
                            for key, vals in epoch_loss_dict.items()}
            logger.info(f"Epoch {epo+1}/{epoch} - Train Loss: {avg_epoch_loss:.4f} "
                        f"(l1: {avg_loss_dict['loss_l1']:.4f}, "
                        f"grad: {avg_loss_dict['loss_grad']:.4f}, "
                        f"reg: {avg_loss_dict['loss_reg']:.4f}, "
                        f"mask_cov: {avg_loss_dict['mask_cov']:.4f}, "
                        f"LR: {current_lr:.6f})")

            writer.add_scalar('train/loss', avg_epoch_loss, epo + 1)
            writer.add_scalar('train/loss_grad', avg_loss_dict['loss_grad'], epo + 1)
            writer.add_scalar('train/loss_l1', avg_loss_dict['loss_l1'], epo + 1)
            writer.add_scalar('train/loss_reg', avg_loss_dict['loss_reg'], epo + 1)
            writer.add_scalar('train/mask_cov', avg_loss_dict['mask_cov'], epo + 1)
            writer.add_scalar('train/lr', current_lr, epo + 1)
            writer.add_scalar('train/grad_norm', float(grad_norm), epo + 1)

            # 验证阶段
            train_model.eval()
            total_mi = 0.0
            total_qabf = 0.0
            total_scd = 0.0
            total_vif = 0.0
            total_ssim = 0.0
            val_count = 0
            
            with torch.no_grad():
                for it, (image_ir, image_vis) in enumerate(valloader):
                    image_vis = image_vis.to(device)
                    image_ir = image_ir.to(device)
                    image_vis_ycrcb = RGB2YCrCb(image_vis)
                    image_vis_y = image_vis_ycrcb[:, 0:1, :, :]

                    fused = forward_fused_y(
                        train_model,
                        image_ir,
                        image_vis_y,
                        ufuser_call=ufuser_call,
                    )
                    fused_clamped = fused.clamp(0, 1)

                    if tb_image_every > 0 and (epo % tb_image_every == 0) and it < 3:
                        use_masks = (
                            PREVIEW_WITH_MASK
                            and hasattr(train_loss, 'get_M_bloom_mask_union')
                            and hasattr(train_loss, 'get_M_halo_mask_union')
                        )
                        if use_masks:
                            Mbloom = train_loss.get_M_bloom_mask_union(image_ir, image_vis_y)
                            Mhalo = train_loss.get_M_halo_mask_union(image_ir, image_vis_y)
                            preview = make_preview_tensor(
                                image_vis, image_ir, fused_clamped, image_vis_ycrcb,
                                Mbloom=Mbloom, Mhalo=Mhalo
                            )
                        else:
                            preview = make_preview_tensor(
                                image_vis, image_ir, fused_clamped, image_vis_ycrcb
                            )
                        writer.add_image(f'val/preview_{it}', preview, epo + 1)
                        
                        if it == 0:
                            writer.add_scalar('val/fused_min', fused.min().item(), epo + 1)
                            writer.add_scalar('val/fused_max', fused.max().item(), epo + 1)
                            writer.add_scalar('val/fused_mean', fused.mean().item(), epo + 1)
                    
                    image_ir_np = (image_ir.squeeze().cpu().numpy() * 255.0).astype(np.float32)
                    image_vis_y_np = (image_vis_y.squeeze().cpu().numpy() * 255.0).astype(np.float32)
                    fused_np = (fused_clamped.squeeze().cpu().numpy() * 255.0).astype(np.float32)
                    
                    mi = MI_function(image_ir_np, image_vis_y_np, fused_np)
                    qabf = Qabf_function(image_ir_np, image_vis_y_np, fused_np)
                    scd = SCD_function(image_ir_np, image_vis_y_np, fused_np)
                    vif = VIF_function(image_ir_np, image_vis_y_np, fused_np)
                    ssim_val = SSIM_function(image_ir_np, image_vis_y_np, fused_np)

                    total_mi += mi
                    total_qabf += qabf
                    total_scd += scd
                    total_vif += vif
                    total_ssim += ssim_val
                    val_count += 1

            # 验证集上记录五项；选模和学习率只看 Qabf
            avg_mi = total_mi / val_count
            avg_qabf = total_qabf / val_count
            avg_scd = total_scd / val_count
            avg_vif = total_vif / val_count
            avg_ssim = total_ssim / val_count
            val_score = float(avg_qabf)

            logger.info(
                f"Epoch {epo + 1} val raw metrics — "
                f"mi={avg_mi:.6f}, qabf={avg_qabf:.6f}, scd={avg_scd:.6f}, "
                f"vif={avg_vif:.6f}, ssim={avg_ssim:.6f}"
            )
            logger.info(
                f"Epoch {epo + 1} val_score (Qabf)={val_score:.6f}"
            )

            writer.add_scalar('val/mi', avg_mi, epo + 1)
            writer.add_scalar('val/qabf', avg_qabf, epo + 1)
            writer.add_scalar('val/scd', avg_scd, epo + 1)
            writer.add_scalar('val/vif', avg_vif, epo + 1)
            writer.add_scalar('val/ssim', avg_ssim, epo + 1)
            writer.add_scalar('val/score', val_score, epo + 1)

            # 固定样本预览（每个 epoch 同一组，来自 TRAIN_PATH）
            if tb_image_every > 0 and (epo % tb_image_every == 0):
                train_model.eval()
                with torch.no_grad():
                    for sample_id, ir_cpu, vis_cpu in fixed_preview_samples:
                        image_ir = ir_cpu.unsqueeze(0).to(device)   # [1,1,H,W]
                        image_vis = vis_cpu.unsqueeze(0).to(device) # [1,3,H,W]
                        image_vis_ycrcb = RGB2YCrCb(image_vis)
                        image_vis_y = image_vis_ycrcb[:, 0:1, :, :]
                        fused = forward_fused_y(
                            train_model,
                            image_ir,
                            image_vis_y,
                            ufuser_call=ufuser_call,
                        ).clamp(0, 1)

                        use_masks = (
                            PREVIEW_WITH_MASK
                            and hasattr(train_loss, 'get_M_bloom_mask_union')
                            and hasattr(train_loss, 'get_M_halo_mask_union')
                        )
                        if use_masks:
                            Mbloom = train_loss.get_M_bloom_mask_union(image_ir, image_vis_y)
                            Mhalo = train_loss.get_M_halo_mask_union(image_ir, image_vis_y)
                            preview = make_preview_tensor(
                                image_vis, image_ir, fused, image_vis_ycrcb,
                                Mbloom=Mbloom, Mhalo=Mhalo
                            )
                        else:
                            preview = make_preview_tensor(image_vis, image_ir, fused, image_vis_ycrcb)
                        writer.add_image(f'fixed_preview/{sample_id}', preview, epo + 1)

            # 根据验证指标更新学习率
            scheduler.step(val_score)

            if val_score > val_best_score:
                old_score = val_best_score
                val_best_score = val_score
                best_model_name = f'{exp_id}-{val_best_score:.6f}-best.pth'
                new_best_model_path = os.path.join(model_path, best_model_name)
                torch.save(train_model.state_dict(), new_best_model_path)
                if best_model_path is not None and best_model_path != new_best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                best_model_path = new_best_model_path
                hparams = snapshot_loss_hparams(train_loss)
                hparams.update({
                    "epoch": epo + 1,
                    "val_score": float(val_best_score),
                    "val_mi": float(avg_mi),
                    "val_qabf": float(avg_qabf),
                    "val_vif": float(avg_vif),
                    "model": best_model_name,
                })
                hparams_json_path = os.path.join(model_path, f"{exp_id}-best-hparams.json")
                with open(hparams_json_path, "w") as f:
                    json.dump(hparams, f, indent=2)
                logger.info("Best model updated: {} (score: {:.4f} -> {:.4f})".format(
                    best_model_name, old_score, val_best_score))
                patience = 0
            else:
                patience += 1
                logger.info(f"Val score not improved. Patience: {patience}/{patience_max}")
                if patience >= patience_max:
                    logger.info("Early stopping triggered at epoch {}".format(epo))
                    break
    finally:
        writer.close()
        hparams = snapshot_loss_hparams(train_loss)
        last_model_path = os.path.join(model_path, f"{exp_id}-last.pth")
        torch.save(train_model.state_dict(), last_model_path)
        with open(os.path.join(model_path, f"{exp_id}-last-hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)
        logger.info("Saved last-epoch model: %s", last_model_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train fusion with fixed halo/washout weights.")
    parser.add_argument("--backbone", type=str, default="ufuser", choices=["ufuser", "metafusion"])
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional run-name suffix. Auto-set from lambda_halo/bloom if empty.")
    parser.add_argument("--lambda_halo", type=float, default=1.0,
                        help="Fixed halo weight. Not learned. Paper default: 1.0")
    parser.add_argument("--lambda_bloom", type=float, default=0.5,
                        help="Fixed washout weight (--lambda_bloom is the historical flag name). Paper default: 0.5")
    parser.add_argument("--q-bright", dest="q_bright", type=float, default=0.90,
                        help="Intensity quantile for candidate halo/washout masks.")
    parser.add_argument(
        "--ufuser-call",
        dest="ufuser_call",
        type=str,
        default="named",
        choices=["named", "paper"],
        help="named: model(IR, VIS_Y). paper: model(VIS_Y, IR).",
    )
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed_everything(2026)

    logpath = './logs'
    stamp = time.strftime("%Y%m%d-%H%M%S")
    auto_tag = []
    if args.backbone != "ufuser":
        auto_tag.append(args.backbone)
    if args.backbone == "ufuser" and args.ufuser_call == "paper":
        auto_tag.append("paper-call")
    auto_tag.append(
        f"h{str(args.lambda_halo).replace('.', 'p')}-w{str(args.lambda_bloom).replace('.', 'p')}"
    )
    if abs(args.q_bright - 0.90) > 1e-8:
        auto_tag.append(f"q{str(args.q_bright).replace('.', 'p')}")
    if args.tag:
        run_id = f"{stamp}-{args.tag}"
    else:
        run_id = f"{stamp}-{'-'.join(auto_tag)}"
    logger = logging.getLogger()
    setup_logger(logpath, run_id=run_id)
    train(
        logger,
        exp_name=run_id,
        tb_root=os.path.join(logpath, 'tensorboard'),
        tb_image_every=1,
        backbone=args.backbone,
        ufuser_call=args.ufuser_call,
        lambda_halo=args.lambda_halo,
        lambda_bloom=args.lambda_bloom,
        q_bright=args.q_bright,
    )
    logger.info("Train finish!")






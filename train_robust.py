from Ufuser import Ufuser

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
import itertools

from metric import (
    VIF_function,
    Qabf_function,
    MI_function,
    SCD_function,
    SSIM_function,
    composite_validation_score,
)


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


def train(logger, exp_name=None, tb_root='./logs/tensorboard', tb_image_every=1, lambda_freeze_epochs=10):

    lr_start = 5e-4
    model_path = './model'
    model_path = os.path.join(model_path)
    os.makedirs(model_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_model = Ufuser()
    train_model.to(device)
    # init_weights(train_model)
    train_model.train()

    train_loss = fusion_loss_mef()
    train_loss.to(device)

    optimizer = torch.optim.Adam(
        itertools.chain(train_model.parameters(), train_loss.parameters()),
        lr=lr_start,
    )

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
    with torch.no_grad():
        halo_l = train_loss.get_lambda_halo().detach().cpu().tolist()
        bloom_l = train_loss.get_lambda_bloom().detach().cpu().tolist()
    logger.info(
        "loss_v3 => w_l1=%.2f, w_grad=%.2f, w_ssim=%.2f, halo_bins=%s, bloom_bins=%s, "
        "lambda_halo_init=%s, lambda_bloom_init=%s, lambda_min=%.2f, lambda_freeze_epochs=%d",
        train_loss.w_l1,
        train_loss.w_grad,
        train_loss.w_ssim,
        train_loss.halo_bins,
        train_loss.bloom_bins,
        [round(v, 4) for v in halo_l],
        [round(v, 4) for v in bloom_l],
        train_loss.lambda_min,
        lambda_freeze_epochs,
    )

    fixed_preview_samples = build_fixed_previews_from_train_path(TRAIN_PATH, PREVIEW_IMAGE_IDS)
    found_ids = {sid for sid, _, _ in fixed_preview_samples}
    missing_ids = PREVIEW_IMAGE_IDS - found_ids
    logger.info(f"Fixed TRAIN_PATH previews found: {sorted(found_ids)}")
    if missing_ids:
        logger.info(f"Fixed TRAIN_PATH previews missing: {sorted(missing_ids)}")

    try:
        for epo in range(epoch):
            if epo == 0:
                train_loss.lambda_halo_raw.requires_grad_(False)
                train_loss.lambda_bloom_raw.requires_grad_(False)
                logger.info(
                    "Lambda learning frozen for the first %d epochs.",
                    lambda_freeze_epochs,
                )
            if epo == lambda_freeze_epochs:
                train_loss.lambda_halo_raw.requires_grad_(True)
                train_loss.lambda_bloom_raw.requires_grad_(True)
                with torch.no_grad():
                    halo_now = train_loss.get_lambda_halo().detach().cpu().tolist()
                    bloom_now = train_loss.get_lambda_bloom().detach().cpu().tolist()
                logger.info(
                    "Lambda unfrozen at epoch %d. lambdas: halo=%s, bloom=%s",
                    epo + 1,
                    [round(v, 4) for v in halo_now],
                    [round(v, 4) for v in bloom_now],
                )
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_losses = []
            epoch_loss_dict = {
                'loss_grad': [],
                'loss_l1': [],
                'loss_ssim': [],
                'loss_reg': [],
            }
            
            for it, (image_ir, image_vis) in enumerate(trainloader):
                
                train_model.train()

                image_vis = image_vis.to(device)
                image_ir = image_ir.to(device)
                image_vis_ycrcb = RGB2YCrCb(image_vis)

                logits = train_model(image_vis_ycrcb[:, 0:1, :, :], image_ir)

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


                loss_total, loss_grad, loss_l1, loss_ssim, loss_reg = train_loss(
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
                epoch_loss_dict['loss_ssim'].append(float(loss_ssim.detach().cpu()))
                epoch_loss_dict['loss_reg'].append(float(loss_reg.detach().cpu()))
                
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    itertools.chain(train_model.parameters(), train_loss.parameters()),
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
                        f"ssim: {avg_loss_dict['loss_ssim']:.4f}, "
                        f"reg(h+b): {avg_loss_dict['loss_reg']:.4f}, "
                        f"LR: {current_lr:.6f})")

            writer.add_scalar('train/loss', avg_epoch_loss, epo + 1)
            writer.add_scalar('train/loss_grad', avg_loss_dict['loss_grad'], epo + 1)
            writer.add_scalar('train/loss_l1', avg_loss_dict['loss_l1'], epo + 1)
            writer.add_scalar('train/loss_ssim', avg_loss_dict['loss_ssim'], epo + 1)
            writer.add_scalar('train/loss_reg', avg_loss_dict['loss_reg'], epo + 1)
            writer.add_scalar('train/lr', current_lr, epo + 1)
            writer.add_scalar('train/grad_norm', float(grad_norm), epo + 1)
            if hasattr(train_loss, 'get_lambda_halo'):
                with torch.no_grad():
                    halo_l = train_loss.get_lambda_halo().detach().cpu().tolist()
                for i, v in enumerate(halo_l):
                    writer.add_scalar(f'train/lambda_halo_{i+1}', v, epo + 1)
            if hasattr(train_loss, 'get_lambda_bloom'):
                with torch.no_grad():
                    bloom_l = train_loss.get_lambda_bloom().detach().cpu().tolist()
                for i, v in enumerate(bloom_l):
                    writer.add_scalar(f'train/lambda_bloom_{i+1}', v, epo + 1)

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

                    fused = train_model(image_vis_y, image_ir)
                    fused_clamped = fused.clamp(0, 1)

                    if tb_image_every > 0 and (epo % tb_image_every == 0) and it < 3:
                        use_masks = (
                            PREVIEW_WITH_MASK
                            and hasattr(train_loss, 'get_M_bloom_mask_union')
                            and hasattr(train_loss, 'get_M_halo_mask_union')
                        )
                        if use_masks:
                            # Preview uses union masks when the loss module provides them.
                            Mbloom = train_loss.get_M_bloom_mask_union(image_ir)
                            Mhalo = train_loss.get_M_halo_mask_union(image_vis_y)
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

            # 验证集上五项指标的原始均值，再按经验范围归一化后平均得到 val_score
            avg_mi = total_mi / val_count
            avg_qabf = total_qabf / val_count
            avg_scd = total_scd / val_count
            avg_vif = total_vif / val_count
            avg_ssim = total_ssim / val_count
            val_score = composite_validation_score(
                avg_mi, avg_qabf, avg_scd, avg_vif, avg_ssim
            )

            logger.info(
                f"Epoch {epo + 1} val raw metrics — "
                f"mi={avg_mi:.6f}, qabf={avg_qabf:.6f}, scd={avg_scd:.6f}, "
                f"vif={avg_vif:.6f}, ssim={avg_ssim:.6f}"
            )
            logger.info(
                f"Epoch {epo + 1} val_score (normalized mean of five)={val_score:.6f}"
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
                        fused = train_model(image_vis_y, image_ir).clamp(0, 1)

                        use_masks = (
                            PREVIEW_WITH_MASK
                            and hasattr(train_loss, 'get_M_bloom_mask_union')
                            and hasattr(train_loss, 'get_M_halo_mask_union')
                        )
                        if use_masks:
                            Mbloom = train_loss.get_M_bloom_mask_union(image_ir)
                            Mhalo = train_loss.get_M_halo_mask_union(image_vis_y)
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


if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    seed_everything(2026)

    logpath = './logs'
    run_id = time.strftime("%Y%m%d-%H%M%S")
    logger = logging.getLogger()
    setup_logger(logpath, run_id=run_id)
    train(logger, exp_name=run_id, tb_root=os.path.join(logpath, 'tensorboard'), tb_image_every=1)
    logger.info("Train finish!")






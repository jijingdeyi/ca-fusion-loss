from Ufuser import Ufuser

from dataset import trainloader, valloader
import datetime
import time
import logging
import os
from logger import setup_logger
from loss import fusion_loss
import torch
from torch.utils.tensorboard import SummaryWriter
import warnings
from rgb2ycbcr import RGB2YCrCb, YCrCb2RGB
import random

from metric import VIF_function, Qabf_function


import numpy as np

warnings.filterwarnings('ignore')

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


def attack(
    image_vis,
    image_ir,
    model,
    loss,
    step_size=1 / 255,
    total_steps=3,
    epsilon=4 / 255,
):


    model.eval()

    # random start in [-epsilon, epsilon]
    adv_img_vis = (image_vis + torch.empty_like(image_vis).uniform_(-epsilon, epsilon)).clamp(0, 1).detach()
    adv_img_ir  = (image_ir  + torch.empty_like(image_ir ).uniform_(-epsilon, epsilon)).clamp(0, 1).detach()

    adv_img_vis.requires_grad_(True)
    adv_img_ir.requires_grad_(True)

    for _ in range(total_steps):
        model.zero_grad(set_to_none=True)
        if adv_img_vis.grad is not None:
            adv_img_vis.grad = None
        if adv_img_ir.grad is not None:
            adv_img_ir.grad = None

        fused = model(adv_img_vis, adv_img_ir)
        loss_total = loss(adv_img_ir, adv_img_vis, fused)
        loss_total.backward()

        with torch.no_grad():

            assert adv_img_vis.grad is not None and adv_img_ir.grad is not None
            adv_img_vis = adv_img_vis + step_size * adv_img_vis.grad.sign()
            adv_img_ir  = adv_img_ir  + step_size * adv_img_ir .grad.sign()

            adv_img_vis = image_vis + (adv_img_vis - image_vis).clamp(-epsilon, epsilon)
            adv_img_ir  = image_ir  + (adv_img_ir  - image_ir ).clamp(-epsilon, epsilon)

            adv_img_vis.clamp_(0, 1)
            adv_img_ir.clamp_(0, 1)

        adv_img_vis = adv_img_vis.detach().requires_grad_(True)
        adv_img_ir  = adv_img_ir.detach().requires_grad_(True)

    return adv_img_vis, adv_img_ir




def make_preview_tensor(image_vis, image_ir, fused_y, image_vis_ycrcb=None):
    """
    Create preview tensor with RGB images: [VIS | IR | FUSED_RGB]
    
    Args:
        image_vis: RGB visible image [B, 3, H, W]
        image_ir: IR image [B, 1, H, W]
        fused_y: Fused Y channel [B, 1, H, W]
        image_vis_ycrcb: Optional YCrCb visible image [B, 3, H, W]
    """
    vis = image_vis[0].detach().cpu().clamp(0, 1)
    ir = image_ir[0].detach().cpu().clamp(0, 1)
    
    # Convert IR to RGB for display
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

    # Concatenate along width: [VIS | IR | FUSED_RGB]
    preview = torch.cat([vis, ir, fused_rgb], dim=2)
    return preview


def train(logger, exp_name=None, tb_root='./logs/tensorboard', tb_image_every=1):

    lr_start = 0.001
    model_path = './model'
    model_path = os.path.join(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_model = Ufuser()
    train_model.to(device)
    # init_weights(train_model)
    train_model.train()

    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr_start)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.75, patience=2, min_lr=1e-6
    )

    train_loss = fusion_loss()
    train_loss.to(device)  
    epoch = 50

    st = glob_st = time.time()
    val_best_score = 0.0
    patience_max = 5
    patience = 0
    
    # 生成实验ID（用于区分不同实验）
    if exp_name is None:
        exp_id = time.strftime("%Y%m%d-%H%M%S")
    else:
        exp_id = f"{exp_name}-{time.strftime('%Y%m%d-%H%M%S')}"
    best_model_name = f'A2RNet-{exp_id}-best.pth'
    best_model_path = os.path.join(model_path, best_model_name)
    writer = SummaryWriter(log_dir=os.path.join(tb_root, exp_id))
    logger.info(f'Train start! Experiment: {exp_id}')

    try:
        for epo in range(epoch):
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_losses = []
            epoch_loss_dict = {
                'loss_sal': [], 'loss_grad': [], 'loss_ssim': [],
                'loss_mean': []
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


                loss_total, loss_dict = train_loss(image_ir, image_vis_ycrcb[:, 0:1, :, :], logits)
                # loss_total_adv, loss_mse_adv, loss_ssim_adv = train_loss(logits_adv, image_gt_ycbcr)

                # loss = loss_total + loss_total_adv
                loss = loss_total
                
                # 检查损失是否为 NaN 或 Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Loss is NaN or Inf at epoch {epo}, iter {it}, skipping...")
                    continue
                
                # 累积loss用于epoch平均
                epoch_losses.append(loss.item())
                for key in epoch_loss_dict:
                    epoch_loss_dict[key].append(loss_dict[key])
                
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
                
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
                       f"(sal: {avg_loss_dict['loss_sal']:.4f}, "
                       f"grad: {avg_loss_dict['loss_grad']:.4f}, "
                       f"ssim: {avg_loss_dict['loss_ssim']:.4f}, "
                       f"mean: {avg_loss_dict['loss_mean']:.4f}, "
                       f"LR: {current_lr:.6f}")

            writer.add_scalar('train/loss', avg_epoch_loss, epo + 1)
            writer.add_scalar('train/loss_sal', avg_loss_dict['loss_sal'], epo + 1)
            writer.add_scalar('train/loss_grad', avg_loss_dict['loss_grad'], epo + 1)
            writer.add_scalar('train/loss_ssim', avg_loss_dict['loss_ssim'], epo + 1)
            writer.add_scalar('train/loss_mean', avg_loss_dict['loss_mean'], epo + 1)
            writer.add_scalar('train/lr', current_lr, epo + 1)
            writer.add_scalar('train/grad_norm', float(grad_norm), epo + 1)

            # 验证阶段
            train_model.eval()
            total_qabf = 0.0
            total_vif = 0.0
            val_count = 0
            
            with torch.no_grad():
                for it, (image_ir, image_vis) in enumerate(valloader):
                    image_vis = image_vis.to(device)
                    image_ir = image_ir.to(device)
                    image_vis_ycrcb = RGB2YCrCb(image_vis)
                    image_vis_y = image_vis_ycrcb[:, 0:1, :, :]

                    fused = train_model(image_vis_y, image_ir)

                    if tb_image_every > 0 and (epo % tb_image_every == 0) and it < 3:
                        # 生成RGB融合图像
                        preview = make_preview_tensor(image_vis, image_ir, fused, image_vis_ycrcb)
                        writer.add_image(f'val/preview_{it}', preview, epo + 1)
                        
                        if it == 0:
                            writer.add_scalar('val/fused_min', fused.min().item(), epo + 1)
                            writer.add_scalar('val/fused_max', fused.max().item(), epo + 1)
                            writer.add_scalar('val/fused_mean', fused.mean().item(), epo + 1)
                    
                    image_ir_np = (image_ir.squeeze().cpu().numpy() * 255.0).astype(np.float32)
                    image_vis_y_np = (image_vis_y.squeeze().cpu().numpy() * 255.0).astype(np.float32)
                    fused_np = (fused.squeeze().cpu().numpy() * 255.0).astype(np.float32)
                    
                    qabf = Qabf_function(image_ir_np, image_vis_y_np, fused_np)
                    vif = VIF_function(image_ir_np, image_vis_y_np, fused_np)

                    total_qabf += qabf
                    total_vif += vif
                    val_count += 1

            # 计算平均指标
            avg_qabf = total_qabf / val_count
            avg_vif = total_vif / val_count
            val_score = avg_qabf + 0.5 * avg_vif

            logger.info(f"Epoch {epo}: val_qabf={avg_qabf:.4f}, val_vif={avg_vif:.4f}, val_score={val_score:.4f}")

            writer.add_scalar('val/qabf', avg_qabf, epo + 1)
            writer.add_scalar('val/vif', avg_vif, epo + 1)
            writer.add_scalar('val/score', val_score, epo + 1)

            # 根据验证指标更新学习率
            scheduler.step(val_score)

            if val_score > val_best_score:
                old_score = val_best_score
                val_best_score = val_score
                torch.save(train_model.state_dict(), best_model_path)
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
    logger = logging.getLogger()
    setup_logger(logpath)
    train(logger, tb_root=os.path.join(logpath, 'tensorboard'), tb_image_every=1)
    logger.info("Train finish!")






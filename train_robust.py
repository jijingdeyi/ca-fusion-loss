from model import ESSA_UNet

from dataset import trainloader
import datetime
import time
import logging
import os
from logger import setup_logger
from loss import fusion_loss
import torch
import warnings
from rgb2ycbcr import RGB2YCrCb
import random

import numpy as np
warnings.filterwarnings('ignore')

def seed_everything(seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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




def train(logger):

    lr_start = 0.001
    model_path = './model'
    model_path = os.path.join(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_model = ESSA_UNet()
    train_model.to(device)
    train_model.train()

    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr_start)

    train_loss = fusion_loss()
    train_loss.to(device)  # 将损失函数也移到 GPU
    epoch = 50

    st = glob_st = time.time()
    logger.info('Train start!')

    for epo in range(epoch):
        lr_start = 0.001
        lr_decay = 0.75
        lr_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_epo
        for it, (image_ir, image_vis) in enumerate(trainloader):
            
            train_model.train()

            image_vis = image_vis.to(device)
            image_ir = image_ir.to(device)
            image_vis_ycrcb = RGB2YCrCb(image_vis)

            logits = train_model(image_vis_ycrcb, image_ir)  # inputs
            
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
            
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
            
            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = len(trainloader) * epo + it + 1
            eta = int((len(trainloader) * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss: {loss:.4f}',
                        # 'loss_total: {loss_total:.4f}',
                        'loss_sal: {loss_sal:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',
                        'loss_mean: {loss_mean:.4f}',
                        'loss_tv: {loss_tv:.4f}',

                        # 对抗loss
                        # 'loss_total_adv: {loss_total_adv:.4f}',

                        'time: {time:.4f}',
                        'eta: {eta}',
                    ]
                ).format(
                        it=now_it,
                        max_it=len(trainloader) * epoch,
                        loss=loss.item(),
                        # loss_total=loss_total.item(),
                        loss_sal=loss_dict['loss_sal'],
                        loss_grad=loss_dict['loss_grad'],
                        loss_ssim=loss_dict['loss_ssim'],
                        loss_mean=loss_dict['loss_mean'],
                        loss_tv=loss_dict['loss_tv'],
                        # 对抗loss
                        # loss_total_adv=loss_total_adv.item(),
                        # loss_mse_adv=loss_mse_adv.item(),
                        # loss_ssim_adv=loss_ssim_adv.item(),

                        time=t_intv,
                        eta=eta,
                    )
                logger.info(msg)
                st = ed


    train_model_file = os.path.join(model_path, f'{loss_total.item()}.pth')
    torch.save(train_model.state_dict(), train_model_file)
    logger.info("Train model save as: {}".format(train_model_file))
    logger.info('\n')

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    seed_everything(2026)

    logpath = './logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    train(logger)
    logger.info("Train finish!")






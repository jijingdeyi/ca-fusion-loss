from torch.autograd import Variable
from model import ESSA_UNet

from dataset import fusion_dataset_gt
import datetime
import time
import logging
import os
from logger import setup_logger
from loss import fusion_loss
import torch
from torch.utils.data import DataLoader
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

    ir_path = '/home/lijiawei/local/dataset/train/ir/'
    vis_path = '/home/lijiawei/local/dataset/train/vis/'
    gt_path = '/home/lijiawei/local/dataset/train/gt/'


    train_dataset = fusion_dataset_gt(ir_path=ir_path, vis_path=vis_path, gt_path=gt_path)
    logger.info("The length of training dataset:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    train_loss = fusion_loss()


    epoch = 50

    st = glob_st = time.time()
    logger.info('Train start!')

    for epo in range(0, epoch):
        lr_start = 0.001
        lr_decay = 0.75
        lr_epo = lr_start * lr_decay ** (epo - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_epo
        for it, (image_vis, image_ir, image_gt, name) in enumerate(train_loader):
            train_model.train()
            image_vis = Variable(image_vis).cuda()
            image_vis_ycrcb = RGB2YCrCb(image_vis)
            image_ir = Variable(image_ir).cuda()

            logits = train_model(image_vis_ycrcb, image_ir)  # inputs
            
            # 生成对抗样本
            # image_vis_adv, image_ir_adv = attack_think2(image_vis, image_ir, logits, train_model, train_loss_adv)
            image_vis_adv, image_ir_adv = attack(image_vis, image_ir, train_model, train_loss)

            logits_adv = train_model(image_vis_adv, image_ir_adv)
            

            optimizer.zero_grad()


            loss_total = train_loss(image)
            loss_total_adv, loss_mse_adv, loss_ssim_adv = train_loss(logits_adv, image_gt_ycbcr)

            loss = loss_total + loss_total_adv
            loss.backward()
            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = len(train_loader) * epo + it + 1
            eta = int((len(train_loader) * epoch - now_it)
                      * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'step: {it}/{max_it}',
                        'loss: {loss:.4f}',
                        'loss_total: {loss_total:.4f}',
                        'loss_mse: {loss_mse:.4f}',
                        'loss_ssim: {loss_ssim:.4f}',

                        # 对抗loss
                        'loss_total_adv: {loss_total_adv:.4f}',
                        'loss_mse_adv: {loss_mse_adv:.4f}',
                        'loss_ssim_adv: {loss_ssim_adv:.4f}',

                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                        it=now_it,
                        max_it=len(train_loader) * epoch,
                        loss=loss.item(),
                        loss_total=loss_total.item(),
                        loss_mse=loss_mse.item(),
                        loss_ssim=loss_ssim.item(),

                        # 对抗loss
                        loss_total_adv=loss_total_adv.item(),
                        loss_mse_adv=loss_mse_adv.item(),
                        loss_ssim_adv=loss_ssim_adv.item(),

                        time=t_intv,
                        eta=eta,
                    )
                logger.info(msg)
                st = ed


    train_model_file = os.path.join(model_path, 'rebuttal_13.pth')
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
    print("Train finish!")






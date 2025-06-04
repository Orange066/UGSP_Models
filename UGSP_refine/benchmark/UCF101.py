import os
import sys
sys.path.append('.')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab
from model.RIFE import Model
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = '../datasets/ucf101_interp/'

count_time = False
use_sparsity = True
model = Model(count_time = count_time)
model.load_model_epoch('train_log', 284)
model.eval()
model.device()
for m in model.flownet.modules():
    if hasattr(m, '_prepare'):
        m._prepare()

dirs = os.listdir(path)
dirs.sort()
psnr_list = []
ssim_list = []
global_sparsity_list = []
local_sparsity_list = []
print(len(dirs))
space_check_8_l =[]
space_check_4_l =[]
space_check_2_l =[]
time_lapse_list = []
for i_count, d in enumerate(dirs):
    img0 = (path + d + '/frame_00.png')
    img1 = (path + d + '/frame_02.png')
    gt = (path + d + '/frame_01_gt.png')
    img0 = (torch.tensor(cv2.imread(img0).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    img1 = (torch.tensor(cv2.imread(img1).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)
    gt = (torch.tensor(cv2.imread(gt).transpose(2, 0, 1) / 255.)).to(device).float().unsqueeze(0)

    if count_time == False:
        pred, sparsity, space_check, mid_c= model.inference(img0, img1)
        pred = pred[0]
        ssim = ssim_matlab(gt, torch.round(pred * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
        out = pred.detach().cpu().numpy().transpose(1, 2, 0)
        out = np.round(out * 255) / 255.
        gt = gt[0].cpu().numpy().transpose(1, 2, 0)
        psnr = -10 * math.log10(((gt - out) * (gt - out)).mean())
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print("Avg PSNR: {} SSIM: {}".format(np.mean(psnr_list), np.mean(ssim_list)))
        space_check_8_l.append(torch.mean(space_check[0].view(-1)).detach().cpu().numpy())
        space_check_4_l.append(torch.mean(space_check[1].view(-1)).detach().cpu().numpy())
        space_check_2_l.append(torch.mean(space_check[2].view(-1)).detach().cpu().numpy())
        space_0 = np.mean(space_check_8_l)
        space_1 = np.mean(space_check_4_l)
        space_2 = np.mean(space_check_2_l)

        conv_flops_0 = 0.4460544
        conv1x1_flops_0 = 0.070340608
        conv_mask_0 = 0.002772992
        flops_0 = (conv_flops_0 * 6 + conv1x1_flops_0) * space_0 + conv_mask_0

        conv_flops_1 = 0.807469056
        conv1x1_flops_1 = 0.131088384
        conv_mask_1 = 0.00503808
        flops_1 = (conv_flops_1 * 6 + conv1x1_flops_1) * space_1 + conv_mask_1

        conv_flops_2 = 1.47456
        conv1x1_flops_2 = 0.8208384
        dconv_flops_2 = 0.839385088
        flops_2 = (conv_flops_2 * 5 + conv1x1_flops_2) * space_2 + dconv_flops_2
        flops = flops_0 + flops_1 + flops_2 + 0.425132032 * 2 + 0.533872896 + 0.08613888 * 2 + 2.161082368
        print('flops', flops)
    else:
        mid, time_lapse = model.inference(img0, img1, count_time=count_time, use_sparsity=use_sparsity)
        if i_count == 0:
            for warm_up in range(200):
                mid, time_lapse = model.inference(img0, img1, count_time=count_time, use_sparsity=use_sparsity)
        time_lapse_list.append(time_lapse)
        print("Time Lapse: {} "
            .format(
            np.mean(time_lapse_list),
        ))
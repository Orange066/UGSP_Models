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
# device = torch.device("cpu")
count_time = False
use_sparsity = True
model = Model(count_time = count_time)
# model.load_model_epoch('train_log', 299) # 0.60 294 best
model.load_model_epoch('train_log', 44)
# model.load_model_pt11('train_log')
model.eval()
model.device()
for m in model.flownet.modules():
    if hasattr(m, '_prepare'):
        m._prepare()

psnr_list = []
ssim_list = []
psnr_c_list = []
ssim_c_list = []
sparsity_list = []
space_check_8_l =[]
space_check_4_l =[]
space_check_2_l =[]

path = '../datasets/middlebury'

name = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale', 'Urban2', 'Urban3', 'Venus', 'Walking']
IE_list = []
flow_sparsity_list = []
time_lapse_list = []
for i_count, i in enumerate(name):
    i0 = cv2.imread(path+'/other-data/{}/frame10.png'.format(i)).transpose(2, 0, 1) / 255.
    i1 = cv2.imread(path+'/other-data/{}/frame11.png'.format(i)).transpose(2, 0, 1) / 255.
    gt = cv2.imread(path+'/other-gt-interp/{}/frame10i11.png'.format(i))
    h, w = i0.shape[1], i0.shape[2]
    imgs = torch.zeros([1, 6, 480, 640]).to(device)
    ph = (480 - h) // 2
    pw = (640 - w) // 2
    imgs[:, :3, :h, :w] = torch.from_numpy(i0).unsqueeze(0).float().to(device)
    imgs[:, 3:, :h, :w] = torch.from_numpy(i1).unsqueeze(0).float().to(device)
    I0 = imgs[:, :3]
    I2 = imgs[:, 3:]
    if count_time == False:
        mid, sparsity, space_check, _ = model.inference(I0, I2)
        ssim = ssim_matlab(torch.tensor(gt.transpose(2, 0, 1)).to(device).unsqueeze(0) / 255.,
                           torch.round(mid[0, :, :h, :w].clone() * 255).unsqueeze(0) / 255.).detach().cpu().numpy()

        out = mid[0].clone().detach().cpu().numpy().transpose(1, 2, 0)
        out = np.round(out[:h, :w] * 255)
        pil_img = (out[..., ::-1]).copy()
        pil_img = Image.fromarray(np.uint8(pil_img))
        pil_img.save('./' + i + '.png' )
        IE_list.append(np.abs((out - gt * 1.0)).mean())
        mid = mid[0]

        mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
        mid = mid[:h,:w,:]
        gt = gt / 255.
        psnr = -10 * math.log10(((gt - mid) * (gt - mid)).mean())
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        print("Avg PSNR: {} SSIM: {}" .format(np.mean(psnr_list), np.mean(ssim_list)))
        sparsity = sparsity.detach().cpu().numpy()
        sparsity_list.append(sparsity)
        space_check_8_l.append(torch.mean(space_check[0].view(-1)).detach().cpu().numpy())
        space_check_4_l.append(torch.mean(space_check[1].view(-1)).detach().cpu().numpy())
        space_check_2_l.append(torch.mean(space_check[2].view(-1)).detach().cpu().numpy())

        # print("Sparsity: {} "
        #       "space_check_8_l: {} "
        #       "space_check_4_l: {} "
        #       "space_check_2_l: {} "
        #     .format(
        #     np.mean(sparsity_list),
        #     np.mean(space_check_8_l),
        #     np.mean(space_check_4_l),
        #     np.mean(space_check_2_l),
        # ))
        space_0 = np.mean(space_check_8_l)
        space_1 = np.mean(space_check_4_l)
        space_2 = np.mean(space_check_2_l)

        conv_flops_0 = 2.09088
        conv1x1_flops_0 = 0.3297216
        conv_mask_0 = 0.0129984
        flops_0 = (conv_flops_0 * 6 + conv1x1_flops_0) * space_0 + conv_mask_0

        conv_flops_1 = 3.7850112
        conv1x1_flops_1 = 0.6144768
        conv_mask_1 = 0.023616
        flops_1 = (conv_flops_1 * 6 + conv1x1_flops_1) * space_1 + conv_mask_1

        conv_flops_2 = 6.912
        conv1x1_flops_2 = 3.84768
        dconv_flops_2 = 3.9346176
        flops_2 = (conv_flops_2 * 5 + conv1x1_flops_2) * space_2 + dconv_flops_2
        flops = flops_0 + flops_1 + flops_2 + 1.9928064 * 2 + 2.5025292
        print('flops', flops)
    else:
        mid, time_lapse = model.inference(I0, I2, count_time=count_time, use_sparsity=use_sparsity)
        if i_count == 0:
            for warm_up in range(200):
                mid, time_lapse = model.inference(I0, I2, count_time=count_time, use_sparsity=use_sparsity)
        time_lapse_list.append(time_lapse)
        print("Time Lapse: {} "
            .format(
            np.mean(time_lapse_list),
        ))
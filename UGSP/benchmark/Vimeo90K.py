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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
count_time = False
use_sparsity = True
model = Model(count_time = count_time)
# model.load_model_pt11('train_log')
# model.load_model_epoch('train_log', 284)
model.load_model_epoch('train_log', 284)
# model.load_model_epoch('train_log', 269)
# model.save_model_pt11()
# exit(0)
model.eval()
model.device()

for m in model.flownet.modules():
    if hasattr(m, '_prepare'):
        m._prepare()

path = '../datasets/vimeo_triplet/'
f = open(path + 'tri_testlist.txt', 'r')
psnr_list = []
ssim_list = []
psnr_c_list = []
ssim_c_list = []
sparsity_list = []
space_check_8_l =[]
space_check_4_l =[]
space_check_2_l =[]

time_lapse_list = []
for i_count, i in enumerate(f):

    name = str(i).strip()
    if(len(name) <= 1):
        continue
    I0 = cv2.imread(path + 'sequences/' + name + '/im1.png')
    I1 = cv2.imread(path + 'sequences/' + name + '/im2.png')
    I2 = cv2.imread(path + 'sequences/' + name + '/im3.png')
    I0 = (torch.tensor(I0.transpose(2, 0, 1)).float().to(device) / 255.).unsqueeze(0)
    I2 = (torch.tensor(I2.transpose(2, 0, 1)).float().to(device) / 255.).unsqueeze(0)

    if count_time == False:
        mid, sparsity, space_check, mid_c = model.inference(I0, I2)
        save_path = 'vimeo_compare/' + i + '/'
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        mid_save = np.round((mid[0] * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0)
        cv2.imwrite(os.path.join(save_path, 'compare_ifrnet.png'), mid_save)

        mid = mid[0]
        mid_c = mid_c[0]

        mid = np.round((mid * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
        mid_c = np.round((mid_c * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
        I1 = I1 / 255.
        psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
        psnr_list.append(psnr)
        psnr_c = -10 * math.log10(((I1 - mid_c) * (I1 - mid_c)).mean())
        psnr_c_list.append(psnr_c)
        print(" PSMR: {} ".format( np.mean(psnr_list)))
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


        conv_flops_0 = 0.7805952
        conv1x1_flops_0 = 0.123096064
        conv_mask_0 = 0.004852736
        flops_0 = (conv_flops_0 * 6 + conv1x1_flops_0) * space_0 + conv_mask_0

        conv_flops_1 = 1.413070848
        conv1x1_flops_1 = 0.229404672
        conv_mask_1 = 0.00881664
        flops_1 = (conv_flops_1 * 6 + conv1x1_flops_1) * space_1 + conv_mask_1

        conv_flops_2 = 2.58048
        conv1x1_flops_2 = 1.4364672
        dconv_flops_2 = 1.468923904
        flops_2 = (conv_flops_2 * 5 + conv1x1_flops_2) * space_2 + dconv_flops_2
        flops = flops_0 + flops_1 + flops_2 + 0.743981056 * 2 + 0.934277568
        print('flops', flops)
    else:
        mid, time_lapse = model.inference(I0, I2, count_time=count_time, use_sparsity=use_sparsity)
        if i_count == 0:
            print('here')
            for warm_up in range(200):
                mid, time_lapse = model.inference(I0, I2, count_time=count_time, use_sparsity=use_sparsity)
        time_lapse_list.append(time_lapse)
        print("Time Lapse: {} "
            .format(
            np.mean(time_lapse_list),
        ))

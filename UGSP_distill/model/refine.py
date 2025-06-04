import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import itertools
from model.warplayer import warp
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
        )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
        )
            
class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
c = 16
class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)
    
    def forward(self, x, flow, spa_mask_img):
        # f1
        spa_mask_img_tmp = F.interpolate(spa_mask_img, scale_factor=x.shape[2] / spa_mask_img.shape[2],
                                       mode="bilinear", align_corners=False)
        x = x * spa_mask_img_tmp[:, 1:]
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f1 = warp(x, flow)

        # f2
        spa_mask_img_tmp = F.interpolate(spa_mask_img, scale_factor=x.shape[2] / spa_mask_img.shape[2],
                                       mode="bilinear", align_corners=False)
        x = x * spa_mask_img_tmp[:, 1:]
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f2 = warp(x, flow)

        # f3
        spa_mask_img_tmp = F.interpolate(spa_mask_img, scale_factor=x.shape[2] / spa_mask_img.shape[2],
                                       mode="bilinear", align_corners=False)
        x = x * spa_mask_img_tmp[:, 1:]
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f3 = warp(x, flow)

        # f4
        spa_mask_img_tmp = F.interpolate(spa_mask_img, scale_factor=x.shape[2] / spa_mask_img.shape[2],
                                       mode="bilinear", align_corners=False)
        x = x * spa_mask_img_tmp[:, 1:]
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 0.5
        f4 = warp(x, flow)

        return [f1, f2, f3, f4]
    
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = Conv2(17, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1, spa_mask_union, spa_mask_warp):
        # s0
        spa_mask_union_tmp = F.interpolate(spa_mask_union, scale_factor=img0.shape[2] / spa_mask_union.shape[2],
                                           mode="bilinear", align_corners=False)
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1) * spa_mask_union_tmp)

        # s1
        spa_mask_union_tmp = F.interpolate(spa_mask_union, scale_factor=s0.shape[2] / spa_mask_union.shape[2],
                                           mode="bilinear", align_corners=False)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1) * spa_mask_union_tmp)

        # s2
        spa_mask_warp_tmp = F.interpolate(spa_mask_warp, scale_factor=s1.shape[2] / spa_mask_warp.shape[2],
                                           mode="bilinear", align_corners=False)
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1) * spa_mask_warp_tmp)

        # s3
        spa_mask_warp_tmp = F.interpolate(spa_mask_warp, scale_factor=s2.shape[2] / spa_mask_warp.shape[2],
                                           mode="bilinear", align_corners=False)
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1) * spa_mask_warp_tmp)

        # x
        spa_mask_warp_tmp = F.interpolate(spa_mask_warp, scale_factor=s3.shape[2] / spa_mask_warp.shape[2],
                                          mode="bilinear", align_corners=False)
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1)  * spa_mask_warp_tmp)

        # x
        spa_mask_warp_tmp = F.interpolate(spa_mask_warp, scale_factor=x.shape[2] / spa_mask_warp.shape[2],
                                          mode="bilinear", align_corners=False)
        x = self.up1(torch.cat((x, s2), 1) * spa_mask_warp_tmp)

        # x
        spa_mask_union_tmp = F.interpolate(spa_mask_union, scale_factor=x.shape[2] / spa_mask_union.shape[2],
                                           mode="bilinear", align_corners=False)
        x = self.up2(torch.cat((x, s1), 1) * spa_mask_union_tmp)

        # x
        spa_mask_union_tmp = F.interpolate(spa_mask_union, scale_factor=x.shape[2] / spa_mask_union.shape[2],
                                           mode="bilinear", align_corners=False)
        x = self.up3(torch.cat((x, s0), 1) * spa_mask_union_tmp)

        # x
        spa_mask_union_tmp = F.interpolate(spa_mask_union, scale_factor=x.shape[2] / spa_mask_union.shape[2],
                                           mode="bilinear", align_corners=False)
        x = self.conv(x * spa_mask_union_tmp)


        return torch.sigmoid(x)

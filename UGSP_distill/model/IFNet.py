

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias),
        nn.PReLU(out_planes)
    )

class LocalMaskPredict(nn.Module):
    def __init__(self, in_channels, nf):
        super(LocalMaskPredict, self).__init__()
        self.spa_mask = nn.Sequential(
            nn.Conv2d(in_channels, nf // 4, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(2),
            nn.Conv2d(nf // 4, nf // 4, 3, 1, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf // 4, 2, 3, 2, 1, output_padding=1),
        )
        self.spa_mask = nn.Sequential(*self.spa_mask)

    def forward(self, x, tau=None):
        spa_mask = self.spa_mask(x)
        # spa_mask = F.gumbel_softmax(spa_mask, tau=tau, hard=True, dim=1)
        spa_mask = gumbel_softmax(spa_mask, 1, tau)
        return spa_mask

def resize(x, scale_factor, mode="bilinear"):
    if mode == 'nearest':
        return F.interpolate(x, scale_factor=scale_factor, mode=mode)
    else:
        return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=False)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pyramid1 = nn.Sequential(
            conv(3, 32, 3, 2, 1),
            conv(32, 32, 3, 1, 1)
        )
        self.pyramid2 = nn.Sequential(
            conv(32, 48, 3, 2, 1),
            conv(48, 48, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            conv(48, 72, 3, 2, 1),
            conv(72, 72, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            conv(72, 96, 3, 2, 1),
            conv(96, 96, 3, 1, 1)
        )

    def forward(self, img):
        f1 = self.pyramid1(img)
        f2 = self.pyramid2(f1)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return f4, f3, f2, f1

class GLBlock_First(nn.Module):
    def __init__(self, nf, in_nf, n_layers):
        super(GLBlock_First, self).__init__()

        self.nf = nf
        self.in_nf = in_nf
        self.n_layers = n_layers
        self.conv_head = conv(in_nf, in_nf, 3, 1, 1, bias=True)

        self.convblock_l = nn.ModuleList()
        self.prelu_l = nn.ModuleList()
        for i in range(self.n_layers):
            self.convblock_l.append(nn.Conv2d(in_nf, in_nf, kernel_size=3, stride=1, padding=1, bias=True))
            self.prelu_l.append(nn.PReLU(in_nf))

        # self.conv_last = nn.Conv2d(in_nf * (self.n_layers + 1), 4 + 2 + nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_last = nn.Conv2d(in_nf * (self.n_layers + 1), 4 + nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_mask = nn.Conv2d(nf, 2, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, inputs, tau):
        feat_0, feat_1 = inputs

        x = torch.cat([feat_0, feat_1], dim=1)
        x = self.conv_head(x)
        x_clone = x.clone()
        out_l = [x]

        for i in range(self.n_layers):
            x = self.convblock_l[i](x)
            x = self.prelu_l[i](x)
            out_l.append(x)

        x = self.conv_last(torch.cat(out_l, 1))
        space_mask_next = self.conv_mask(x[:, 4:])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        space_mask_next = F.interpolate(space_mask_next, scale_factor=2, mode="bilinear", align_corners=False)

        feat_t = x[:, 4:]
        flow = x[:, :4]
        # space_mask_next = x[:, 4:6]
        if self.training == True:
            space_mask_next = gumbel_softmax(space_mask_next, 1, tau)[:, :1, ...]
        else:
            space_mask_next = (space_mask_next[:, :1, ...] > space_mask_next[:, 1:, ...]).float()

        channel_mask_check = torch.ones(1, self.in_nf, 1, 1).to(x.device).view(1, -1)

        channel_mask = channel_mask_check.view(1, -1, 1, 1) * torch.ones_like(x_clone[:, :1, ...])
        channel_mask = channel_mask.view(channel_mask.shape[0], -1)

        return flow, feat_t, space_mask_next, channel_mask, [channel_mask_check]

class GLBlock(nn.Module):
    def __init__(self, nf, in_nf, n_layers):
        super(GLBlock, self).__init__()

        self.nf = nf
        self.n_layers = n_layers

        self.conv_head = conv(in_nf, in_nf, 3, 1, 1, bias=False)

        self.convblock_l = nn.ModuleList()
        self.prelu_l = nn.ModuleList()
        for i in range(self.n_layers):
            self.convblock_l.append(nn.Conv2d(in_nf, in_nf, kernel_size=3, stride=1 , padding=1, bias=False))
            self.prelu_l.append(nn.PReLU(in_nf))

        # self.conv_last = nn.Conv2d(in_nf * (self.n_layers + 1), 4 + 2 + nf, kernel_size=1, stride=1, padding=0, bias=True)

        self.conv_last = nn.Conv2d(in_nf * (self.n_layers + 1), 4 + nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_mask = nn.Conv2d(nf, 2, kernel_size=3, stride=1, padding=1, bias=True)

    def get_channel_mask(self):

        return self.channel_mask.softmax(3).round()

    def forward(self, inputs, tau):
        feat_0, feat_1, feat_dense, flow, space_mask, feat_dense_c, flow_c = inputs

        feat_0_warp = warp(feat_0, flow[:, :2])
        feat_1_warp = warp(feat_1, flow[:, 2:4])
        x = torch.cat([feat_0_warp, feat_1_warp, feat_dense, flow], dim=1)
        x_c = None
        if feat_dense_c is not None:
            feat_0_warp = warp(feat_0, flow_c[:, :2])
            feat_1_warp = warp(feat_1, flow_c[:, 2:4])
            x_c = torch.cat([feat_0_warp, feat_1_warp, feat_dense, flow], dim=1)
        x_clone = x.clone()

        space_mask_up = resize(space_mask, 2, 'nearest')
        x = self.conv_head(x)
        if x_c is None:
            x_c = x.clone()
        else:
            x_c = self.conv_head(x_c)
        x = x * space_mask
        out_l = [x]
        out_c_l = [x_c]

        for i in range(self.n_layers):
            x = self.convblock_l[i](x)
            x = x * space_mask
            x = self.prelu_l[i](x)
            out_l.append(x)

            x_c = self.convblock_l[i](x_c)
            x_c = self.prelu_l[i](x_c)
            out_c_l.append(x_c)

        x = self.conv_last(torch.cat(out_l, 1))
        space_mask_next = self.conv_mask(x[:, 4:])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        space_mask_next = F.interpolate(space_mask_next, scale_factor=2, mode="bilinear", align_corners=False)

        feat_t = x[:, 4:]
        flow = x[:, :4]
        # space_mask_next = x[:, 4:6]
        if self.training == True:
            space_mask_next = gumbel_softmax(space_mask_next, 1, tau)[:, :1, ...] * space_mask_up
        else:
            space_mask_next = (space_mask_next[:, :1, ...] > space_mask_next[:, 1:, ...]).float() * space_mask_up

        space_mask_check = space_mask[:, ...].clone()

        sparsity = space_mask * torch.ones_like(x_clone)
        sparsity = sparsity.view(sparsity.shape[0], -1)

        x_c = self.conv_last(torch.cat(out_c_l, 1))
        x_c = F.interpolate(x_c, scale_factor=2, mode="bilinear", align_corners=False)
        feat_t_c = x_c[:, 4:]
        flow_c = x_c[:, :4]

        return flow, feat_t, space_mask_next, sparsity, [space_mask_check], flow_c, feat_t_c

class GLBlock_Final(nn.Module):
    def __init__(self, nf, in_nf, n_layers):
        super(GLBlock_Final, self).__init__()

        self.nf = nf
        self.n_layers = n_layers
        self.conv_head = conv(in_nf, in_nf, 3, 1, 1, bias=False)

        self.convblock_l = nn.ModuleList()
        self.prelu_l = nn.ModuleList()
        for i in range(self.n_layers):
            self.convblock_l.append(nn.Conv2d(in_nf, in_nf, kernel_size=3, stride=1, padding=1, bias=False))
            self.prelu_l.append(nn.PReLU(in_nf))

        self.conv_last = nn.Conv2d(in_nf * (self.n_layers + 1), in_nf, kernel_size=1, stride=1, padding=0, bias=True)
        self.prelu_last = nn.PReLU(in_nf)

        self.dconv_last = nn.ConvTranspose2d(in_nf, 4 + 1 + 3, 4, 2, 1, bias=True)

    def get_channel_mask(self):

        return  self.channel_mask.softmax(3).round()

    def forward(self, inputs, tau):
        feat_0, feat_1, feat_dense, flow, space_mask, feat_dense_c, flow_c = inputs
        feat_0_warp = warp(feat_0, flow[:, :2])
        feat_1_warp = warp(feat_1, flow[:, 2:4])
        x = torch.cat([feat_0_warp, feat_1_warp, feat_dense, flow], dim=1)
        x_clone = x.clone()
        x = self.conv_head(x)
        x = x * space_mask

        out_l = [x]

        feat_0_warp = warp(feat_0, flow_c[:, :2])
        feat_1_warp = warp(feat_1, flow_c[:, 2:4])
        x_c = torch.cat([feat_0_warp, feat_1_warp, feat_dense, flow], dim=1)
        x_c = self.conv_head(x_c)
        out_c_l = [x_c]

        for i in range(self.n_layers):

            x = self.convblock_l[i](x)
            x = x * space_mask
            x = self.prelu_l[i](x)
            out_l.append(x)

            x_c = self.convblock_l[i](x_c)
            x_c = self.prelu_l[i](x_c)
            out_c_l.append(x_c)

        x_feat = self.conv_last(torch.cat(out_l, 1))
        x = self.dconv_last(self.prelu_last(x_feat))

        flow = x[:, :4]
        mask = x[:, 4:5]
        rgb_residual = x[:, 5:]

        space_mask_check = space_mask[:, ...].clone()

        sparsity = space_mask * torch.ones_like(x_clone)
        sparsity = sparsity.view(sparsity.shape[0], -1)

        x_feat_c = self.conv_last(torch.cat(out_c_l, 1))
        x_c = self.dconv_last(self.prelu_last(x_feat_c))
        flow_c = x_c[:, :4]
        mask_c = x_c[:, 4:5]
        rgb_residual_c = x_c[:, 5:]

        return flow, mask, rgb_residual, sparsity, [space_mask_check], flow_c, mask_c, rgb_residual_c, x_feat, x_feat_c

class IFNet(nn.Module):
    def __init__(self, epoch = None):
        super(IFNet, self).__init__()
        self.epoch = epoch
        # nf = 64
        self.context = Encoder()
        # 32 48 72 96
        self.glblock_first = GLBlock_First(nf=72, in_nf=96 * 2, n_layers=5)
        self.glblock_1 = GLBlock(nf=48, in_nf=72 * 2 + 72 + 4, n_layers=5)
        self.glblock_2 = GLBlock(nf=32, in_nf=48 * 2 + 48 + 4, n_layers=5)
        self.glblock_final = GLBlock_Final(nf=None, in_nf=32 * 2 + 32 + 4, n_layers=4)



    def get_channel_mask(self):
        channl_mask_8 = self.glblock_1.get_channel_mask()
        channl_mask_4 = self.glblock_2.get_channel_mask()
        channl_mask_2 = self.glblock_final.get_channel_mask()
        return [channl_mask_8, channl_mask_4, channl_mask_2]

    def forward(self, x, tau=0.01, epoch = None, timestep = None, use_up_sample = False):

        img0 = x[:, :3]
        img1 = x[:, 3:6]
        imgt = x[:, 6:]
        bs, _, h, w = img0.shape
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        imgt = imgt - mean_

        img0_context_l = self.context(img0)
        img1_context_l = self.context(img1)
        imgt_context_l = self.context(imgt)

        check_sapce_mask_l = []
        # scale 16
        flow_0, feat_t_0, space_mask_next, sparsity, check_mask = self.glblock_first([img0_context_l[0], img1_context_l[0]], tau)
        loss_sparsity = sparsity

        # scale 8
        flow_1, feat_t_1, space_mask_next, sparsity, check_mask, flow_1_c, feat_t_1_c = self.glblock_1([img0_context_l[1], img1_context_l[1], feat_t_0, flow_0, space_mask_next, None, None], tau)
        flow_1 = flow_1 + 2.0 * resize(flow_0, scale_factor=2.0)
        flow_1_c = flow_1_c + 2.0 * resize(flow_0, scale_factor=2.0)
        loss_sparsity = torch.cat([loss_sparsity, sparsity], dim=1)

        check_sapce_mask_l.append(check_mask[0])

        # scale 4
        flow_2, feat_t_2, space_mask_next, sparsity, check_mask, flow_2_c, feat_t_2_c = self.glblock_2([img0_context_l[2], img1_context_l[2], feat_t_1, flow_1, space_mask_next, feat_t_1_c, flow_1_c], tau)
        flow_2 = flow_2 + 2.0 * resize(flow_1, scale_factor=2.0)
        flow_2_c = flow_2_c + 2.0 * resize(flow_1_c, scale_factor=2.0)
        loss_sparsity = torch.cat([loss_sparsity, sparsity], dim=1)

        check_sapce_mask_l.append(check_mask[0])

        # scale 2
        flow_3, mask, rgb_residual, sparsity, check_mask, flow_3_c, mask_c, rgb_residual_c, feat_t_3, feat_t_3_c = self.glblock_final([img0_context_l[3], img1_context_l[3], feat_t_2, flow_2, space_mask_next, feat_t_2_c, flow_2_c], tau)
        flow_3 = flow_3 + 2.0 * resize(flow_2, scale_factor=2.0)
        flow_3_c = flow_3_c + 2.0 * resize(flow_2_c, scale_factor=2.0)
        loss_sparsity = torch.cat([loss_sparsity, sparsity], dim=1)

        check_sapce_mask_l.append(check_mask[0])

        warp_0 = warp(img0, flow_3[:, :2])
        warp_1 = warp(img1, flow_3[:, 2:4])
        mask = torch.sigmoid(mask)
        imgt_pred = mask * warp_0 + (1 - mask) * warp_1 + mean_
        imgt_pred = imgt_pred + rgb_residual
        imgt_pred = torch.clamp(imgt_pred, 0, 1)



        warp_0_c = warp(img0, flow_3_c[:, :2])
        warp_1_c = warp(img1, flow_3_c[:, 2:4])
        mask_c = torch.sigmoid(mask_c)
        imgt_pred_c = mask_c * warp_0_c + (1 - mask_c) * warp_1_c + mean_
        imgt_pred_c = imgt_pred_c + rgb_residual_c
        imgt_pred_c = torch.clamp(imgt_pred_c, 0, 1)

        flow_l = [flow_0.clone(), flow_1.clone(), flow_2.clone(), flow_3.clone()]
        flow_c_l = [flow_1_c.clone(), flow_2_c.clone(), flow_3_c.clone()]
        feat_l = [feat_t_1.clone(), feat_t_2.clone(), feat_t_3.clone()]
        feat_c_l = [feat_t_1_c.clone(), feat_t_2_c.clone(), feat_t_3_c.clone()]


        check_mask = [ check_sapce_mask_l]
        gc_loss_l = [feat_t_0.clone(), feat_t_1.clone(), feat_t_2.clone(), imgt_context_l]


        bs, _ = loss_sparsity.shape

        loss_sparsity = loss_sparsity.mean()

        return flow_l, mask, imgt_pred, loss_sparsity, check_mask, flow_c_l, imgt_pred_c, feat_l, feat_c_l, gc_loss_l

    def forward_time(self, x, tau=0.01, epoch = None, timestep = None):

        img0 = x[:, :3]
        img1 = x[:, 3:6]
        imgt = x[:, 6:]
        bs, _, h, w = img0.shape
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        imgt = imgt - mean_

        img0_context_l = self.context(img0)
        img1_context_l = self.context(img1)
        imgt_context_l = self.context(imgt)

        check_sapce_mask_l = []
        # scale 16
        flow_0, feat_t_0, space_mask_next, sparsity, check_mask = self.glblock_first([img0_context_l[0], img1_context_l[0]], tau)
        loss_sparsity = sparsity

        # scale 8
        flow_1, feat_t_1, space_mask_next, sparsity, check_mask, flow_1_c, feat_t_1_c = self.glblock_1([img0_context_l[1], img1_context_l[1], feat_t_0, flow_0, space_mask_next, None, None], tau)
        flow_1 = flow_1 + 2.0 * resize(flow_0, scale_factor=2.0)
        flow_1_c = flow_1_c + 2.0 * resize(flow_0, scale_factor=2.0)
        loss_sparsity = torch.cat([loss_sparsity, sparsity], dim=1)

        check_sapce_mask_l.append(check_mask[0])

        # scale 4
        flow_2, feat_t_2, space_mask_next, sparsity, check_mask, flow_2_c, feat_t_2_c = self.glblock_2([img0_context_l[2], img1_context_l[2], feat_t_1, flow_1, space_mask_next, feat_t_1_c, flow_1_c], tau)
        flow_2 = flow_2 + 2.0 * resize(flow_1, scale_factor=2.0)
        flow_2_c = flow_2_c + 2.0 * resize(flow_1_c, scale_factor=2.0)
        loss_sparsity = torch.cat([loss_sparsity, sparsity], dim=1)

        check_sapce_mask_l.append(check_mask[0])

        # scale 2
        flow_3, mask, rgb_residual, sparsity, check_mask, flow_3_c, mask_c, rgb_residual_c = self.glblock_final([img0_context_l[3], img1_context_l[3], feat_t_2, flow_2, space_mask_next, feat_t_2_c, flow_2_c], tau)
        flow_3 = flow_3 + 2.0 * resize(flow_2, scale_factor=2.0)
        flow_3_c = flow_3_c + 2.0 * resize(flow_2_c, scale_factor=2.0)
        loss_sparsity = torch.cat([loss_sparsity, sparsity], dim=1)

        check_sapce_mask_l.append(check_mask[0])

        warp_0 = warp(img0, flow_3[:, :2])
        warp_1 = warp(img1, flow_3[:, 2:4])
        mask = torch.sigmoid(mask)
        imgt_pred = mask * warp_0 + (1 - mask) * warp_1 + mean_
        imgt_pred = imgt_pred + rgb_residual
        imgt_pred = torch.clamp(imgt_pred, 0, 1)

        warp_0_c = warp(img0, flow_3_c[:, :2])
        warp_1_c = warp(img1, flow_3_c[:, 2:4])
        mask_c = torch.sigmoid(mask_c)
        imgt_pred_c = mask_c * warp_0_c + (1 - mask_c) * warp_1_c + mean_
        imgt_pred_c = imgt_pred_c + rgb_residual_c
        imgt_pred_c = torch.clamp(imgt_pred_c, 0, 1)

        flow_l = [flow_0.clone(), flow_1.clone(), flow_2.clone(), flow_3.clone()]
        flow_c_l = [flow_1_c.clone(), flow_2_c.clone(), flow_3_c.clone()]
        feat_l = [feat_t_1.clone(), feat_t_2.clone()]
        feat_c_l = [feat_t_1_c.clone(), feat_t_2.clone()]


        check_mask = [ check_sapce_mask_l]
        gc_loss_l = [feat_t_0.clone(), feat_t_1.clone(), feat_t_2.clone(), imgt_context_l]


        bs, _ = loss_sparsity.shape

        loss_sparsity = loss_sparsity.mean()


        return flow_l, mask, imgt_pred, loss_sparsity, check_mask, flow_c_l, imgt_pred_c, feat_l, feat_c_l, gc_loss_l


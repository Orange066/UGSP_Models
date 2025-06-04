

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

    def forward_1(self, img):
        f1 = self.pyramid1(img)

        return f1

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

    def forward_tea(self, inputs, tau):
        feat_0, feat_1, feat_dense, flow, space_mask = inputs
        feat_0_warp = warp(feat_0, flow[:, :2])
        feat_1_warp = warp(feat_1, flow[:, 2:4])
        x = torch.cat([feat_0_warp, feat_1_warp, feat_dense, flow], dim=1)
        x = self.conv_head(x)
        out_l = [x]

        for i in range(self.n_layers):

            x = self.convblock_l[i](x)
            x = self.prelu_l[i](x)
            out_l.append(x)


        x = self.dconv_last(self.prelu_last(self.conv_last(torch.cat(out_l, 1))))

        flow = x[:, :4]
        mask = x[:, 4:5]
        rgb_residual = x[:, 5:]

        return flow, mask, rgb_residual

c = 10

class Contextnet(nn.Module):
    def __init__(self):
        super(Contextnet, self).__init__()
        self.conv1 = Conv2(3, c)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x, flow):
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f1 = x
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f2 = x
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f3 = x
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                             recompute_scale_factor=False) * 0.5
        f4 = x
        return [f1, f2, f3, f4]


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.down0 = Conv2(17, 2 * c)
        self.down1 = Conv2(4 * c, 4 * c)
        self.down2 = Conv2(8 * c, 8 * c)
        self.down3 = Conv2(16 * c, 16 * c)
        self.up0 = deconv(32 * c, 8 * c)
        self.up1 = deconv(16 * c, 4 * c)
        self.up2 = deconv(8 * c, 2 * c)
        self.up3 = deconv(4 * c, c)
        self.conv = nn.Conv2d(c, 3, 3, 1, 1)

    def forward(self, img0, img1, warped_img0, warped_img1, mask, flow, c0, c1):
        s0 = self.down0(torch.cat((img0, img1, warped_img0, warped_img1, mask, flow), 1))
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return torch.sigmoid(x)

class IFNet(nn.Module):
    def __init__(self, epoch = None):
        super(IFNet, self).__init__()
        self.epoch = epoch
        self.context = Encoder()
        self.glblock_first = GLBlock_First(nf=72, in_nf=96 * 2, n_layers=5)
        self.glblock_1 = GLBlock(nf=48, in_nf=72 * 2 + 72 + 4, n_layers=5)
        self.glblock_2 = GLBlock(nf=32, in_nf=48 * 2 + 48 + 4, n_layers=5)
        self.glblock_final = GLBlock_Final(nf=None, in_nf=32 * 2 + 32 + 4, n_layers=4)
        self.glblock_final_teacher = GLBlock_Final(nf=None, in_nf=32 * 2 + 32 + 4 + 32, n_layers=4)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def get_channel_mask(self):
        channl_mask_8 = self.glblock_1.get_channel_mask()
        channl_mask_4 = self.glblock_2.get_channel_mask()
        channl_mask_2 = self.glblock_final.get_channel_mask()
        return [channl_mask_8, channl_mask_4, channl_mask_2]

    def forward(self, x, tau=0.01, epoch = None, timestep = None):

        img0 = x[:, :3]
        img1 = x[:, 3:6]
        gt = x[:, 6:]
        bs, _, h, w = img0.shape
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_
        gt = gt - mean_

        img0_context_l = self.context(img0)
        img1_context_l = self.context(img1)
        gt_1 = self.context.forward_1(gt)

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
        flow_2_clone = flow_2.clone()
        flow_2_c_clone = flow_2_c.clone()
        flow_3 = flow_3 + 2.0 * resize(flow_2, scale_factor=2.0)
        flow_3_c = flow_3_c + 2.0 * resize(flow_2_c, scale_factor=2.0)
        loss_sparsity = torch.cat([loss_sparsity, sparsity], dim=1)

        check_sapce_mask_l.append(check_mask[0])

        warp_0 = warp(img0, flow_3[:, :2])
        warp_1 = warp(img1, flow_3[:, 2:4])
        mask = torch.sigmoid(mask)
        imgt_pred = mask * warp_0 + (1 - mask) * warp_1 + mean_
        imgt_pred_clone = imgt_pred.clone()
        c0 = self.contextnet(img0, flow_3[:, :2])
        c1 = self.contextnet(img1, flow_3[:, 2:4])
        tmp = self.unet(img0, img1, warp_0, warp_1, mask, flow_3, c0, c1)
        res = tmp[:, :3] * 2 - 1
        imgt_pred = torch.clamp(imgt_pred + res, 0, 1)

        warp_0_c = warp(img0, flow_3_c[:, :2])
        warp_1_c = warp(img1, flow_3_c[:, 2:4])
        mask_c = torch.sigmoid(mask_c)
        imgt_pred_c = mask_c * warp_0_c + (1 - mask_c) * warp_1_c + mean_
        c0 = self.contextnet(img0, flow_3_c[:, :2])
        c1 = self.contextnet(img1, flow_3_c[:, 2:4])
        tmp = self.unet(img0, img1, warp_0_c, warp_1_c, mask_c, flow_3, c0, c1)
        res = tmp[:, :3] * 2 - 1
        imgt_pred_c = torch.clamp(imgt_pred_c + res, 0, 1)


        flow_l = [flow_0.clone(), flow_1.clone(), flow_2.clone(), flow_3.clone()]
        flow_c_l = [flow_1_c.clone(), flow_2_c.clone(), flow_3_c.clone()]
        feat_l = [feat_t_1.clone(), feat_t_2.clone(), feat_t_3.clone()]
        feat_c_l = [feat_t_1_c.clone(), feat_t_2_c.clone(), feat_t_3_c.clone()]

        check_mask = [ check_sapce_mask_l]


        bs, _ = loss_sparsity.shape
        loss_sparsity = loss_sparsity.mean()

        flow_3_tea, mask_tea, _ = self.glblock_final_teacher.forward_tea([img0_context_l[3], img1_context_l[3], feat_t_2_c, torch.cat([flow_2_c_clone, gt_1], dim=1), None], tau)
        flow_3_tea = flow_3_tea + 2.0 * resize(flow_2_c_clone, scale_factor=2.0)
        warp_0 = warp(img0, flow_3_tea[:, :2])
        warp_1 = warp(img1, flow_3_tea[:, 2:4])
        mask_tea = torch.sigmoid(mask_tea)
        imgt_pred_tea = mask_tea * warp_0 + (1 - mask_tea) * warp_1 + mean_

        loss_mask = ((imgt_pred_clone - gt).abs().mean(1, True) > (imgt_pred_tea - gt).abs().mean(1, True) + 0.01).float().detach()
        loss_distill = (((flow_3_tea.detach() - flow_3) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        loss_distill = loss_distill + (((flow_3_tea.detach() - 2.0 * resize(flow_2, scale_factor=2.0)) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        loss_distill = loss_distill + (((flow_3_tea.detach() - 4.0 * resize(flow_1, scale_factor=4.0)) ** 2).mean(1,
                                                                                                                  True) ** 0.5 * loss_mask).mean()

        return flow_l, mask, imgt_pred, loss_sparsity, check_mask, flow_c_l, imgt_pred_c, feat_l, feat_c_l, loss_distill, flow_3_tea, imgt_pred_tea



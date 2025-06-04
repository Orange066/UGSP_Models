

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

        return flow, feat_t, space_mask_next

class GLBlock(nn.Module):
    def __init__(self, nf, in_nf, n_layers):
        super(GLBlock, self).__init__()

        self.nf = nf
        self.in_nf = in_nf
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

    def _prepare(self):

        # number of channels
        self.d_in_num = []
        self.s_in_num = []
        self.d_out_num = []
        self.s_out_num = []

        for i in range(self.n_layers+1):
            if i == 0:
                self.d_in_num.append(self.in_nf)
                self.s_in_num.append(0)
                self.d_out_num.append(0)
                self.s_out_num.append(self.in_nf)
            else:
                self.d_in_num.append(0)
                self.s_in_num.append(self.in_nf)
                self.d_out_num.append(0)
                self.s_out_num.append(self.in_nf)

        # print('self.d_out_num',self.d_out_num)
        # kernel split
        kernel = []
        prelu = []

        for i in range(self.n_layers + 1):
            if i == 0:
                kernel.append(self.conv_head[0].weight.view(self.s_out_num[i], -1))
                prelu.append(self.conv_head[1].weight)
            else:
                kernel.append(self.convblock_l[i - 1].weight.view(self.s_out_num[i], -1))
                prelu.append(self.prelu_l[i-1].weight)

        # the last 1x1 conv
        # nn.Parameter(torch.rand(1, out_channels, n_layers, 2))
        self.d_in_num.append(0)
        self.s_in_num.append(self.in_nf * (self.n_layers + 1))
        self.d_out_num.append(self.nf + 4)
        self.s_out_num.append(0)

        kernel.append(self.conv_last.weight.squeeze())

        self.kernel = kernel
        self.prelu = prelu
        self.bias = self.conv_last.bias

    def _generate_indices(self):

        A = torch.arange(3).to(self.spa_mask.device).view(-1, 1, 1)
        mask_indices = torch.nonzero(self.spa_mask.squeeze())  # Z, 2

        # indices: dense to sparse (1x1)
        self.h_idx_1x1 = mask_indices[:, 0]
        self.w_idx_1x1 = mask_indices[:, 1]

        # indices: dense to sparse (3x3)
        mask_indices_repeat = mask_indices.unsqueeze(0).repeat([3, 1, 1]) + A # 3, Z, 2

        self.h_idx_3x3 = mask_indices_repeat[..., 0].repeat(1, 3).view(-1)  # 3, Z * 3
        self.w_idx_3x3 = mask_indices_repeat[..., 1].repeat(3, 1).view(-1)  # 3 * 3, Z

        # indices: sparse to sparse (3x3)
        indices = torch.arange(float(mask_indices.size(0))).view(1, -1).to(self.spa_mask.device) + 1
        self.spa_mask[0, 0, self.h_idx_1x1, self.w_idx_1x1] = indices

        self.idx_s2s = F.pad(self.spa_mask, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9, -1).long()

    def _mask_select(self, x, k):
        if k == 1:
            return x[0, :, self.h_idx_1x1, self.w_idx_1x1]
        if k == 3:
            return F.pad(x, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9 * x.size(1), -1)

    def _sparse_conv(self, fea, k, index):
        '''
        :param fea_dense: (B, C_d, H, W)
        :param fea_sparse: (C_s, N)
        :param k: kernel size
        :param index: layer index
        '''
        # dense input
        if self.d_in_num[index] > 0:
            if self.s_out_num[index] > 0:
                # dense to sparse
                fea_d2s = torch.mm(self.kernel[index], self._mask_select(fea, k))

        # sparse input
        if self.s_in_num[index] > 0:
            # sparse to dense & sparse
            if k == 1:
                fea_s2ds = torch.mm(self.kernel[index], fea)
            else:
                fea_s2ds = torch.mm(self.kernel[index], F.pad(fea, [1,0,0,0])[:, self.idx_s2s].view(self.s_in_num[index] * k * k, -1))

        # fusion
        if self.d_out_num[index] > 0:
            fea_s = torch.zeros_like(self.spa_mask).repeat([1, self.d_out_num[index], 1, 1])
            fea_s[0, :, self.h_idx_1x1, self.w_idx_1x1] = fea_s2ds
            if index < len(self.prelu):
                fea_s = F.prelu(fea_s, self.prelu[index])


        if self.s_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                fea_s = fea_d2s
            else:
                fea_s = fea_s2ds
            if index < len(self.prelu):
                fea_s = F.prelu(fea_s.permute(1, 0), self.prelu[index])
                fea_s = fea_s.permute(1, 0)

        if index == len(self.prelu):
            fea_s += self.bias.view(1, -1, 1, 1)

        return fea_s

    def forward(self, inputs, tau, use_sparsity):
        feat_0, feat_1, feat_dense, flow, space_mask = inputs

        feat_0_warp = warp(feat_0, flow[:, :2])
        feat_1_warp = warp(feat_1, flow[:, 2:4])

        x = torch.cat([feat_0_warp, feat_1_warp, feat_dense, flow], dim=1)

        space_mask_up = resize(space_mask, 2, 'nearest')
        if use_sparsity == False:
            x = self.conv_head(x)
            x = x * space_mask
            out_l = [x]

            for i in range(self.n_layers):
                x = self.convblock_l[i](x)

                x = x * space_mask

                x = self.prelu_l[i](x)
                out_l.append(x)

            x = self.conv_last(torch.cat(out_l, 1))

        else:
            self.spa_mask = space_mask.clone()
            self.spa_mask = self.spa_mask.round()
            self._generate_indices()

            # sparse conv
            fea = x
            fea_l = []
            for i in range(self.n_layers + 1):
                fea = self._sparse_conv(fea, k=3, index=i)
                fea_l.append(fea)
            fea = torch.cat(fea_l, 0)
            x = self._sparse_conv(fea, k=1, index=self.n_layers + 1)

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


        return flow, feat_t, space_mask_next

class GLBlock_Final(nn.Module):
    def __init__(self, nf, in_nf, n_layers):
        super(GLBlock_Final, self).__init__()

        self.nf = nf
        self.in_nf = in_nf
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

    def _prepare(self):

        # number of channels
        self.d_in_num = []
        self.s_in_num = []
        self.d_out_num = []
        self.s_out_num = []

        for i in range(self.n_layers+1):
            if i == 0:
                self.d_in_num.append(self.in_nf)
                self.s_in_num.append(0)
                self.d_out_num.append(0)
                self.s_out_num.append(self.in_nf)
            else:
                self.d_in_num.append(0)
                self.s_in_num.append(self.in_nf)
                self.d_out_num.append(0)
                self.s_out_num.append(self.in_nf)

        # print('self.d_out_num',self.d_out_num)
        # kernel split
        kernel = []
        prelu = []

        for i in range(self.n_layers + 1):
            if i == 0:
                kernel.append(self.conv_head[0].weight.view(self.s_out_num[i], -1))
                prelu.append(self.conv_head[1].weight)
            else:
                kernel.append(self.convblock_l[i - 1].weight.view(self.s_out_num[i], -1))
                prelu.append(self.prelu_l[i-1].weight)

        # the last 1x1 conv
        # nn.Parameter(torch.rand(1, out_channels, n_layers, 2))
        self.d_in_num.append(0)
        self.s_in_num.append(self.in_nf * (self.n_layers + 1))
        self.d_out_num.append(self.in_nf)
        self.s_out_num.append(0)

        kernel.append(self.conv_last.weight.squeeze())

        self.kernel = kernel
        self.prelu = prelu
        self.bias = self.conv_last.bias

    def _generate_indices(self):

        A = torch.arange(3).to(self.spa_mask.device).view(-1, 1, 1)
        mask_indices = torch.nonzero(self.spa_mask.squeeze())  # Z, 2

        # indices: dense to sparse (1x1)
        self.h_idx_1x1 = mask_indices[:, 0]
        self.w_idx_1x1 = mask_indices[:, 1]

        # indices: dense to sparse (3x3)
        mask_indices_repeat = mask_indices.unsqueeze(0).repeat([3, 1, 1]) + A # 3, Z, 2

        self.h_idx_3x3 = mask_indices_repeat[..., 0].repeat(1, 3).view(-1)  # 3, Z * 3
        self.w_idx_3x3 = mask_indices_repeat[..., 1].repeat(3, 1).view(-1)  # 3 * 3, Z

        # indices: sparse to sparse (3x3)
        indices = torch.arange(float(mask_indices.size(0))).view(1, -1).to(self.spa_mask.device) + 1
        self.spa_mask[0, 0, self.h_idx_1x1, self.w_idx_1x1] = indices

        self.idx_s2s = F.pad(self.spa_mask, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9, -1).long()

    def _mask_select(self, x, k):
        if k == 1:
            return x[0, :, self.h_idx_1x1, self.w_idx_1x1]
        if k == 3:
            return F.pad(x, [1, 1, 1, 1])[0, :, self.h_idx_3x3, self.w_idx_3x3].view(9 * x.size(1), -1)

    def _sparse_conv(self, fea, k, index):
        '''
        :param fea_dense: (B, C_d, H, W)
        :param fea_sparse: (C_s, N)
        :param k: kernel size
        :param index: layer index
        '''
        # dense input
        if self.d_in_num[index] > 0:
            if self.s_out_num[index] > 0:
                # dense to sparse
                fea_d2s = torch.mm(self.kernel[index], self._mask_select(fea, k))

        # sparse input
        if self.s_in_num[index] > 0:
            # sparse to dense & sparse
            if k == 1:
                fea_s2ds = torch.mm(self.kernel[index], fea)
            else:
                fea_s2ds = torch.mm(self.kernel[index], F.pad(fea, [1,0,0,0])[:, self.idx_s2s].view(self.s_in_num[index] * k * k, -1))

        # fusion
        if self.d_out_num[index] > 0:
            fea_s = torch.zeros_like(self.spa_mask).repeat([1, self.d_out_num[index], 1, 1])
            fea_s[0, :, self.h_idx_1x1, self.w_idx_1x1] = fea_s2ds
            if index < len(self.prelu):
                fea_s = F.prelu(fea_s, self.prelu[index])


        if self.s_out_num[index] > 0:
            if self.d_in_num[index] > 0:
                fea_s = fea_d2s
            else:
                fea_s = fea_s2ds
            if index < len(self.prelu):
                fea_s = F.prelu(fea_s.permute(1, 0), self.prelu[index])
                fea_s = fea_s.permute(1, 0)

        if index == len(self.prelu):
            fea_s += self.bias.view(1, -1, 1, 1)

        return fea_s

    def forward(self, inputs, tau, use_sparsity):
        feat_0, feat_1, feat_dense, flow, space_mask = inputs
        feat_0_warp = warp(feat_0, flow[:, :2])
        feat_1_warp = warp(feat_1, flow[:, 2:4])
        x = torch.cat([feat_0_warp, feat_1_warp, feat_dense, flow], dim=1)

        if use_sparsity == False:
            x = self.conv_head(x)
            x = x * space_mask

            out_l = [x]

            for i in range(self.n_layers):

                x = self.convblock_l[i](x)
                x = x * space_mask

                x = self.prelu_l[i](x)
                out_l.append(x)

            x = self.conv_last(torch.cat(out_l, 1))
        else:
            self.spa_mask = space_mask.clone()
            self.spa_mask = self.spa_mask.round()
            self._generate_indices()

            # sparse conv
            fea = x
            fea_l = []
            for i in range(self.n_layers + 1):
                fea = self._sparse_conv(fea, k=3, index=i)
                fea_l.append(fea)
            fea = torch.cat(fea_l, 0)
            x = self._sparse_conv(fea, k=1, index=self.n_layers + 1)

        x = self.dconv_last(self.prelu_last(x))

        flow = x[:, :4]
        mask = x[:, 4:5]
        rgb_residual = x[:, 5:]


        return flow, mask, rgb_residual,

class IFNet_time(nn.Module):
    def __init__(self, epoch = None):
        super(IFNet_time, self).__init__()
        self.epoch = epoch
        # nf = 64
        self.context = Encoder()
        # 32 48 72 96
        self.glblock_first = GLBlock_First(nf=72, in_nf=96 * 2, n_layers=5)
        self.glblock_1 = GLBlock(nf=48, in_nf=72 * 2 + 72 + 4, n_layers=5)
        self.glblock_2 = GLBlock(nf=32, in_nf=48 * 2 + 48 + 4, n_layers=5)
        self.glblock_final = GLBlock_Final(nf=None, in_nf=32 * 2 + 32 + 4, n_layers=4)

        self.print_parameters()

    def print_parameters(self):

        total_params = sum(p.numel() for p in self.context.parameters())
        print(f'{total_params:,} self.context parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.context.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        total_params = sum(p.numel() for p in self.glblock_first.parameters())
        print(f'{total_params:,} self.glblock_first parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.glblock_first.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        total_params = sum(p.numel() for p in self.glblock_1.parameters())
        print(f'{total_params:,} self.glblock_1 parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.glblock_1.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        total_params = sum(p.numel() for p in self.glblock_2.parameters())
        print(f'{total_params:,} self.glblock_2 parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.glblock_2.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        total_params = sum(p.numel() for p in self.glblock_final.parameters())
        print(f'{total_params:,} self.glblock_final parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.glblock_final.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')


    def forward(self, x, tau=0.01, epoch = None, timestep = None, use_sparsity=False):

        img0 = x[:, :3]
        img1 = x[:, 3:6]
        bs, _, h, w = img0.shape
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_

        img0_context_l = self.context(img0)
        img1_context_l = self.context(img1)

        # scale 16
        flow_0, feat_t_0, space_mask_next = self.glblock_first([img0_context_l[0], img1_context_l[0]], tau)


        # scale 8
        flow_1, feat_t_1, space_mask_next = self.glblock_1([img0_context_l[1], img1_context_l[1], feat_t_0, flow_0, space_mask_next], tau, use_sparsity)
        flow_1 = flow_1 + 2.0 * resize(flow_0, scale_factor=2.0)


        # scale 4
        flow_2, feat_t_2, space_mask_next = self.glblock_2([img0_context_l[2], img1_context_l[2], feat_t_1, flow_1, space_mask_next], tau, use_sparsity)
        flow_2 = flow_2 + 2.0 * resize(flow_1, scale_factor=2.0)


        # scale 2
        flow_3, mask, rgb_residual = self.glblock_final([img0_context_l[3], img1_context_l[3], feat_t_2, flow_2, space_mask_next], tau, use_sparsity)
        flow_3 = flow_3 + 2.0 * resize(flow_2, scale_factor=2.0)


        warp_0 = warp(img0, flow_3[:, :2])
        warp_1 = warp(img1, flow_3[:, 2:4])
        mask = torch.sigmoid(mask)
        imgt_pred = mask * warp_0 + (1 - mask) * warp_1 + mean_
        imgt_pred = imgt_pred + rgb_residual
        imgt_pred = torch.clamp(imgt_pred, 0, 1)


        return imgt_pred


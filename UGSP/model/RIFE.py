import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from model.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.IFNet import *
from model.IFNet_time import *
import torch.nn.functional as F
from model.loss import *
from model.laplacian import *
from model.refine import *
import math
device = torch.device("cuda")
# device = torch.device("cpu")
import cv2 as cv2
import time as time


class Ternary_ifr(nn.Module):
    def __init__(self, patch_size=7):
        super(Ternary_ifr, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, tensor):
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size // 2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss

class Geometry(nn.Module):
    def __init__(self, patch_size=3):
        super(Geometry, self).__init__()
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float().to(device)

    def transform(self, tensor):
        b, c, h, w = tensor.size()
        tensor_ = tensor.reshape(b*c, 1, h, w)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size//2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_ = loc_diff.reshape(b, c*(self.patch_size**2), h, w)
        loc_diff_norm = loc_diff_ / torch.sqrt(0.81 + loc_diff_ ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size//2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def forward(self, x, y):
        loc_diff_x = self.transform(x)
        loc_diff_y = self.transform(y)
        diff = loc_diff_x - loc_diff_y
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(x)
        loss = (dist * mask).mean()
        return loss

class Charbonnier_Ada(nn.Module):
    def __init__(self):
        super(Charbonnier_Ada, self).__init__()

    def forward(self, diff, weight):
        alpha = weight / 2
        epsilon = 10 ** (-(10 * weight - 1) / 3)
        loss = ((diff ** 2 + epsilon ** 2) ** alpha).mean()
        return loss

class Model:
    def __init__(self, local_rank=-1, arbitrary=False, epoch = None, count_time = False):
        if arbitrary == True:
            self.flownet = IFNet_m()
        else:
            if count_time == False:
                self.flownet = IFNet(epoch)
            else:
                self.flownet = IFNet_time(epoch)
            # total_params = sum(p.numel() for p in self.flownet.parameters())
            # print(f'{total_params:,} total parameters.')
            # total_trainable_params = sum(
            #     p.numel() for p in self.flownet.parameters() if p.requires_grad)
            # print(f'{total_trainable_params:,} training parameters.')


        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.epe = EPE()
        self.lap = LapLoss()
        self.sobel = SOBEL()
        self.gc_loss = Geometry(3)
        self.rb_loss = Charbonnier_Ada()
        self.tr_loss = Ternary_ifr(7)
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def load_model_u(self, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.flownet_u.load_state_dict(convert(torch.load('/home/user3/ECCV2022-RIFE-21/train_log/flownet.pkl')))

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }
            
        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location=device)))

    def load_model_pt11(self, path, rank=0):
        self.flownet.load_state_dict(torch.load('{}/flownet_11.pkl'.format(path), map_location=device))

    def load_model_module(self, path, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.flownet.load_state_dict(torch.load('{}/flownet.pkl'.format(path)))

    def load_model_state(self, state, rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.flownet.load_state_dict(state)

    def load_model_epoch(self, path, epoch,rank=0):
        def convert(param):
            return {
                k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k
            }

        if rank <= 0:
            self.flownet.load_state_dict(convert(torch.load('{}/flownet_{}.pkl'.format(path, epoch), map_location=device)))

    def save_model(self, path, rank=0, epoch=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))
            state_dict = {
                "epoch": epoch,
                "state_dict": self.flownet.state_dict(),
            }
            torch.save(state_dict, '{}/flownet_state.pkl'.format(path))

    def save_model_pt11(self):
        torch.save(self.flownet.state_dict(),'./train_log/flownet_11.pkl',_use_new_zipfile_serialization=False)

    def save_model_epoch(self, path, rank=0, epoch=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet_{}.pkl'.format(path, epoch))
            state_dict = {
                "epoch": epoch,
                "state_dict": self.flownet.state_dict(),
            }
            torch.save(state_dict, '{}/flownet_state_{}.pkl'.format(path, epoch))

    def save_model_best(self, path, rank=0, epoch=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet_best.pkl'.format(path))
            state_dict = {
                "epoch": epoch,
                "state_dict": self.flownet.state_dict(),
            }
            torch.save(state_dict, '{}/flownet_state_best.pkl'.format(path))

    def get_channel_mask(self):
        return self.flownet.get_channel_mask()

    def inference(self, img0, img1, scale_list=[4, 2, 1], TTA=False, timestep=0.5, count_time = False, use_sparsity=False):
        if count_time == False:
            imgs = torch.cat((img0, img1), 1)
            flow_l, mask, merged, loss_sparsity, check_mask, flow_c_l, imgt_pred_c, feat_l, feat_c_l = self.flownet(imgs, timestep=timestep, tau=0.01)
            if TTA == False:
                return merged, loss_sparsity, check_mask[-1], imgt_pred_c
            else:
                flow2, mask2, merged2, loss_space_mask = self.flownet(imgs.flip(2).flip(3), timestep=timestep)
                return (merged + merged2[-1].flip(2).flip(3)) / 2
        else:
            imgs = torch.cat((img0, img1), 1)
            time_start = time.time()
            merged = self.flownet(imgs, timestep=timestep, tau=0.01, use_sparsity=use_sparsity)
            time_end = time.time()
            return merged, time_end - time_start

    def pad(self, im, pad_width):
        h, w = im.shape[-2:]
        mh = h % pad_width
        ph = 0 if mh == 0 else pad_width - mh
        mw = w % pad_width
        pw = 0 if mw == 0 else pad_width - mw
        shape = [s for s in im.shape]
        shape[-2] += ph
        shape[-1] += pw
        im_p = torch.zeros(shape).float()
        im_p = im_p.to(im.device)
        im_p[..., :h, :w] = im
        im = im_p
        return im

    def get_robust_weight(self, flow_pred, flow_gt, beta):
        epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=1, keepdim=True) ** 0.5
        robust_weight = torch.exp(-beta * epe)
        return robust_weight

    def update(self, imgs, gt, space_mask_gpu, var_gpu, learning_rate=0, mul=1, training=True, flow_gt=None, epoch = None, tau = None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()

        flow_l, mask, merged, loss_sparsity, check_mask, flow_c_l, merged_c, feat_l, feat_c_l = self.flownet(torch.cat((imgs, gt), 1), tau=tau, epoch = epoch)

        # uncertainty loss
        # b, c, h, w = var_gpu.shape
        # s1 = var_gpu.clone().view(b, c, -1)
        # pmin = torch.min(s1, dim=-1)
        # pmin = pmin[0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        # s = var_gpu.clone()
        # s = s - pmin + 1
        # # s = s.detach()
        # # merged_c_ = torch.mul(merged_c, s)
        # merged_ = torch.mul(merged, s)
        # merged_c_ = torch.mul(merged_c, s)
        # gt_ = torch.mul(gt, s)
        loss_l1 = (self.lap(merged, gt)).mean()
        loss_l1_c = (self.lap(merged_c, gt)).mean()

        # loss_l1 = (self.lap(merged, gt)).mean()
        # loss_l1_c = (self.lap(merged_c, gt)).mean()
        # loss_contrast_pixel = (self.tr_loss(merged, merged_c.detach()))
        loss_contrast_feat = self.gc_loss(feat_l[0], feat_c_l[0].detach()) + self.gc_loss(feat_l[1], feat_c_l[1].detach()) + self.gc_loss(feat_l[2], feat_c_l[2].detach())
        # loss_contrast_pixel = (self.tr_loss(merged, merged_c.detach()))
        # loss_contrast_feat = (self.gc_loss(feat_l[0], feat_c_l[0].detach()) + self.gc_loss(feat_l[1], feat_c_l[1].detach()))



        # loss_space_l1 = (self.lap(merged_c_, gt_)).mean()

        # uncertainty guide
        w_step = max(min(1 - (epoch - 200) / 100, 1) * .5, .1)
        if training:
            self.optimG.zero_grad()
            if epoch is not None and epoch < 1:
                loss_space_mask_w = 0.1
            else:
                loss_space_mask_w = 0.1
            lambda_sparsity = min((epoch - 1) / 15, 1) * loss_space_mask_w


            local_mask_l = check_mask[0]
            _, _, h, w = merged.shape
            loss_local_mask_label = 0.0
            for i, local_mask in enumerate(local_mask_l):
                _, _, h_tmp, w_tmp = local_mask.shape
                gt_mask_tmp = F.interpolate(space_mask_gpu[:, i:i+1], scale_factor=h_tmp / h, mode="nearest")
                # gt_mask_one = gt_mask_tmp > 0.5
                # loss_local_mask_label = torch.abs(gt_mask_tmp[gt_mask_one] - local_mask[gt_mask_one]).mean()
                loss_local_mask_label = loss_local_mask_label + torch.abs(gt_mask_tmp - local_mask).mean()
            # loss_local_mask_label = loss_local_mask_label / len(local_mask_l)
            loss_local_mask_label = loss_local_mask_label / 10
            loss_G = loss_l1 * 1. + loss_l1_c * 0.5 + 0.001 * loss_contrast_feat   \
                     + (torch.abs(loss_sparsity - 0.35) * .1 + loss_local_mask_label * w_step ) * lambda_sparsity

            # loss_G = loss_l1 + (loss_global_mask * 0.005 + loss_local_mask * 0.01 + loss_local_mask_label * 0.01) * lambda_sparsity

            loss_G.backward()
            for name, param in self.flownet.named_parameters():
                if param.grad is None:
                    print(name)
            self.optimG.step()
        else:

            local_mask_l = check_mask[0]
            _, _, h, w = merged.shape
            for i, local_mask in enumerate(local_mask_l):
                _, _, h_tmp, w_tmp = local_mask.shape
                gt_mask_tmp = F.interpolate(space_mask_gpu[:, i:i + 1], scale_factor=h_tmp / h, mode="nearest")
                # gt_mask_one = gt_mask_tmp > 0.5
                # loss_local_mask_label = torch.abs(gt_mask_tmp[gt_mask_one] - local_mask[gt_mask_one]).mean()
                loss_local_mask_label = torch.abs(gt_mask_tmp - local_mask).mean()
            loss_local_mask_label = loss_local_mask_label / len(local_mask_l)
        return merged, merged_c, {
            'mask': mask,
            'mask_tea': mask,
            'flow': flow_l,
            'flow_c': flow_c_l,
            'loss_l1': loss_l1,
            'loss_l1_c': loss_l1_c,
            'loss_sparsity': loss_sparsity,
            'loss_contrast_feat': loss_contrast_feat,
            # 'loss_contrast_pixel': loss_contrast_pixel,
            'loss_local_mask_label': loss_local_mask_label,
            'check_mask': check_mask,
            'w_step': w_step,
            'error': torch.abs(torch.mean(merged, dim=1, keepdim=True) - torch.mean(gt, dim=1, keepdim=True)),
            }

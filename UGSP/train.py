import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse

from model.RIFE import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")

log_path = 'train_log'


def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 1e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (1e-4 - 1e-5) * mul + 1e-5


def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def visual_feat_visualize(feats):
    # feature_depth = feature_depth.detach().cpu().numpy()
    feat_l = []
    for i in range(feats.shape[0]):
        feature = feats[i]
        depth_min = np.min(feature)
        depth_max = np.max(feature)
        # print('min', np.min(feature_depth))
        # print('max', np.max(feature_depth))
        feature = (feature - depth_min) / (depth_max - depth_min)
        feature = np.uint8(feature * 255)
        feature = cv2.applyColorMap(feature, cv2.COLORMAP_VIRIDIS)
        feat_l.append(feature)
        # cv2.imwrite(save_path, visualize)
    feat_l = np.stack(feat_l, axis=0)
    return feat_l


def train(model, local_rank, start_epoch):
    # if local_rank == 0:
    writer = SummaryWriter('train')
    writer_val = SummaryWriter('validate')
    nr_eval = 0
    dataset = VimeoDataset('train')
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True,
                            sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    step = args.step_per_epoch * start_epoch
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    total_epoch = args.epoch
    best_psnr = 0.0
    for epoch in range(start_epoch, args.epoch):
        sampler.set_epoch(epoch)
        # tau = 1 - (epoch / total_epoch) * 0.99
        tau = max(1 - (epoch) / 150, 0.4)
        # tau = 0.4
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep, space_mask_gpu, var_gpu = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            space_mask_gpu = space_mask_gpu.to(device, non_blocking=True) / 255.

            var_gpu = var_gpu.to(device, non_blocking=True)
            # space_mask = space_mask.to(device, non_blocking=True)
            # var = var.to(device, non_blocking=True)
            timestep = timestep.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step) * args.world_size / 4
            pred, pred_c, info = model.update(imgs, gt, space_mask_gpu, var_gpu, learning_rate, training=True, epoch=epoch,
                                      tau=tau)  # pass timestep if you are training RIFEm
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/loss_sparsity', info['loss_sparsity'], step)
                writer.add_scalar('loss/local_mask_label', info['loss_local_mask_label'], step)
                writer.add_scalar('loss/loss_contrast_feat', info['loss_contrast_feat'], step)
                # writer.add_scalar('loss/loss_contrast_pixel', info['loss_contrast_pixel'], step)

                writer.add_scalar('loss/check_space_mask_l_scale8', info['check_mask'][0][0].mean(), step)
                writer.add_scalar('loss/check_space_mask_l_scale4', info['check_mask'][0][1].mean(), step)
                writer.add_scalar('loss/check_space_mask_l_scale2', info['check_mask'][0][2].mean(), step)

                # writer.add_scalar('loss/refine_img0_mask', info['loss_refine_spa_mask_img0'], step)
                # writer.add_scalar('loss/refine_img1_mask', info['loss_refine_spa_mask_img1'], step)
                # writer.add_scalar('loss/refine_union_mask', info['loss_refine_spa_mask_union'], step)
                # writer.add_scalar('loss/refine_warp_mask', info['loss_refine_spa_mask_warp'], step)
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = ((info['mask']).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                check_mask_8 = ((info['check_mask'][0][0]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
                    'uint8')
                check_mask_4 = ((info['check_mask'][0][1]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
                    'uint8')
                check_mask_2 = ((info['check_mask'][0][2]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
                    'uint8')
                guide_mask_8 = ((space_mask_gpu[:, 0:1]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
                    'uint8')
                guide_mask_4 = ((space_mask_gpu[:, 1:2]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
                    'uint8')
                guide_mask_2 = ((space_mask_gpu[:, 2:3]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
                    'uint8')
                error = (info['error']).permute(0, 2, 3, 1).detach().cpu().numpy()
                error = visual_feat_visualize(error)
                uncertainty = (
                    torch.mean(var_gpu.permute(0, 2, 3, 1), dim=3, keepdim=True).detach().cpu().numpy())
                uncertainty = visual_feat_visualize(uncertainty)
                # uncertainty_2 = (
                #     torch.mean(info['var'][:, 1].permute(0, 2, 3, 1), dim=3, keepdim=True).detach().cpu().numpy())
                # uncertainty_3 = (
                #     torch.mean(info['var'][:, 2].permute(0, 2, 3, 1), dim=3, keepdim=True).detach().cpu().numpy())
                # uncertainty = visual_feat_visualize(
                #     np.concatenate([uncertainty_3, uncertainty_2, uncertainty_1], axis=2))

                # mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred_c = (pred_c.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flows = info['flow']
                flows_c = info['flow_c']
                # flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(5):
                    imgs = np.concatenate((pred[i], pred_c[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    flow_0_tmp, flow_1_tmp, flow_2_tmp, flow_3_tmp = flows
                    flow_0_tmp = flow_0_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
                    flow_1_tmp = flow_1_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
                    flow_2_tmp = flow_2_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
                    flow_3_tmp = flow_3_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()

                    flow0 = flow2rgb(flow_0_tmp[i, :, :, :2])
                    flow_0_tmp = flow2rgb(flow_0_tmp[i, :, :, 2:4])
                    flow0 = np.concatenate([flow0, flow_0_tmp], axis=1)
                    flow1 = flow2rgb(flow_1_tmp[i, :, :, :2])
                    flow_1_tmp = flow2rgb(flow_1_tmp[i, :, :, 2:4])
                    flow1 = np.concatenate([flow1, flow_1_tmp], axis=1)
                    flow2 = flow2rgb(flow_2_tmp[i, :, :, :2])
                    flow_2_tmp = flow2rgb(flow_2_tmp[i, :, :, 2:4])
                    flow2 = np.concatenate([flow2, flow_2_tmp], axis=1)

                    flow3 = flow2rgb(flow_3_tmp[i, :, :, :2])
                    flow_3_tmp = flow2rgb(flow_3_tmp[i, :, :, 2:4])
                    flow3 = np.concatenate([flow3, flow_3_tmp], axis=1)

                    flow_0_tmp, flow_1_tmp, flow_2_tmp, flow_3_tmp = flows_c
                    flow_0_tmp = flow_0_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
                    flow_1_tmp = flow_1_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
                    flow_2_tmp = flow_2_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
                    flow_3_tmp = flow_3_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()

                    flow0_c = flow2rgb(flow_0_tmp[i, :, :, :2])
                    flow_0_tmp = flow2rgb(flow_0_tmp[i, :, :, 2:4])
                    flow0_c = np.concatenate([flow0_c, flow_0_tmp], axis=1)

                    flow1_c = flow2rgb(flow_1_tmp[i, :, :, :2])
                    flow_1_tmp = flow2rgb(flow_1_tmp[i, :, :, 2:4])
                    flow1_c = np.concatenate([flow1_c, flow_1_tmp], axis=1)

                    flow2_c = flow2rgb(flow_2_tmp[i, :, :, :2])
                    flow_2_tmp = flow2rgb(flow_2_tmp[i, :, :, 2:4])
                    flow2_c = np.concatenate([flow2_c, flow_2_tmp], axis=1)

                    flow3_c = flow2rgb(flow_3_tmp[i, :, :, :2])
                    flow_3_tmp = flow2rgb(flow_3_tmp[i, :, :, 2:4])
                    flow3_c = np.concatenate([flow3_c, flow_3_tmp], axis=1)


                    # writer.add_image(str(i) + '/flow', flow2rgb(flow0[i]), step, dataformats='HWC')

                    writer.add_image(str(i) + '/flow_0', flow0, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow_1', flow1, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow_2', flow2, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow_3', flow3, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow_c_0', flow0, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow_c_1', flow1, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow_c_2', flow2, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow_c_3', flow3, step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                    writer.add_image(str(i) + '/check_mask_8', check_mask_8[i], step, dataformats='HWC')
                    writer.add_image(str(i) + '/check_mask_4', check_mask_4[i], step, dataformats='HWC')
                    writer.add_image(str(i) + '/check_mask_2', check_mask_2[i], step, dataformats='HWC')
                    writer.add_image(str(i) + '/guide_mask',
                                     np.concatenate([guide_mask_8[i], guide_mask_4[i], guide_mask_2[i]], axis=1), step,
                                     dataformats='HWC')
                    writer.add_image(str(i) + '/error', error[i], step, dataformats='HWC')
                    writer.add_image(str(i) + '/uncertainty', uncertainty[i], step, dataformats='HWC')

                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} lr:{:.4e} tau:{:.2f} w_step:{:.4e} '
                      'loss_l1:{:.4e} '
                      'loss_l1_c:{:.4e} '
                      'loss_sparsity:{:.4e} '
                      'loss_local_mask_label:{:.4e} '
                      'loss_contrast_feat:{:.4e} '
                      # 'loss_contrast_pixel:{:.4e} '

                      .format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, learning_rate,
                              tau, info['w_step'],
                              info['loss_l1'],
                              info['loss_l1_c'],
                              info['loss_sparsity'],
                              info['loss_local_mask_label'],
                              info['loss_contrast_feat'],
                              # info['loss_contrast_pixel'],
                              ))
            step += 1
            # break
        nr_eval += 1
        if nr_eval % 5 == 0:
            psnr = evaluate(model, val_data, step, local_rank, writer_val)
            model.save_model_epoch(log_path, local_rank, epoch)
            print(psnr, psnr>best_psnr)
            if psnr > best_psnr:
                best_psnr = psnr
                model.save_model_best(log_path, local_rank, epoch)
        model.save_model(log_path, local_rank, epoch)
        dist.barrier()


def evaluate(model, val_data, nr_eval, local_rank, writer_val):
    loss_l1_list = []
    loss_sparsity_list = []
    loss_local_mask_label_list = []
    psnr_list = []
    time_stamp = time.time()
    cout = 0
    for i, data in enumerate(val_data):
        if cout == 10:
            break
        cout = cout+1
        data_gpu, timestep, space_mask_gpu, var_gpu = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        space_mask_gpu = space_mask_gpu.to(device, non_blocking=True) / 255.
        var_gpu = var_gpu.to(device, non_blocking=True)
        # space_mask = space_mask.to(device, non_blocking=True)
        # var = var.to(device, non_blocking=True)
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, pred_c, info = model.update(imgs, gt, space_mask_gpu, var_gpu, training=False, epoch=epoch,
                                      tau=0.01)  # pass timestep if you are training RIFEm
        # loss_l1_list.append(info['loss_l1'].cpu().numpy())
        # loss_sparsity_list.append(info['loss_sparsity'].cpu().numpy())
        # loss_local_mask_label_list.append(info['loss_local_mask_label'].cpu().numpy())
        # loss_refine_img0_mask_list.append(info['loss_refine_spa_mask_img0'].cpu().numpy())
        # loss_refine_img1_mask_list.append(info['loss_refine_spa_mask_img1'].cpu().numpy())
        # loss_refine_union_mask_list.append(info['loss_refine_spa_mask_union'].cpu().numpy())
        # loss_refine_warp_mask_list.append(info['loss_refine_spa_mask_warp'].cpu().numpy())
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
        # gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        # # flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        # flows = info['flow']
        # mask = ((info['mask']).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
        # check_mask_8 = ((info['check_mask'][1][0]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
        #     'uint8')
        # check_mask_4 = ((info['check_mask'][1][1]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
        #     'uint8')
        # check_mask_2 = ((info['check_mask'][1][2]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
        #     'uint8')
        # guide_mask_8 = ((info['guide_mask'][:, 0]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
        # guide_mask_4 = ((info['guide_mask'][:, 1]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
        #     'uint8')
        # guide_mask_2 = ((info['guide_mask'][:, 2]).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(
        #     'uint8')
        # error = (info['error']).permute(0, 2, 3, 1).detach().cpu().numpy()
        # error = visual_feat_visualize(error)
        # if i == 0 and local_rank == 0:
        #     for j in range(10):
        #         imgs = np.concatenate((pred[j], gt[j]), 1)[:, :, ::-1]
        #         writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
        #
        #         flow_0_tmp, flow_1_tmp, flow_2_tmp, flow_3_tmp = flows
        #         flow_0_tmp = flow_0_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
        #         flow_1_tmp = flow_1_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
        #         flow_2_tmp = flow_2_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
        #         flow_3_tmp = flow_3_tmp.permute(0, 2, 3, 1).detach().cpu().numpy()
        #         flow0 = flow2rgb(flow_0_tmp[i, :, :, :2])
        #         flow_0_tmp = flow2rgb(flow_0_tmp[i, :, :, 2:4])
        #         flow0 = np.concatenate([flow0, flow_0_tmp], axis=1)
        #         flow1 = flow2rgb(flow_1_tmp[i, :, :, :2])
        #         flow_1_tmp = flow2rgb(flow_1_tmp[i, :, :, 2:4])
        #         flow1 = np.concatenate([flow1, flow_1_tmp], axis=1)
        #         flow2 = flow2rgb(flow_2_tmp[i, :, :, :2])
        #         flow_2_tmp = flow2rgb(flow_2_tmp[i, :, :, 2:4])
        #         flow2 = np.concatenate([flow2, flow_2_tmp], axis=1)
        #         flow3 = flow2rgb(flow_3_tmp[i, :, :, :2])
        #         flow_3_tmp = flow2rgb(flow_3_tmp[i, :, :, 2:4])
        #         flow3 = np.concatenate([flow3, flow_3_tmp], axis=1)
        #
        #         # writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
        #         # writer_val.add_image(str(j) + '/flow', flow0, nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(i) + '/flow_0', flow0, nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(i) + '/flow_1', flow1, nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(i) + '/flow_2', flow2, nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(i) + '/flow_3', flow3, nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(i) + '/mask', mask[i], nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(i) + '/check_mask_8', check_mask_8[i], nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(i) + '/check_mask_4', check_mask_4[i], nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(i) + '/check_mask_2', check_mask_2[i], nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(i) + '/guide_mask',
        #                              np.concatenate([guide_mask_8[i], guide_mask_4[i], guide_mask_2[i]], axis=1),
        #                              nr_eval, dataformats='HWC')
        #         writer_val.add_image(str(i) + '/error', error[i], nr_eval, dataformats='HWC')

    # eval_time_interval = time.time() - time_stamp

    # if local_rank != 0:
    #     return
    # writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    # writer_val.add_scalar('loss_sparsity', np.array(loss_sparsity_list).mean(), nr_eval)
    # writer_val.add_scalar('loss_local_mask_label', np.array(loss_local_mask_label_list).mean(), nr_eval)
    # writer_val.add_scalar('loss_refine_spa_mask_img0', np.array(loss_refine_img0_mask_list).mean(), nr_eval)
    # writer_val.add_scalar('loss_refine_spa_mask_img1', np.array(loss_refine_img1_mask_list).mean(), nr_eval)
    # writer_val.add_scalar('loss_refine_spa_mask_union', np.array(loss_refine_union_mask_list).mean(), nr_eval)
    # writer_val.add_scalar('loss_refine_spa_mask_warp', np.array(loss_refine_warp_mask_list).mean(), nr_eval)
    return np.array(psnr_list).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=8, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(args.local_rank, epoch=args.epoch)

    epoch = 0
    if args.resume == True:
        # model.load_model_module('train_log')
        state = torch.load('/home/user3/ECCV2022-RIFE-31-sparse-l1/train_log/flownet_state_199.pkl')
        epoch = state["epoch"] + 1
        model.load_model_state(state["state_dict"])
        # epoch = 200
    train(model, args.local_rank, epoch)


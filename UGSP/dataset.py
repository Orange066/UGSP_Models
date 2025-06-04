import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
import lmdb
import re

cv2.setNumThreads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def _read_img_lmdb_space(env, key, size):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.float32)
    N, C, H, W = size
    img = img_flat.reshape(N, C, H, W)
    return img

def _read_img_lmdb_var(env, key, size):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.float32)
    B, N, C, H, W = size
    img = img_flat.reshape(B, N, C, H, W)
    return img



class VimeoDataset(Dataset):
    def __init__(self, dataset_name, batch_size=32):
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.h = 256
        self.w = 448
        self.data_root = '/opt/SSD/cr/vimeo_triplet'
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.flow_root = os.path.join(self.data_root, 'flow')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt]
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist
        else:
            self.meta_data = self.trainlist[cnt:]

    def crop(self, img0, gt, img1, space_mask,  var, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x + h, y:y + w, :]
        img1 = img1[x:x + h, y:y + w, :]
        gt = gt[x:x + h, y:y + w, :]
        space_mask = space_mask[x:x + h, y:y + w, :]
        var = var[ x:x + h, y:y + w, :]

        return img0, gt, img1, space_mask, var

    def readPFM(self, file):
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            endian = '<'
            scale = -scale
        else:
            endian = '>'

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale

    def readFlow(self, name):
        if name.endswith('.pfm') or name.endswith('.PFM'):
            return self.readPFM(name)[0][:, :, 0:2]

        f = open(name, 'rb')

        header = f.read(4)
        if header.decode("utf-8") != 'PIEH':
            raise Exception('Flow file header does not contain PIEH')

        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()

        flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

        return flow.astype(np.float32)

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png',
                    imgpath + '/guide_mask_3.png', imgpath + '/guide_mask_1.png', imgpath + '/guide_mask_2.png',
                    imgpath + '/var.npy',]

        # Load images
        img0 = cv2.imread(imgpaths[0])
        gt = cv2.imread(imgpaths[1])
        img1 = cv2.imread(imgpaths[2])
        space_mask0 = cv2.imread(imgpaths[3], flags=0)[..., np.newaxis]
        space_mask1 = cv2.imread(imgpaths[4], flags=0)[..., np.newaxis]
        space_mask2 = cv2.imread(imgpaths[5], flags=0)[..., np.newaxis]
        space_mask = np.concatenate((space_mask0, space_mask1, space_mask2), 2).astype(np.float64)

        timestep = 0.5

        var = np.load(imgpaths[6]).astype(np.float64)

        return img0, gt, img1, timestep, space_mask, var


    def __getitem__(self, index):
        img0, gt, img1, timestep, space_mask, var = self.getimg(index)
        # print('img0_0', img0.shape)
        # print('space_0', space_mask.shape)
        # print('var_0', var.shape)
        if self.dataset_name == 'train':
            img0, gt, img1,  space_mask, var = self.crop(img0, gt, img1, space_mask, var, 224, 224)
            # print('img0_1', img0.shape)
            # print('space_1', space_mask.shape)
            # print('var_1', var.shape)
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
                var = var[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                var = var[::-1]
                space_mask = space_mask[::-1]
            if random.uniform(0, 1) < 0.5:
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                var = var[:, ::-1]
                space_mask = space_mask[:, ::-1]
            if random.uniform(0, 1) < 0.5:
                tmp = img1
                img1 = img0
                img0 = tmp
                timestep = 1 - timestep
            # random rotation
            p = random.uniform(0, 1)
            space_mask_l = []
            var_l = []
            if random.uniform(0, 1) < 0.05:
                img0 = img0.transpose((1, 0, 2))
                gt = gt.transpose((1, 0, 2))
                img1 = img1.transpose((1, 0, 2))
                var = var.transpose((1, 0, 2))
                space_mask = space_mask.transpose((1, 0, 2))

            # if p < 0.25:
            #     img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
            #     gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
            #     img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            # elif p < 0.5:
            #     img0 = cv2.rotate(img0, cv2.ROTATE_180)
            #     gt = cv2.rotate(gt, cv2.ROTATE_180)
            #     img1 = cv2.rotate(img1, cv2.ROTATE_180)
            # elif p < 0.75:
            #     img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #     gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #     img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)

        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        timestep = torch.tensor(timestep).reshape(1, 1, 1)
        space_mask = torch.from_numpy(space_mask.transpose((2, 0, 1)).astype(np.float32))
        var = torch.from_numpy(var.transpose((2, 0, 1)).astype(np.float32))
        # space_mask = torch.unsqueeze(torch.from_numpy(space_mask.copy()), dim=1)
        # var = torch.from_numpy(var.copy()).permute(0, 3, 1, 2)
        # print('img0', img0.shape)
        # print('space', space_mask.shape)
        # print('var', var.shape)

        return torch.cat((img0, img1, gt), 0), timestep, space_mask, var

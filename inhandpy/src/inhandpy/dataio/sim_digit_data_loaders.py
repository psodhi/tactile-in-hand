# Copyright (c) Facebook, Inc. and its affiliates.

import os
import glob
import pandas as pd

import numpy as np
from PIL import Image
from functools import lru_cache

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from inhandpy.utils import utils

class DigitTactoImageTfSeqDataset(Dataset):

    def __init__(self, dir_dataset, dataset_name='', seq_step=1, transform=None, downsample_imgs=False, img_types=['color'], digit_id='bot', poses_flag=True):

        self.dir_dataset = dir_dataset
        self.dataset_name = dataset_name

        self.transform = transform
        self.downsample_imgs = downsample_imgs

        self.img_types = img_types
        self.poses_flag = poses_flag

        self.min_seq_data = 10
        self.seq_step = seq_step

        self.digit_id = digit_id

        self.seq_data_list = []
        seq_dirs = sorted(glob.glob(f"{dir_dataset}/*"))
        for seq_dir in seq_dirs:
    
            seq_data = self.load_seq_from_file(f"{seq_dir}/poses_imgs.csv")

            if seq_data is None:
                continue

            self.seq_data_list.append(seq_data)
    
    def get_dataset_name(self):
        return self.dataset_name

    def get_uids(self):
        seq_idxs = list( range(0, len(self.seq_data_list)) )
        uids = [f"{self.dataset_name}/episode_{seq_idx:04d}" for seq_idx in seq_idxs]
        return uids

    def load_seq_from_file(self, csvfile):

        seq_data = pd.read_csv(csvfile) if os.path.exists(csvfile) else None

        if seq_data is not None:
            seq_data = None if (len(seq_data) < self.min_seq_data) else seq_data

        return seq_data
    
    def get_images_seq(self, seq_idx, img_type):

        seq_data = self.seq_data_list[seq_idx]
        img_locs = seq_data[f"img_{self.digit_id}_{img_type}_loc"]
        imgs = []
        for idx in range(0, len(seq_data), self.seq_step):
            imgs.append(Image.open(f"{self.dir_dataset}/{img_locs[idx]}"))

        return imgs

    def get_poses_seq(self, seq_idx):

        seq_data = self.seq_data_list[seq_idx]

        obj_pos = torch.FloatTensor(utils.pandas_col_to_numpy(seq_data["obj_pos"])[0::self.seq_step])
        obj_ori = torch.FloatTensor(utils.pandas_col_to_numpy(seq_data["obj_ori"])[0::self.seq_step])
        obj_ori = obj_ori.reshape(obj_ori.shape[0], 3, 3)
        obj_poses = (obj_ori, obj_pos)

        digit_top_pos = torch.FloatTensor(utils.pandas_col_to_numpy(seq_data["digit_top_pos"])[0::self.seq_step])
        digit_top_ori = torch.FloatTensor(utils.pandas_col_to_numpy(seq_data["digit_top_ori"])[0::self.seq_step])
        digit_top_ori = digit_top_ori.reshape(digit_top_ori.shape[0], 3, 3)
        digit_top_poses = (digit_top_ori, digit_top_pos)

        digit_bot_pos = torch.FloatTensor(utils.pandas_col_to_numpy(seq_data["digit_bot_pos"])[0::self.seq_step])
        digit_bot_ori = torch.FloatTensor(utils.pandas_col_to_numpy(seq_data["digit_bot_ori"])[0::self.seq_step])
        digit_bot_ori = digit_bot_ori.reshape(digit_bot_ori.shape[0], 3, 3)
        digit_bot_poses = (digit_bot_ori, digit_bot_pos)

        poses = [obj_poses, digit_top_poses, digit_bot_poses]

        return poses

    def __getitem__(self, seq_idx):
        """
        :return: imgs nseq x C x H x W
        :return: poses nseq x 6
        """
        
        # load image data
        imgs_seq_list = []
        for img_type in self.img_types:

            imgs_seq = self.get_images_seq(seq_idx, img_type)
            if self.transform is not None:
                for idx, img in enumerate(imgs_seq):
                    imgs_seq[idx] = self.transform(img)

            if self.downsample_imgs:
                for idx, img in enumerate(imgs_seq):
                    img = torch.nn.functional.interpolate(img, size=64)
                    img = torch.nn.functional.interpolate(img.permute(0, 2, 1), size=64)
                    img = img.permute(0, 2, 1)

                    imgs_seq[idx] = img
            
            imgs_seq_list.append(imgs_seq)

        # imgs list of lists -> list of nd tensors S x C x H x W
        imgs_seq_list = [torch.stack(imgs_seq) for imgs_seq in imgs_seq_list]

        if self.poses_flag is False:
            data = imgs_seq_list
            return data

        # load pose data
        poses = self.get_poses_seq(seq_idx)

        imgs_seq_pose_list = imgs_seq_list + poses

        # assertion checks
        assert(imgs_seq_list[0].shape[0] == poses[0][0].shape[0])        
        if any(elem is None for elem in imgs_seq_pose_list):
            print(
                "[DigitTactoImageTfSeqDataset] Unable to read img, pose data at seq_idx {0}".format(seq_idx))
            return

        # return data
        data = tuple(imgs_seq_pose_list)

        return data

    def __len__(self):
        return len(self.seq_data_list)
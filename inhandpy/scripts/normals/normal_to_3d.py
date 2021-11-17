# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import math

import hydra
import logging
import glob
import json

import pandas as pd
from PIL import Image
import cv2
import imageio

import copy
from attrdict import AttrDict
import open3d as o3d

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

from inhandpy.dataio.real_digit_data_loaders import DigitRealImageAnnotDataset
from inhandpy.utils import geom_utils, vis_utils

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
plt.ion()

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "config/rgb_to_normal.yaml")        

# visualizer
view_params = AttrDict({'fov': 60, 'front': [-0.56, -0.5, 0.4], 'lookat': [
                        -0.006, -0.0117, -0.02], 'up': [0.0816, -0.112, 0.990], 'zoom': 0.5})
# view_params = AttrDict({'fov': 60, 'front': [0.2, 0.95, 0.14], 'lookat': [
#                         -0.00, -0.0, 0.025], 'up': [0.0, 0.0, -0.4], 'zoom': 2.0})
vis3d = vis_utils.Visualizer3d(base_path=BASE_PATH, view_params=view_params)

def data_loader(dataset_names, datatype, params):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_list = []
    for (ds_idx, dataset_name) in enumerate(dataset_names):
        srcdir_dataset = f"{BASE_PATH}/local/datasets/{dataset_name}"
        dataset_list.append(DigitRealImageAnnotDataset(
            dir_dataset=f"{srcdir_dataset}/{datatype}", transform=transform, annot_flag=False))

    dataset = ConcatDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=params.batch_size,
                            shuffle=params.shuffle, num_workers=params.num_workers)

    return dataloader, dataset

@hydra.main(config_path=CONFIG_PATH)
def main(cfg):

    train_dataloader, train_dataset = data_loader(
        dataset_names=cfg.dataset_names, datatype=cfg.dataset_type, params=cfg.dataloader)
    
    dataset_name = cfg.dataset_names[0]
    dataset_type = cfg.dataset_type
    
    # projection params
    T_cam_offset = torch.tensor(cfg.sensor.T_cam_offset)
    proj_mat = torch.tensor(cfg.sensor.P)

    meshes = vis3d.init_geometry(geom_type="mesh", num_items=1, file_names=[cfg.sensor.mesh_name], wireframes=[True])

    # iterate over dataset
    ds_idx = 0

    pix2pix_results_dir = f"{BASE_PATH}/local/datasets/pix2pix/results/{dataset_name}"

    color_files = sorted(glob.glob(f"{pix2pix_results_dir}/test/color/*.png"))
    normal_files = sorted(glob.glob(f"{pix2pix_results_dir}/test/normal/*.png"))

    for color_file, normal_file in zip(color_files, normal_files):

        print(f"Dataset idx: {ds_idx:04d}")

        fig1, axs1 = plt.subplots(nrows=3, ncols=3, num=1, clear=True, figsize=(24, 12))

        img_color_np = imageio.imread(color_file)
        img_normal_np = imageio.imread(normal_file)
        img_normal_np = img_normal_np / 255.

        # normal integration params
        img_normal = torch.FloatTensor(img_normal_np)
        img_normal = img_normal.permute(2,0,1) # (320,240,3) -> (3,320,240)
        boundary = torch.zeros((img_normal.shape[-2], img_normal.shape[-1]))
        bg_mask = None

        # 2. normal -> grad depth
        img_normal = geom_utils._preproc_normal(img_normal=img_normal, bg_mask=bg_mask)
        gradx, grady = geom_utils._normal_to_grad_depth(img_normal=img_normal, gel_width=cfg.sensor.gel_width,
                                                        gel_height=cfg.sensor.gel_height, bg_mask=bg_mask, real=True)

        # 3. grad depth -> depth
        img_depth = geom_utils._integrate_grad_depth(gradx, grady, boundary=boundary, bg_mask=bg_mask, max_depth=0.02)

        # 4. depth -> points3d
        view_mat = torch.eye(4) # torch.inverse(T_cam_offset)
        points3d = geom_utils.depth_to_pts3d(depth=img_depth, P=proj_mat, V=view_mat, params=cfg.sensor)
        points3d = geom_utils.remove_background_pts(points3d, bg_mask=None)

        # visualize normals/depth
        img_depth_np = img_depth.cpu().detach().numpy()
        if cfg.visualize.normals:

            vis_utils.visualize_imgs(fig=fig1, axs=[axs1[0,0], axs1[0,1], axs1[0,2]], 
                                     img_list=[img_normal_np[:, :, 0], img_normal_np[:, :, 1], img_normal_np[:, :, 2]],
                                     titles=['nx', 'ny', 'nz'], cmap='coolwarm')
            vis_utils.visualize_imgs(fig=fig1, axs=[axs1[1, 0], axs1[1, 1]],
                                     img_list=[gradx.cpu().detach().numpy(), grady.cpu().detach().numpy()],
                                     titles=['gradx', 'grady'], cmap='coolwarm')
            vis_utils.visualize_imgs(fig=fig1, axs=[axs1[2, 0], axs1[2, 1], axs1[2, 2]],
                                     img_list=[img_color_np, img_normal_np, img_depth_np], 
                                     titles=['img_color', 'img_normal', 'img_depth_recon'], cmap='coolwarm')
            plt.pause(1e-3)

        # visualize cloud
        if cfg.visualize.points3d:
            cloud = o3d.geometry.PointCloud()
            clouds = geom_utils.init_points_to_clouds(clouds=[copy.deepcopy(cloud)], points3d=[points3d])
            vis_utils.visualize_geometries_o3d(vis3d=vis3d, clouds=clouds, meshes=meshes)
        
        ds_idx = ds_idx + 1

if __name__ == '__main__':
    main()

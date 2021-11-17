# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os
import math

import hydra
import logging
import glob
import json
from attrdict import AttrDict

import pandas as pd
from PIL import Image
import cv2
import imageio
import copy
import open3d as o3d

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

from inhandpy.dataio.real_digit_data_loaders import DigitRealImageAnnotDataset
from inhandpy.utils import geom_utils, vis_utils, data_utils

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
plt.ion()

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "config/rgb_to_normal.yaml")        

# visualizer
view_params = AttrDict({'fov': 60, 'front': [-0.56, 0.81, 0.14], 'lookat': [
                        -0.006, -0.0117, 0.033], 'up': [0.0816, -0.112, 0.990], 'zoom': 5.0})
# view_params = AttrDict({'fov': 60, 'front': [0.2, 0.95, 0.14], 'lookat': [
#                         -0.00, -0.0, 0.025], 'up': [0.0, 0.0, -0.4], 'zoom': 2.0})
vis3d = vis_utils.Visualizer3d(base_path=BASE_PATH, view_params=view_params)

def data_loader(dataset_names, datatype, params):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_list = []
    for (ds_idx, dataset_name) in enumerate(dataset_names):
        srcdir_dataset = f"{BASE_PATH}/local/datasets/{dataset_name}"
        dataset_list.append(DigitRealImageAnnotDataset(
            dir_dataset=f"{srcdir_dataset}/{datatype}", transform=transform, annot_flag=params.annot_flag))

    dataset = ConcatDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=params.batch_size,
                            shuffle=params.shuffle, num_workers=params.num_workers)

    return dataloader, dataset

def generate_sphere_gt_normals(img_mask, center_x, center_y, radius):

    img_normal = np.zeros(img_mask.shape, dtype="float64")

    for y in range(img_mask.shape[0]):
        for x in range(img_mask.shape[1]):

            img_normal[y, x, 0] = 0.
            img_normal[y, x, 1] = 0.
            img_normal[y, x, 2] = 1.

            if (np.sum(img_mask[y, x, :]) > 0):

                dist = np.sqrt((x-center_x)**2 + (y-center_y)**2)
                ang_xz = math.acos(dist / radius)
                ang_xy = math.atan2(y-center_y, x-center_x)

                nx = math.cos(ang_xz) * math.cos(ang_xy)
                ny = math.cos(ang_xz) * math.sin(ang_xy)
                nz = math.sin(ang_xz)

                img_normal[y, x, 0] = nx
                img_normal[y, x, 1] = -ny
                img_normal[y, x, 2] = nz

            norm_val = np.linalg.norm(img_normal[y, x, :])
            img_normal[y, x, :] = img_normal[y, x, :] / norm_val

    # img_normal between [-1., 1.], converting to [0., 1.]
    img_normal = (img_normal + 1.) * 0.5

    return img_normal

@hydra.main(config_path=CONFIG_PATH)
def main(cfg):

    train_dataloader, train_dataset = data_loader(
        dataset_names=cfg.dataset_names, datatype=cfg.dataset_type, params=cfg.dataloader)
    
    dataset_name = cfg.dataset_names[0]
    dataset_type = cfg.dataset_type

    dirs = []
    dirs.append(f"{BASE_PATH}/local/datasets/pix2pix/{dataset_name}/A")
    dirs.append(f"{BASE_PATH}/local/datasets/pix2pix/{dataset_name}/B")
    dirs.append(f"{BASE_PATH}/local/datasets/pix2pix/{dataset_name}/AB")
    
    for dir in dirs: os.makedirs(f"{dir}/{dataset_type}", exist_ok=True)

    # projection params
    T_cam_offset = torch.tensor(cfg.sensor.T_cam_offset)
    proj_mat = torch.tensor(cfg.sensor.P)

    # iterate over dataset
    ds_idx = 0

    mm_to_pixel = 181 / 9.525
    # radius_bearing = np.int(0.5 * 6.35 * mm_to_pixel) # 1/4" dia
    radius_bearing = np.int(0.5 * 12.7 * mm_to_pixel) # 1/2" dia

    while ds_idx < len(train_dataset):
        print(f"Dataset idx: {ds_idx:04d}")

        fig1, axs1 = plt.subplots(nrows=3, ncols=3, num=1, clear=True, figsize=(24, 12))
    
        # read img + annotations
        data = train_dataset[ds_idx]

        if cfg.dataloader.annot_flag: 
            img, annot = data
            if (annot.shape[0] == 0):
                ds_idx = ds_idx + 1
                continue
        else:
            img = data
        
        # get annotation circle params
        if cfg.dataloader.annot_flag:
            annot_np = annot.cpu().detach().numpy()        
            center_y, center_x, radius_annot = annot_np[0], annot_np[1], annot_np[2]
        else:
            center_y, center_x, radius_annot = 0, 0, 0

        img_color_np = img.permute(1,2,0).cpu().detach().numpy() # (3,320,240) -> (320,240,3)

        # apply foreground mask
        fg_mask = np.zeros(img_color_np.shape[:2], dtype='uint8')
        fg_mask = cv2.circle(fg_mask, (center_x, center_y), radius_annot, 255, -1)

        # 1. rgb -> normal (generate gt surface normals)
        img_mask = cv2.bitwise_and(img_color_np, img_color_np, mask=fg_mask)
        img_normal_np = generate_sphere_gt_normals(img_mask, center_x, center_y, radius=radius_bearing)

        # normal integration params
        img_normal = torch.FloatTensor(img_normal_np)
        img_normal = img_normal.permute(2,0,1) # (320,240,3) -> (3,320,240)
        boundary = torch.zeros((img_normal.shape[-2], img_normal.shape[-1]))
        bg_mask = torch.tensor(1-fg_mask/255., dtype=torch.bool)

        # 2. normal -> grad depth
        img_normal = geom_utils._preproc_normal(img_normal=img_normal, bg_mask=bg_mask)
        gradx, grady = geom_utils._normal_to_grad_depth(img_normal=img_normal, gel_width=cfg.sensor.gel_width,
                                                        gel_height=cfg.sensor.gel_height, bg_mask=bg_mask, real=False)

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
            vis_utils.visualize_geometries_o3d(vis3d=vis3d, clouds=clouds)

        # save dataset for learning pix2pix model
        if cfg.pix2pix.save_dataset:

            # downsample: (320,240,3) -> (160,120,3)
            img_color_ds = data_utils.interpolate_img(img=torch.tensor(img_color_np).permute(2,0,1), rows=160, cols=120)
            img_normal_ds = data_utils.interpolate_img(img=torch.tensor(img_normal_np).permute(2,0,1), rows=160, cols=120)
            img_color_np = img_color_ds.permute(1,2,0).cpu().detach().numpy()
            img_normal_np = img_normal_ds.permute(1,2,0).cpu().detach().numpy()

            imageio.imwrite(f"{dirs[0]}/{dataset_type}/{ds_idx:04d}.png", img_color_np)
            imageio.imwrite(f"{dirs[1]}/{dataset_type}/{ds_idx:04d}.png", img_normal_np)
        
        ds_idx = ds_idx + 1

    if cfg.pix2pix.save_dataset:
        os.system(f"python {cfg.pix2pix.scripts_dir}/datasets/combine_A_and_B.py --fold_A {dirs[0]} --fold_B {dirs[1]} --fold_AB {dirs[2]}")
        log.info(f"Created color+normal dataset of {ds_idx} images at {dirs[2]}/{dataset_type}.")

if __name__ == '__main__':
    main()

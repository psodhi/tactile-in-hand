# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os

import hydra
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from inhandpy.dataio.sim_digit_data_loaders import DigitTactoImageTfSeqDataset
from inhandpy.utils import geom_utils, vis_utils

log = logging.getLogger(__name__)
plt.ion()

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "config/depth_normals_cloud.yaml")        

def data_loader(dataset_names, datatype, params):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_list = []
    for (ds_idx, dataset_name) in enumerate(dataset_names):
        srcdir_dataset = f"{BASE_PATH}/local/datasets/{dataset_name}"
        dataset_list.append(DigitTactoImageTfSeqDataset(
            dir_dataset=f"{srcdir_dataset}/{datatype}", base_path=BASE_PATH, transform=transform,
            downsample_imgs=params.downsample_imgs, img_types=params.img_types, digit_id=params.digit_id))

    dataset = ConcatDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=params.batch_size,
                            shuffle=params.shuffle, num_workers=params.num_workers)

    return dataloader, dataset

@hydra.main(config_path=CONFIG_PATH)
def main(cfg):

    # data loader
    train_dataloader, train_dataset = data_loader(
        dataset_names=cfg.dataset_names, datatype="train", params=cfg.dataloader)

    gel_depth = torch.tensor(np.loadtxt(f"{BASE_PATH}/{cfg.sensor.gel_depth_map_loc}", delimiter=',') - cfg.sensor.gel_depth_offset)

    # iterate over dataset
    for ds_idx, dataset in enumerate(train_dataset):

        print(f"Dataset idx: {ds_idx:03d}")

        # load imgs: S x C x H x W , poses: (S x 3 x 3, S x 3) <-> (R, t)
        img_seq_color, img_seq_depth, img_seq_normal, pose_seq_obj, pose_seq_digit_top, pose_seq_digit_bot = dataset
        pose_seq_sensor = pose_seq_digit_bot if (cfg.dataloader.digit_id == "bot") else pose_seq_digit_top

        S, C, H, W = img_seq_color.shape

        skip = 2
        seq_idxs_1 = torch.arange(0, S-skip, skip)
        seq_idxs_2 = torch.arange(skip, S, skip)
        idx = 0
        while (idx < len(seq_idxs_1)):

            fig1, axs1 = plt.subplots(nrows=2, ncols=2, num=1, clear=True, figsize=(12, 8))
            fig2, axs2 = plt.subplots(nrows=1, ncols=3, num=2, clear=True, figsize=(12, 8))
            fig3, axs3 = plt.subplots(nrows=1, ncols=3, num=3, clear=True, figsize=(12, 8))

            print(f"Sequence idx pair: ({seq_idxs_1[idx]:03d}, {seq_idxs_2[idx]:03d})")

            # get current img pairs
            img_normal_1 = img_seq_normal[seq_idxs_1[idx], :, :, :]
            img_normal_2 = img_seq_normal[seq_idxs_2[idx], :, :, :]

            img_depth_1 = (img_seq_depth[seq_idxs_1[idx], :, :, :])
            img_depth_2 = (img_seq_depth[seq_idxs_2[idx], :, :, :])

            # background mask
            bg_mask_1 = (img_depth_1 > gel_depth).squeeze()
            bg_mask_2 = (img_depth_2 > gel_depth).squeeze()

            # preproc imgs
            img_normal_1 = geom_utils._preproc_normal(img_normal=img_normal_1, bg_mask=bg_mask_1)
            img_depth_1 = geom_utils._preproc_depth(img_depth=img_depth_1.squeeze(), bg_mask=bg_mask_1)
            
            # get grad imgs
            gradx_depth, grady_depth = geom_utils._depth_to_grad_depth(img_depth=img_depth_1.squeeze(), bg_mask=bg_mask_1) # grad from depth imgs
            gradx_normal, grady_normal = geom_utils._normal_to_grad_depth(img_normal=img_normal_1, gel_width=cfg.sensor.gel_width,
                                                              gel_height=cfg.sensor.gel_height, bg_mask=bg_mask_1)  # grad from surface normal imgs

            # integrate grad depth
            boundary = torch.zeros_like(gel_depth) # bdry for gxx, gyy
            img_depth_recon_depth = geom_utils._integrate_grad_depth(gradx_depth, grady_depth, boundary=boundary, bg_mask=bg_mask_1, max_depth=0.02)
            img_depth_recon_normal = geom_utils._integrate_grad_depth(gradx_normal, grady_normal, boundary=boundary, bg_mask=bg_mask_1, max_depth=0.02)

            # mask out background on final recon img
            img_depth_recon_depth = geom_utils.mask_background(img_depth_recon_depth, bg_mask=bg_mask_1)
            img_depth_recon_normal = geom_utils.mask_background(img_depth_recon_normal, bg_mask=bg_mask_1)

            ## plotting
            gxd = axs1[0, 0].imshow(gradx_depth, cmap='coolwarm')
            gyd = axs1[0, 1].imshow(grady_depth, cmap='coolwarm')
            gxn = axs1[1, 0].imshow(gradx_normal, cmap='coolwarm')
            gyn = axs1[1, 1].imshow(grady_normal, cmap='coolwarm')

            axs1[0, 0].set_title("grad x (from depth)")
            axs1[0, 1].set_title("grad y (from depth)")
            axs1[1, 0].set_title("grad x (from normals)")
            axs1[1, 1].set_title("grad y (from normals)")

            fig1.colorbar(gxd, ax=axs1[0, 0])
            fig1.colorbar(gyd, ax=axs1[0, 1])
            fig1.colorbar(gxn, ax=axs1[1, 0])
            fig1.colorbar(gyn, ax=axs1[1, 1])

            nx = axs2[0].imshow(img_normal_1[0, :], cmap='coolwarm', vmin=-1., vmax=1.)
            ny = axs2[1].imshow(img_normal_1[1, :], cmap='coolwarm', vmin=-1., vmax=1.)
            nz = axs2[2].imshow(img_normal_1[2, :], cmap='coolwarm', vmin=-1., vmax=1.)

            axs2[0].set_title("normals x")
            axs2[1].set_title("normals y")
            axs2[2].set_title("normals z")

            fig2.colorbar(nx, ax=axs2[0])
            fig2.colorbar(ny, ax=axs2[1])
            fig2.colorbar(nz, ax=axs2[2])

            min_depth_gt, max_depth_gt = torch.min(img_depth_1), torch.max(img_depth_1)

            d1 = axs3[0].imshow(geom_utils.flip(img_depth_1.squeeze()), cmap='coolwarm', vmin=min_depth_gt, vmax=max_depth_gt)
            d2 = axs3[1].imshow(geom_utils.flip(img_depth_recon_depth), cmap='coolwarm', vmin=min_depth_gt, vmax=max_depth_gt)
            d3 = axs3[2].imshow(geom_utils.flip(img_depth_recon_normal), cmap='coolwarm', vmin=min_depth_gt, vmax=max_depth_gt)

            axs3[0].set_title("depth groundtruth")
            axs3[1].set_title("depth from grad depth")
            axs3[2].set_title("depth from normals")

            fig3.colorbar(d1, ax=axs3[0])
            fig3.colorbar(d2, ax=axs3[1])
            fig3.colorbar(d3, ax=axs3[2])

            plt.show()
            plt.pause(1e-1)

            idx = (idx + 1)

if __name__ == '__main__':
    main()

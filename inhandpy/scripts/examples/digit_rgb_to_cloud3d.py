# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os

import hydra
import logging

from PIL import Image

from tqdm import tqdm

import copy
from attrdict import AttrDict
import open3d as o3d

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

from inhandpy.dataio.real_digit_data_loaders import DigitRealImageTfSeqDataset
from inhandpy.dataio.sim_digit_data_loaders import DigitTactoImageTfSeqDataset
from inhandpy.utils import geom_utils, vis_utils, data_utils

from inhandpy.thirdparty.pix2pix.options.test_options import TestOptions
from inhandpy.thirdparty.pix2pix.models import create_model
from inhandpy.thirdparty.pix2pix.data.base_dataset import get_params, get_transform

import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
plt.ion()

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "config/digit_rgb_to_cloud.yaml")        

# visualizer
view_params = AttrDict({'fov': 60, 'front': [0.4257, -0.2125, -0.8795], 'lookat': [
                        0.02,0.0,0.0], 'up': [0.9768, -0.0694, 0.2024], 'zoom': 0.25})

vis3d = vis_utils.Visualizer3d(base_path=BASE_PATH, view_params=view_params)

def data_loader(dataset_names, params, datatype="test", setup="real"):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_list = []
    for (ds_idx, dataset_name) in enumerate(dataset_names):

        srcdir_dataset = f"{BASE_PATH}/local/datasets/{dataset_name}"

        if (setup == "real"):
            img_types = ['color']
            dataset = DigitRealImageTfSeqDataset(dir_dataset=f"{srcdir_dataset}/{datatype}", dataset_name=dataset_name, transform=transform, img_types=img_types, poses_flag=False)
        elif (setup == "sim"):
            img_types = ['color', 'normal', 'depth']
            dataset = DigitTactoImageTfSeqDataset(dir_dataset=f"{srcdir_dataset}/{datatype}", dataset_name=dataset_name, transform=transform,img_types=img_types, poses_flag=False)
        else:
            logging.error(f"[digit_rgb_to_cloud::data_loader] data loader for setup {setup} not found")
        
        dataset_list.append(dataset)

    dataset = ConcatDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=params.batch_size,
                            shuffle=params.shuffle, num_workers=params.num_workers)

    return dataloader, dataset

def init_image_translation_model(params):

    # set model options
    opt = TestOptions().parse()
    opt.name = params.model_name
    opt.model = params.model_type
    opt.dataset_mode = params.dataset_mode
    opt.direction = params.direction
    opt.netG = opt.netG

    opt.num_threads = 0
    opt.batch_size = 1 
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.checkpoints_dir = f'{BASE_PATH}/local/pix2pix/checkpoints'

    # create model
    model = create_model(opt)
    model.setup(opt)

    if opt.eval: model.eval()

    return model, opt

def color_to_normal_pix2pix(img_seq_color, params_pix2pix):

    model, opt = init_image_translation_model(params=params_pix2pix)
    
    img_seq_normal = torch.zeros_like(img_seq_color) # S x C x H x W    

    for seq_idx in range(0, img_seq_color.shape[0]):

        img_color = img_seq_color[seq_idx, :]

        # preprocess / transform input image 
        img_color = (img_color * 255.).permute(1,2,0).cpu().float().numpy().astype(np.uint8)
        img_color = Image.fromarray(img_color)
        transform_params = get_params(opt, img_color.size)
        A_transform = get_transform(opt, transform_params, grayscale=False)
        img_color = A_transform(img_color)

        # create data input to model
        img_color = img_color.unsqueeze(0) # B x C x H x W        
        data = {}
        data['A'], data['B'] = img_color, img_color
        data['A_paths'], data['B_paths'] = '', ''

        # call model    
        model.set_input(data)
        model.test()
        output = model.get_current_visuals()

        # post-process model output
        img_normal = ((output['fake_B']).squeeze(0) + 1) / 2.0
        img_normal = data_utils.interpolate_img(img=img_normal, rows=160, cols=120)

        img_seq_normal[seq_idx, :] = img_normal
    
    return img_seq_normal

@hydra.main(config_path=CONFIG_PATH)
def main(cfg):
    
    # data loader
    dataloader, dataset = data_loader(
        dataset_names=cfg.dataset_names, datatype=cfg.datatype, setup=cfg.setup, params=cfg.dataloader)
    
    dataset_uids = [item.get_uids() for item in dataset.datasets]
    dataset_uids = [item for sublist in dataset_uids for item in sublist]

    # projection params
    T_cam_offset = torch.tensor(cfg.sensor.T_cam_offset_real) if (cfg.setup is "real") else torch.tensor(cfg.sensor.T_cam_offset_sim)
    proj_mat = torch.tensor(cfg.sensor.P)
    gel_depth = torch.tensor((np.loadtxt(
        f"{BASE_PATH}/{cfg.sensor.gel_depth_map_loc}", delimiter=',') - cfg.sensor.gel_depth_offset).astype(np.float32))

    meshes = vis3d.init_geometry(geom_type="mesh", num_items=1, colors=[[0.55, 0.55, 0.55]], file_names=[cfg.sensor.mesh_name], wireframes=[True])

    # iterate over dataset
    for ds_idx in tqdm(range(0, len(dataset))):

        log.info(f"[run_tracker] Running dataset {dataset_uids[ds_idx]}")
        
        # load imgs: S x C x H x W
        data = dataset[ds_idx]
                
        img_seq_color, img_seq_normal_gt, img_seq_depth_gt = None, None, None
        if (cfg.setup == 'real'):
            img_seq_color = data
        elif (cfg.setup == 'sim'):
            img_seq_color, img_seq_normal_gt, img_seq_depth_gt = data

        S, C, H, W = img_seq_color.shape       

        # run pixpix image translation network: img_seq_color -> img_seq_normal
        img_seq_normal_pred = color_to_normal_pix2pix(img_seq_color=img_seq_color, params_pix2pix=cfg.pix2pix)

        if (img_seq_normal_gt is not None) & (cfg.use_gt_normals):
            img_seq_normal = img_seq_normal_gt
        else:
            img_seq_normal = img_seq_normal_pred

        for seq_idx in tqdm(range(0, S)):

            fig1, axs1 = plt.subplots(nrows=3, ncols=3, num=1, clear=True, figsize=(20, 12))

            img_color = copy.deepcopy(img_seq_color[seq_idx, :])
            img_normal = copy.deepcopy(img_seq_normal[seq_idx, :])

            # TODO (psodhi): Background gt normals nx, ny are non-zero for sim but correctly zero for real.
            # Hence, we mask out background in sim relying on gt depth. Once background is fixed, we can remove the code snippet below.
            bg_mask = None
            if cfg.setup == 'sim':
                img_depth_gt = copy.deepcopy(img_seq_depth_gt[seq_idx, :])
                bg_mask = (img_depth_gt > gel_depth).squeeze()

            # Step 1. normal -> grad depth
            img_normal = geom_utils._preproc_normal(img_normal=img_normal, bg_mask=bg_mask)
            gradx, grady = geom_utils._normal_to_grad_depth(img_normal=img_normal, gel_width=cfg.sensor.gel_width,
                                                            gel_height=cfg.sensor.gel_height, bg_mask=bg_mask)
            # Step 2. grad depth -> depth
            boundary = torch.zeros((img_normal.shape[-2], img_normal.shape[-1]))
            img_depth = geom_utils._integrate_grad_depth(gradx, grady, boundary=boundary, bg_mask=bg_mask, max_depth=0.02)

            if cfg.remove_background_depth is True:
                img_depth = geom_utils.mask_background(img_depth, bg_mask=(img_depth >= cfg.max_depth), bg_val=0.)

            # Step 3. depth -> points3d
            view_mat = torch.inverse(T_cam_offset)
            points3d = geom_utils.depth_to_pts3d(depth=img_depth, P=proj_mat, V=view_mat, params=cfg.sensor)
            points3d = geom_utils.remove_outlier_pts(points3d, nb_neighbors=20, std_ratio=10.)

            # Visualize normals/depth
            img_color_np = (img_color.permute(1,2,0)).cpu().detach().numpy()
            img_normal_np = (img_normal.permute(1,2,0)).cpu().detach().numpy()
            img_depth_np = img_depth.cpu().detach().numpy()

            if cfg.visualize.imgs:

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
            if cfg.visualize.cloud:
                cloud = o3d.geometry.PointCloud()
                clouds = geom_utils.init_points_to_clouds(clouds=[copy.deepcopy(cloud)], points3d=[points3d])
                vis_utils.visualize_geometries_o3d(vis3d=vis3d, clouds=clouds, meshes=meshes)

if __name__ == '__main__':
    main()

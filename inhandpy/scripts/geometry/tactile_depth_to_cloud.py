# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os

import hydra
import logging
from attrdict import AttrDict
import copy

import cv2
import open3d as o3d

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import types

from inhandpy.dataio.sim_digit_data_loaders import DigitTactoImageTfSeqDataset
from inhandpy.utils import geom_utils, vis_utils

log = logging.getLogger(__name__)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "config/depth_normals_cloud.yaml")        

def visualize_geometries(vis3d, seq_idx, pose_seq_obj, pose_seq_digit_top, pose_seq_digit_bot, 
                    T_cam_offset, points3d, clouds, frames, meshes, params=None):
        
        # transforms
        T_obj_curr = geom_utils.Rt_to_T(pose_seq_obj[0][seq_idx, :], pose_seq_obj[1][seq_idx, :])
        T_digit_top_curr = geom_utils.Rt_to_T(pose_seq_digit_top[0][seq_idx, :], pose_seq_digit_top[1][seq_idx, :])
        T_digit_bot_curr = geom_utils.Rt_to_T(pose_seq_digit_bot[0][seq_idx, :], pose_seq_digit_bot[1][seq_idx, :])

        T_cam_top_curr = torch.matmul(T_digit_top_curr, T_cam_offset)
        T_cam_bot_curr = torch.matmul(T_digit_bot_curr, T_cam_offset)

        # clouds
        points3d_vis = points3d.cpu().detach().numpy()
        clouds[0].points = o3d.utility.Vector3dVector(points3d_vis.transpose())

        # meshes, frames
        meshes = [copy.deepcopy(mesh) for mesh in meshes]
        frames = [copy.deepcopy(frame) for frame in frames]

        vis3d.transform_geometry_absolute([T_obj_curr, T_digit_top_curr, T_digit_bot_curr], meshes)
        vis3d.transform_geometry_absolute([T_obj_curr, T_cam_top_curr, T_cam_bot_curr], frames)

        vis3d.add_geometry(clouds)
        vis3d.add_geometry(meshes)
        vis3d.add_geometry(frames)

        vis3d.render()
        # vis3d.pan_scene()        

        vis3d.remove_geometry(clouds)
        vis3d.remove_geometry(meshes)
        vis3d.remove_geometry(frames)


def visualize_imgs(seq_idx, img_seq_depth, img_seq_normal, params=None, tpause=100):

    img_depth = ((img_seq_depth[seq_idx, :, :, :]).permute(1, 2, 0)).cpu().detach().numpy()
    img_normal = ((img_seq_normal[seq_idx, :, :, :]).permute(1, 2, 0)).cpu().detach().numpy()
    img_depth = vis_utils.depth_to_color(img_depth)

    cv2.imshow("img_depth", img_depth)
    cv2.imshow("img_normal", img_normal)

    cv2.waitKey(tpause)

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

    # init values
    view_params = AttrDict({'fov': 60, 'front': [-0.56, 0.81, 0.14], 'lookat': [
                           -0.006, -0.0117, 0.043], 'up': [0.0816, -0.112, 0.990], 'zoom': 0.5})
    vis3d = vis_utils.Visualizer3d(base_path=BASE_PATH, view_params=view_params)

    T_cam_offset = torch.tensor(cfg.sensor.T_cam_offset)
    proj_mat = torch.tensor(cfg.sensor.P)
    gel_depth = torch.tensor(np.loadtxt(f"{BASE_PATH}/{cfg.sensor.gel_depth_map_loc}", delimiter=',') - cfg.sensor.gel_depth_offset)

    # iterate over dataset
    for ds_idx, dataset in enumerate(train_dataset):

        print(f"Dataset idx: {ds_idx:03d}")

        # load imgs: S x C x H x W , poses: (S x 3 x 3, S x 3) <-> (R, t)
        img_seq_color, img_seq_depth, img_seq_normal, pose_seq_obj, pose_seq_digit_top, pose_seq_digit_bot = dataset
        normals3d_seq = img_seq_normal.view(img_seq_normal.shape[0], img_seq_normal.shape[1], -1)  # S x 3 x N
        pose_seq_sensor = pose_seq_digit_bot if (cfg.dataloader.digit_id == "bot") else pose_seq_digit_top

        S, C, H, W = img_seq_color.shape

        # initialize open3d geometries
        clouds = vis3d.init_geometry(geom_type="cloud", num_items=1)
        frames = vis3d.init_geometry(geom_type="frame", num_items=3) # obj, digit_top, digit_bot
        meshes = vis3d.init_geometry(geom_type="mesh", num_items=3, file_names=[
                                     "textured_cube_rounded.obj", "digit.STL", "digit.STL"]) # obj, digit_top, digit_bot

        meshes[0].scale(0.25 * 0.05, center=(0, 0, 0))

        step = 1
        seq_idx = 0
        while (seq_idx < S):

            print(f"Sequence idx: {seq_idx:03d}")

            # get points, normals
            normals3d = normals3d_seq[seq_idx, :, :]
            img_depth = img_seq_depth[seq_idx, :, :] 

            # get transforms
            T_sensor = geom_utils.Rt_to_T(R=pose_seq_sensor[0][seq_idx, :], t=pose_seq_sensor[1][seq_idx, :])
            T_camera = torch.matmul(T_sensor, T_cam_offset)
            T_obj = geom_utils.Rt_to_T(R=pose_seq_obj[0][seq_idx, :], t=pose_seq_obj[1][seq_idx, :])

            # preprocess depth
            img_depth = torch.flip(img_depth, dims=[0, 1])
            img_depth[img_depth >= gel_depth] = 0

            # inverse projection
            points3d_world = geom_utils.depth_to_pts3d(depth=img_depth, P=proj_mat, V=torch.inverse(T_camera), params=cfg.sensor, ordered_pts=False)
                        
            # visualization
            visualize_geometries(vis3d=vis3d, seq_idx=seq_idx, pose_seq_obj=pose_seq_obj, pose_seq_digit_top=pose_seq_digit_top,
                            pose_seq_digit_bot=pose_seq_digit_bot, T_cam_offset=T_cam_offset, points3d=points3d_world, clouds=clouds, frames=frames, meshes=meshes, params=cfg)

            visualize_imgs(seq_idx, img_seq_depth, img_seq_normal, params=cfg)

            if not vis3d.paused.value:
                seq_idx = (seq_idx + step) % S

        
        vis3d.clear_geometries()

    vis3d.destroy()

if __name__ == '__main__':
    main()

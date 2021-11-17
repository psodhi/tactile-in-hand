# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import os

import hydra
import logging
from attrdict import AttrDict
from tqdm import tqdm

import gtsam
from graphopt_helpers import *
from tracker_factors import *

import open3d as o3d
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from inhandpy.dataio.sim_digit_data_loaders import DigitTactoImageTfSeqDataset
from inhandpy.utils import geom_utils, vis_utils, tf_utils
from inhandpy.utils.logger import Logger

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 24})
plt.ion()

log = logging.getLogger(__name__)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "config/sim_inhand_graph.yaml")        

# visualizer
view_params = AttrDict({'fov': 60, 'front': [-0.56, 0.81, 0.14], 'lookat': [
                        -0.006, -0.027, 0.033], 'up': [0.0816, -0.112, 0.990], 'zoom': 1.5})
# view_params = AttrDict({'fov': 60, 'front': [0.2, 0.95, 0.14], 'lookat': [
#                         -0.00, -0.0, 0.025], 'up': [0.0, 0.0, -0.4], 'zoom': 2.0})
vis3d = vis_utils.Visualizer3d(base_path=BASE_PATH, view_params=view_params)

fig1, axs1 = plt.subplots(nrows=1, ncols=2, num=1, clear=True, figsize=(16, 8))

def data_loader(dataset_names, datatype, params):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_list = []
    for (ds_idx, dataset_name) in enumerate(dataset_names):
        srcdir_dataset = f"{BASE_PATH}/local/datasets/{dataset_name}"
        dataset_list.append(DigitTactoImageTfSeqDataset(
            dir_dataset=f"{srcdir_dataset}/{datatype}", dataset_name=dataset_name, transform=transform,
            downsample_imgs=params.downsample_imgs, img_types=params.img_types, digit_id=params.digit_id))

    dataset = ConcatDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=params.batch_size,
                            shuffle=params.shuffle, num_workers=params.num_workers)

    return dataloader, dataset

def update_global_map(global_map_cloud, img_normal, bg_mask, proj_mat, T_cam, T_obj,
                      params=None, img_depth_gt=None, use_gt_depth=False, debug_vis=False):

    # get depth maps
    if not use_gt_depth: # normal to depth conversion
        boundary = torch.zeros((img_normal.shape[-2], img_normal.shape[-1])) # bdry for gxx, gyy
        img_depth = geom_utils.normal_to_depth(img_normal, bg_mask=bg_mask, boundary=boundary,
                                                    gel_width=params.sensor.gel_width, gel_height=params.sensor.gel_height)
        img_depth = geom_utils.mask_background(img_depth, bg_mask=bg_mask, bg_val=0.)
    else:
        img_depth = geom_utils.mask_background(img_depth_gt.squeeze(), bg_mask=bg_mask, bg_val=0.)

    img_depth = geom_utils.flip(img_depth)

    # inverse projection
    points3d = geom_utils.depth_to_pts3d(depth=img_depth, P=proj_mat,
                                         V=torch.inverse(T_cam), params=params.sensor) # world frame

    points3d = geom_utils.transform_pts3d(torch.inverse(T_obj), points3d) # object frame
    points3d = geom_utils.remove_outlier_pts(points3d, nb_neighbors=20, std_ratio=10.)
    
    points3d_map = np.asarray(global_map_cloud.points).transpose(1, 0)
    points3d_map = np.concatenate((points3d_map, points3d), axis=1) if (points3d_map.shape[-1] > 0) else points3d
    global_map_cloud = geom_utils.init_points_to_clouds(clouds=[global_map_cloud], points3d=[points3d_map])[0]

    if debug_vis:
        vis_utils.visualize_geometries_o3d(vis3d=vis3d, clouds=[global_map_cloud])
        # o3d.visualization.draw_geometries([global_map_cloud])

    return global_map_cloud

def load_map_from_file(filename, T_init, map_type='cloud'):
    
    obj_cloud_map = o3d.geometry.PointCloud()

    if (map_type == 'cloud'):
        obj_cloud_map = o3d.io.read_point_cloud(filename)
    elif (map_type == 'voxel'):
        obj_voxel_map = o3d.io.read_voxel_grid(filename)    
        obj_points_map = np.asarray([obj_voxel_map.origin + pt.grid_index*obj_voxel_map.voxel_size for pt in obj_voxel_map.get_voxels()])
        obj_points_map = geom_utils.transform_pts3d(T_init, obj_points_map.transpose(1,0))
        obj_cloud_map = geom_utils.init_points_to_clouds(clouds=[obj_cloud_map], points3d=[obj_points_map])[0]
    else:
        log.info(f"[load_map_from_file] map_type: {map_type} not found")
    
    return obj_cloud_map

def save_to_file(logger, ds_name, method_type, patch_map=None, verbose=False):

    logdir = f"{BASE_PATH}/local/datalogs/{method_type}/{ds_name}/"
    os.makedirs(logdir, exist_ok=True)

    if logger is not None: logger.write_data_to_file(csvfile=f"{logdir}/datalog.csv")
    if patch_map is not None: o3d.io.write_point_cloud(f"{logdir}/patch.ply", patch_map)    
    if verbose: log.info(f"Saved logged data to {logdir}")        

def is_keyframe(step):
    if (step % 10 == 0):
        return True
    return False

def visualize_color_normal_depth(img_color, img_normal, img_depth):
    fig1, axs1 = plt.subplots(nrows=1, ncols=3, num=1, clear=True, figsize=(16, 8))

    img_color_np = (img_color.permute(1,2,0)).cpu().detach().numpy()
    img_normal_np = (img_normal.permute(1,2,0)).cpu().detach().numpy()
    img_depth_np = img_depth.cpu().detach().numpy()

    vis_utils.visualize_imgs(fig=fig1, axs=[axs1[0], axs1[1], axs1[2]],
                            img_list=[img_color_np, img_normal_np, img_depth_np], 
                            titles=['img_color', 'img_normal', 'img_depth_recon'], cmap='coolwarm')
    plt.pause(1e-3)

def set_method_type_config(params):

    params.object.use_prior_map = False

    if (params.method_type == 'const_vel'):
        params.factors.enable_scan_to_scan_binary = False
        params.factors.enable_smoothness_velocity = True
        params.factors.enable_scan_to_map_unary = False
    elif (params.method_type == 'scan2scan'):
        params.factors.enable_scan_to_scan_binary = True
        params.factors.enable_smoothness_velocity = False
        params.factors.enable_scan_to_map_unary = False
    elif (params.method_type == 'proposed'):
        params.factors.enable_scan_to_scan_binary = True
        params.factors.enable_smoothness_velocity = True
        params.factors.enable_scan_to_map_unary = True
    elif (params.method_type == 'global'):
        params.factors.enable_scan_to_scan_binary = False
        params.factors.enable_smoothness_velocity = True
        params.factors.enable_scan_to_map_unary = True
        params.object.use_prior_map = True
    else:
        log.info(f"[update_params] method_type: {params.method_type} not found")

    return params

def run_tracker(cfg):
    
    # data loader
    train_dataloader, train_dataset = data_loader(
        dataset_names=cfg.dataset_names, datatype="test", params=cfg.dataloader)

    dataset_uids = [item.get_uids() for item in train_dataset.datasets]
    dataset_uids = [item for sublist in dataset_uids for item in sublist]

    T_cam_offset = torch.tensor(cfg.sensor.T_cam_offset)
    proj_mat = torch.tensor(cfg.sensor.P)
    gel_depth = torch.tensor((np.loadtxt(
        f"{BASE_PATH}/{cfg.sensor.gel_depth_map_loc}", delimiter=',') - cfg.sensor.gel_depth_offset).astype(np.float32))
  
    # init open3d geometries
    colors = np.array(cfg.visualize.color_palette)
    frames = vis3d.init_geometry(geom_type="frame", num_items=2)
    meshes = vis3d.init_geometry(geom_type="mesh", num_items=3, 
                                    file_names=[cfg.object.mesh_name, cfg.object.mesh_name, cfg.sensor.mesh_name],
                                    colors=[colors[2], colors[3], colors[-1]], wireframes=[True, True, True])
    meshes[0].scale(cfg.object.global_scaling * cfg.object.mesh_scaling, center=(0, 0, 0))
    meshes[1].scale(cfg.object.global_scaling * cfg.object.mesh_scaling, center=(0, 0, 0))
    
    # iterate over dataset
    for ds_idx in tqdm(range(0, len(train_dataset))):

        log.info(f"[run_tracker] Running dataset {dataset_uids[ds_idx]}")

        # init logger
        logger = Logger() if cfg.logger.enable else None
                
        # load imgs: S x C x H x W , poses: (S x 3 x 3, S x 3) <-> (R, t)
        dataset = train_dataset[ds_idx]
        img_seq_color, img_seq_depth, img_seq_normal, pose_seq_obj, pose_seq_digit_top, pose_seq_digit_bot = dataset
        pose_seq_sensor = pose_seq_digit_bot if (cfg.dataloader.digit_id == "bot") else pose_seq_digit_top

        # init seq idxs
        S, C, H, W = img_seq_color.shape
        start, skip = cfg.factors.seq_start, cfg.factors.seq_skip
        seq_idx_vec_1 = torch.arange(start, S-skip, skip)
        seq_idx_vec_2 = torch.arange(start+skip, S, skip)

        # init graph
        graphopt = GraphOpt()
        graphopt = reset_graph(graphopt)
        gt_vals = gtsam.Values()

        # init noise models
        first_pose_prior = np.array(cfg.noise_models.first_pose_prior)
        ee_pose_prior = np.array(cfg.noise_models.ee_pose_prior)
        obj_pose_prior = np.array(cfg.noise_models.obj_pose_prior)
        smoothness_velocity = np.array(cfg.noise_models.smoothness_velocity)

        scan_to_scan_binary = np.array(cfg.noise_models.scan_to_scan_binary)
        scan_to_map_unary = np.array(cfg.noise_models.scan_to_map_unary)

        # init graph with first pose
        T0_obj = geom_utils.Rt_to_T(R=pose_seq_obj[0][seq_idx_vec_1[0], :], t=pose_seq_obj[1][seq_idx_vec_1[0], :])
        T0_sensor = geom_utils.Rt_to_T(R=pose_seq_sensor[0][seq_idx_vec_1[0], :], t=pose_seq_sensor[1][seq_idx_vec_1[0], :])
        T0_camera = torch.matmul(T0_sensor, T_cam_offset)

        ee_key_0 = gtsam.symbol(ord('e'), 0)
        graphopt.init_vals.insert(ee_key_0, tf_utils.T_to_pose3(T0_camera))
        graphopt.graph = add_unary_factor(graph=graphopt.graph, keys=[ee_key_0],
                                          factor_cov=first_pose_prior, factor_meas=T0_camera)

        obj_key_0 = gtsam.symbol(ord('o'), 0)
        graphopt.init_vals.insert(obj_key_0, tf_utils.T_to_pose3(T0_obj))
        graphopt.graph = add_unary_factor(graph=graphopt.graph, keys=[obj_key_0],
                                          factor_cov=first_pose_prior, factor_meas=T0_obj)

        graphopt = optimizer_update(graphopt)
        graphopt = reset_graph(graphopt)

        # init local patch map
        if cfg.object.use_prior_map:
            filename = f"{BASE_PATH}/local/maps/{dataset_uids[ds_idx]}/prior_map.ply"
            obj_cloud_map = load_map_from_file(filename=filename, T_init=torch.eye(4), map_type=cfg.object.prior_map_type)
        else:
            obj_cloud_map = o3d.geometry.PointCloud()

        nsteps = len(seq_idx_vec_1)        
        for step in tqdm(range(0, nsteps)):

            seq_idx_1 = seq_idx_vec_1[step]
            seq_idx_2 = seq_idx_vec_2[step]

            # print(f"*** step {step:03d}, seq_idx_1 {seq_idx_1:03d}, seq_idx_2 {seq_idx_2:03d} ***")

            ## get current object and sensor transforms
            T_obj_gt_1 = copy.deepcopy(geom_utils.Rt_to_T(R=pose_seq_obj[0][seq_idx_1, :], t=pose_seq_obj[1][seq_idx_1, :]))            
            T_sensor_gt_1 = copy.deepcopy(geom_utils.Rt_to_T(R=pose_seq_sensor[0][seq_idx_1, :], t=pose_seq_sensor[1][seq_idx_1, :]))
            T_cam_gt_1 = torch.matmul(T_sensor_gt_1, T_cam_offset)
            
            T_obj_gt_2 = copy.deepcopy(geom_utils.Rt_to_T(R=pose_seq_obj[0][seq_idx_2, :], t=pose_seq_obj[1][seq_idx_2, :]))
            T_sensor_gt_2 = copy.deepcopy(geom_utils.Rt_to_T(R=pose_seq_sensor[0][seq_idx_2, :], t=pose_seq_sensor[1][seq_idx_2, :]))
            T_cam_gt_2 = torch.matmul(T_sensor_gt_2, T_cam_offset)

            ## get variable keys
            obj_key_1 = gtsam.symbol(ord('o'), step)
            obj_key_2 = gtsam.symbol(ord('o'), step + 1)
            ee_key_1 = gtsam.symbol(ord('e'), step)
            ee_key_2 = gtsam.symbol(ord('e'), step + 1)

            img_color = copy.deepcopy(img_seq_color[seq_idx_1, :])
            img_normal = copy.deepcopy(img_seq_normal[seq_idx_1, :])
            img_depth = copy.deepcopy(img_seq_depth[seq_idx_1, :])            
            bg_mask = (img_depth > gel_depth).squeeze()

            if cfg.visualize.vis_imgs:
                visualize_color_normal_depth(img_color=img_color, img_normal=img_normal, img_depth=img_depth.squeeze())

            ## update online global map
            T_cam_gt = torch.matmul(T_sensor_gt_1, T_cam_offset)
            pose_obj_est_1 = (graphopt.optimizer.calculateEstimate()).atPose3(obj_key_1)
            T_obj_est_1 = torch.tensor( (tf_utils.pose3_to_T(pose_obj_est_1)).astype(np.float32) )
            T_obj_map = T_obj_gt_1 if (step < 1.) else T_obj_est_1            

            if (not cfg.object.use_prior_map) & (is_keyframe(step)):
                obj_cloud_map = update_global_map(obj_cloud_map, img_normal=img_normal,
                                                bg_mask=bg_mask, proj_mat=proj_mat, T_cam=T_cam_gt, T_obj=T_obj_map,
                                                img_depth_gt=img_depth, params=cfg, 
                                                use_gt_depth=cfg.factors.use_gt_depth, debug_vis=False)
                                                
            ## add factor: ee unary pose factor
            graphopt.graph = add_unary_factor(graph=graphopt.graph, keys=[ee_key_2], factor_cov=ee_pose_prior, factor_meas=T_cam_gt_2)

            ## add factor: velocity smoothness factor
            if (cfg.factors.enable_smoothness_velocity) & (step > 0):
                obj_key_0 = gtsam.symbol(ord('o'), step-1)
                graphopt.graph = add_smoothness_factor(graph=graphopt.graph, keys=[obj_key_0, obj_key_1, obj_key_2], factor_cov=smoothness_velocity)

            ## add factor: scan to scan binary
            if cfg.factors.enable_scan_to_scan_binary:
                reg_idx_2 = seq_idx_2
                step_start = np.maximum(0, step)
                step_skip = 2
                for step_prev in range(step_start, step+1, step_skip):
                    reg_idx_1 = seq_idx_vec_1[step_prev]
                    # print(f"reg_idx_1: {reg_idx_1:03d}, reg_idx_2: {reg_idx_2:03d}")

                    T_obj_gt_1 = copy.deepcopy(geom_utils.Rt_to_T(R=pose_seq_obj[0][reg_idx_1, :], t=pose_seq_obj[1][reg_idx_1, :]))            
                    T_sensor_gt_1 = copy.deepcopy(geom_utils.Rt_to_T(R=pose_seq_sensor[0][reg_idx_1, :], t=pose_seq_sensor[1][reg_idx_1, :]))
                    T_cam_gt_1 = torch.matmul(T_sensor_gt_1, T_cam_offset)

                    img_normal_1 = copy.deepcopy(img_seq_normal[reg_idx_1, :])
                    img_depth_1 = copy.deepcopy(img_seq_depth[reg_idx_1, :])
                    bg_mask_1 = (img_depth_1 > gel_depth).squeeze()

                    img_normal_2 = copy.deepcopy(img_seq_normal[reg_idx_2, :])
                    img_depth_2 = copy.deepcopy(img_seq_depth[reg_idx_2, :])            
                    bg_mask_2 = (img_depth_2 > gel_depth).squeeze()
                    graphopt.graph = add_scan_to_scan_factor(graphopt.graph, step_prev, step+1, img_normal_1, img_normal_2, bg_mask_1, bg_mask_2, proj_mat, T_cam_1=T_cam_gt_1, T_cam_2=T_cam_gt_2, img_depth_gt_1=img_depth_1, img_depth_gt_2=img_depth_2,
                                                             factor_cov=scan_to_scan_binary, T_obj_gt_1=T_obj_gt_1, T_obj_gt_2=T_obj_gt_2, est_vals=graphopt.optimizer.calculateEstimate(), params=cfg, use_gt_depth=cfg.factors.use_gt_depth, debug_vis=False, logger=None)

            ## add factor: scan to map unary
            if cfg.factors.enable_scan_to_map_unary:
                img_normal = copy.deepcopy(img_seq_normal[seq_idx_2, :])
                img_depth = copy.deepcopy(img_seq_depth[seq_idx_2, :])            
                bg_mask = (img_depth > gel_depth).squeeze()

                graphopt.graph = add_scan_to_map_factor(graphopt.graph, obj_cloud_map=copy.deepcopy(obj_cloud_map), scan_idx=step+1, img_normal=img_normal,
                                                        bg_mask=bg_mask, proj_mat=proj_mat, T_cam=T_cam_gt_2, img_depth_gt=img_depth, factor_cov=scan_to_map_unary, T_obj_gt=T_obj_gt_2,
                                                        est_vals=graphopt.optimizer.calculateEstimate(), params=cfg, use_gt_depth=cfg.factors.use_gt_depth, debug_vis=False, logger=None)


            ## add factor: obj unary pose factor
            if (step % cfg.factors.obj_prior_interval == 0):
                # todo: add gaussian noise
                graphopt.graph = add_unary_factor(graph=graphopt.graph, keys=[obj_key_2], factor_cov=obj_pose_prior, factor_meas=T_obj_gt_2)

            ## init variables
            graphopt = init_vars_step(graphopt, step+1)

            ## optimize
            graphopt = optimizer_update(graphopt)

            ## print step
            graph_vals = graphopt.optimizer.calculateEstimate()
            # print_step(est_pose=graph_vals.atPose3(ee_key_2), gt_pose=tf_utils.T_to_pose3(T_cam_gt_2))
            # print_step(est_pose=graph_vals.atPose3(obj_key_2), gt_pose=tf_utils.T_to_pose3(T_obj_gt_2))

            ## visualize step
            if cfg.visualize.vis_tracker:
                T_obj_est_2 = tf_utils.pose3_to_T(graph_vals.atPose3(obj_key_2))
                points3d = geom_utils.transform_pts3d(T_obj_est_2, (np.asarray(obj_cloud_map.points)).transpose(1,0))
                obj_cloud_map_vis = geom_utils.init_points_to_clouds(clouds=[o3d.geometry.PointCloud()], points3d=[points3d])[0]

                clouds_vis = []
                if ((cfg.method_type=='proposed') | (cfg.method_type=='global')): clouds_vis = [obj_cloud_map_vis]
                vis_utils.visualize_geometries_o3d(vis3d=vis3d, clouds=clouds_vis, frames=frames,
                                    meshes=meshes, transforms=[T_obj_est_2, T_obj_gt_2, T_sensor_gt_2])

            ## log step
            if logger is not None:
                gt_vals.insert(obj_key_2, tf_utils.T_to_pose3(T_obj_gt_2))
                gt_vals.insert(ee_key_2, tf_utils.T_to_pose3(T_cam_gt_2))
                log_step(logger=logger, step=step+1, graph_poses=graph_vals, gt_poses=gt_vals)

            ## reset graph
            graphopt = reset_graph(graphopt)

            # save logger at some step freq
            if (cfg.logger.save_csv) & (step % cfg.logger.save_step_freq == 0):
                save_to_file(logger=logger, ds_name=dataset_uids[ds_idx],
                            method_type=cfg.method_type, patch_map=obj_cloud_map, verbose=False)
            
            # break episode when error exceeds a threshold
            error_pose = np.linalg.norm(gtsam.Pose3.Logmap(
                (graph_vals.atPose3(obj_key_2)).between(gt_vals.atPose3(obj_key_2))))
            
            if (error_pose > cfg.factors.max_err_pose):
                break
        
        # o3d.visualization.draw_geometries([obj_cloud_map_vis])

@hydra.main(config_path=CONFIG_PATH)
def main(cfg):
    
    cfg.prefix = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    cfg = set_method_type_config(cfg)

    run_tracker(cfg)

if __name__ == '__main__':
    main()

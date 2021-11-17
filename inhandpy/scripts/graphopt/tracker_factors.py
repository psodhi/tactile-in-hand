# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import os

import logging
from attrdict import AttrDict

import gtsam
import inhand

import open3d as o3d

import torch
import torch.nn as nn
import torch.optim as optim

from inhandpy.utils import tf_utils, geom_utils, vis_utils

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 36})
plt.ion()

log = logging.getLogger(__name__)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def scale_depth(img_depth, scale=1.):
    img_depth_scaled = scale * (img_depth - torch.min(img_depth)) + torch.min(img_depth)
    return img_depth_scaled

def get_noise_model(factor_sigmas, robust=False):

    factor_noise_model = gtsam.noiseModel_Diagonal.Sigmas(factor_sigmas)

    if robust:
        huber = gtsam.noiseModel_mEstimator_Huber.Create(1e-2)
        factor_noise_model = gtsam.noiseModel_Robust(huber, factor_noise_model)

    return factor_noise_model

def add_unary_factor(graph, keys, factor_cov, factor_meas, robust=False):

    factor_noise_model = get_noise_model(factor_cov, robust=robust)
    factor_meas_pose = tf_utils.T_to_pose3(factor_meas)
    factor = gtsam.PriorFactorPose3(keys[0], factor_meas_pose, factor_noise_model)

    graph.push_back(factor)

    return graph

def add_binary_factor(graph, keys, factor_cov, factor_meas, robust=False):

    factor_noise_model = get_noise_model(factor_cov, robust=robust)
    factor_meas_pose = tf_utils.T_to_pose3(factor_meas)
    factor = gtsam.BetweenFactorPose3(
        keys[0], keys[1], factor_meas_pose, factor_noise_model)

    graph.push_back(factor)

    return graph

def add_smoothness_factor(graph, keys, factor_cov, robust=False):

    factor_noise_model = get_noise_model(factor_cov, robust=robust)
    factor = inhand.VelocitySmoothnessFactorPose3(keys[0], keys[1], keys[2], factor_noise_model)

    graph.push_back(factor)

    return graph

def run_registration(points3d_1, points3d_2, T_init=np.eye(4), T_gt=None, reg_type='icp_pt2pl', debug_vis=False):

    cloud_1, cloud_2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

    points3d_1 = points3d_1.cpu().detach().numpy()
    points3d_2 = points3d_2.cpu().detach().numpy()
    cloud_1.points = o3d.utility.Vector3dVector(points3d_1.transpose())
    cloud_2.points = o3d.utility.Vector3dVector(points3d_2.transpose())

    cloud_1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=50))
    cloud_2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=50))
    
    if (reg_type == 'icp_pt2pt'):
        T_reg, metrics_reg = geom_utils.icp(source=cloud_1, target=cloud_2, T_init=T_init, type='point_to_point')
    elif (reg_type == 'icp_pt2pl'):
        T_reg, metrics_reg = geom_utils.icp(source=cloud_1, target=cloud_2, T_init=T_init, type='point_to_plane')
    elif (reg_type == 'fgr'):
        T_reg, metrics_reg = geom_utils.fgr(source=cloud_1, target=cloud_2, src_normals=cloud_1.normals, tgt_normals=cloud_2.normals)
    else:
        log.error(f"[run_registration] reg_type {reg_type} not found.")
    
    if debug_vis:
        colors = [[252/255., 141/255., 98/255.], [102/255., 194/255., 165/255.], [141/255., 160/255., 203/255.]] # Or, Gr, Pu
        vis_utils.visualize_registration(source=cloud_1, target=cloud_2, transformation=T_gt,
                                        vis3d=vis3d, colors=[colors[1], colors[2]])
        import time; time.sleep(0.5)
        vis_utils.visualize_registration(source=cloud_1, target=cloud_2, transformation=T_reg,
                                        vis3d=vis3d, colors=[colors[0], colors[2]])
        import time; time.sleep(0.5)

    return T_reg, metrics_reg

def add_scan_to_map_factor(graph, obj_cloud_map, scan_idx, img_normal, bg_mask, proj_mat, T_cam, img_depth_gt=None, factor_cov=None, T_obj_gt=None,est_vals=None, params=None, use_gt_depth=False, debug_vis=False, logger=None, real_flag=False):

    # get depth maps
    if not use_gt_depth: # normal to depth conversion
        boundary = torch.zeros((img_normal.shape[-2], img_normal.shape[-1])) # bdry for gxx, gyy
        img_depth = geom_utils.normal_to_depth(img_normal, bg_mask=bg_mask, boundary=boundary,
                                                    gel_width=params.sensor.gel_width, gel_height=params.sensor.gel_height)
        if real_flag: bg_mask = (img_depth >= 0.0198)
        img_depth = geom_utils.mask_background(img_depth, bg_mask=bg_mask, bg_val=0.)
    else:
        img_depth = geom_utils.mask_background(img_depth_gt.squeeze(), bg_mask=bg_mask, bg_val=0.)

    if not real_flag: img_depth = geom_utils.flip(img_depth)

    # inverse projection
    points3d = geom_utils.depth_to_pts3d(depth=img_depth, P=proj_mat,
                                         V=torch.inverse(T_cam), params=params.sensor)
    points3d = geom_utils.remove_outlier_pts(points3d, nb_neighbors=20, std_ratio=10.)

    # register scans
    obj_key_0 = gtsam.symbol(ord('o'), scan_idx-1)
    T_obj_est = tf_utils.pose3_to_T(est_vals.atPose3(obj_key_0))
    T_gt = np.linalg.inv(T_obj_gt)
    T_init = np.linalg.inv(T_obj_est)
    T_reg, metrics_reg = run_registration(points3d_1=points3d, points3d_2=torch.transpose(torch.tensor(obj_cloud_map.points), 1, 0),
                             reg_type=params.factors.reg_type, debug_vis=debug_vis, T_gt=T_gt, T_init=T_init)
        
    # add registration factor
    obj_key_1 = gtsam.symbol(ord('o'), scan_idx)
    graph = add_unary_factor(graph=graph, keys=[obj_key_1], factor_cov=factor_cov, factor_meas=np.linalg.inv(T_reg), robust=True)

    # log values
    if logger is not None:
        rot_reg, trans_reg = tf_utils.T_to_rpy_xyz(T_reg)
        rot_gt, trans_gt = tf_utils.T_to_rpy_xyz(T_gt)
        logger.log_val(names=['reg/s2m/rot', 'reg/s2m/trans', 'reg/s2m/gt/rot', 'reg/s2m/gt/trans'], 
                    vals=[rot_reg, trans_reg, rot_gt, trans_gt], index_val=scan_idx, index_name='step')

    return graph

def add_scan_to_scan_factor(graph, idx_1, idx_2, img_normal_1, img_normal_2, bg_mask_1, bg_mask_2, proj_mat,
                            T_cam_1, T_cam_2, factor_cov, img_depth_gt_1=None, img_depth_gt_2=None, use_gt_depth=False,
                            est_vals=None, T_obj_gt_1=None, T_obj_gt_2=None, params=None, debug_vis=False, logger=None, real_flag=False):

    # get depth maps
    if not use_gt_depth: # normal to depth conversion
        boundary = torch.zeros((img_normal_1.shape[-2], img_normal_1.shape[-1])) # bdry for gxx, gyy

        img_depth_1 = geom_utils.normal_to_depth(img_normal_1, bg_mask=bg_mask_1, boundary=boundary,
                                                    gel_width=params.sensor.gel_width, gel_height=params.sensor.gel_height)
        img_depth_2 = geom_utils.normal_to_depth(img_normal_2, bg_mask=bg_mask_2, boundary=boundary,
                                                    gel_width=params.sensor.gel_width, gel_height=params.sensor.gel_height)

        if real_flag: bg_mask_1 = (img_depth_1 >= 0.0198)
        if real_flag: bg_mask_2 = (img_depth_2 >= 0.0198)

        img_depth_1 = geom_utils.mask_background(img_depth_1, bg_mask=bg_mask_1, bg_val=0.)
        img_depth_2 = geom_utils.mask_background(img_depth_2, bg_mask=bg_mask_2, bg_val=0.)
    else:
        img_depth_1 = geom_utils.mask_background(img_depth_gt_1.squeeze(), bg_mask=bg_mask_1, bg_val=0.)
        img_depth_2 = geom_utils.mask_background(img_depth_gt_2.squeeze(), bg_mask=bg_mask_2, bg_val=0.)

    if not real_flag: img_depth_1 = geom_utils.flip(img_depth_1)
    if not real_flag: img_depth_2 = geom_utils.flip(img_depth_2)

    # inverse projection
    points3d_1 = geom_utils.depth_to_pts3d(depth=img_depth_1, P=proj_mat,
                                                 V=torch.inverse(T_cam_1), params=params.sensor)
    points3d_2 = geom_utils.depth_to_pts3d(depth=img_depth_2, P=proj_mat,
                                                 V=torch.inverse(T_cam_2), params=params.sensor)    

    points3d_1 = geom_utils.remove_outlier_pts(points3d_1, nb_neighbors=20, std_ratio=10.)
    points3d_2 = geom_utils.remove_outlier_pts(points3d_2, nb_neighbors=20, std_ratio=10.)
    
    # init registration
    init_prev_delta = (idx_1 - (idx_2 - idx_1) > 0)
    if init_prev_delta:
        obj_key_0 = gtsam.symbol(ord('o'), idx_1 - (idx_2 - idx_1))
        obj_key_1 = gtsam.symbol(ord('o'), idx_1)
        T_init = tf_utils.pose3_to_T((est_vals.atPose3(obj_key_0)).between(est_vals.atPose3(obj_key_1)))
    else:
        T_init = np.eye(4)

    # register scans
    T_gt = torch.matmul(T_obj_gt_2, torch.inverse(T_obj_gt_1)) # left multiply
    T_reg, metrics_reg = run_registration(points3d_1=points3d_1, points3d_2=points3d_2,
                             reg_type=params.factors.reg_type, debug_vis=debug_vis, T_gt=T_gt, T_init=T_init)
        
    # add registration factor
    obj_key_1, obj_key_2 = gtsam.symbol(ord('o'), idx_1), gtsam.symbol(ord('o'), idx_2)
    T_obj_est_1 = tf_utils.pose3_to_T(est_vals.atPose3(obj_key_1))
    T_meas = np.matmul(np.linalg.inv(T_obj_est_1), np.matmul(T_reg, T_obj_est_1))
    graph = add_binary_factor(graph=graph, keys=[obj_key_1, obj_key_2],
                                factor_cov=factor_cov, factor_meas=T_meas, robust=True)

    # log values
    if logger is not None:
        rot_reg, trans_reg = tf_utils.T_to_rpy_xyz(T_reg)
        rot_gt, trans_gt = tf_utils.T_to_rpy_xyz(T_gt)
        logger.log_val(names=['reg/s2s/rot', 'reg/s2s/trans', 'reg/s2s/gt/rot', 'reg/s2s/gt/trans'], 
                    vals=[rot_reg, trans_reg, rot_gt, trans_gt], index_val=idx_2, index_name='step')

    return graph

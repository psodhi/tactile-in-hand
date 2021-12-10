# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch

import open3d as o3d
import copy

from scipy import ndimage
from inhandpy.thirdparty import poisson
from inhandpy.utils import vis_utils

"""
3D registration functions
"""

def fpfh(pcd, normals):
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=64))
    return pcd_fpfh


def fast_global_registration(source, target, source_fpfh, target_fpfh):
    distance_threshold = 0.01
    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))

    transformation = result.transformation
    metrics = [result.fitness, result.inlier_rmse, result.correspondence_set]

    return transformation, metrics

def fgr(source, target, src_normals, tgt_normals):
    source_fpfh = fpfh(source, src_normals)
    target_fpfh = fpfh(target, tgt_normals)
    transformation, metrics = fast_global_registration(source=source,
                                 target=target,
                                 source_fpfh=source_fpfh,
                                 target_fpfh=target_fpfh)
    return transformation, metrics

def icp(source, target, T_init=np.eye(4), mcd=0.1, max_iter=50, type='point_to_plane'):

    if (type == 'point_to_point'):
        result = o3d.pipelines.registration.registration_icp(source=source, target=target,
                                                            max_correspondence_distance=mcd, init=T_init,
                                                            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                                                            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))
    else: 
        result = o3d.pipelines.registration.registration_icp(source=source, target=target,
                                                            max_correspondence_distance=mcd, init=T_init,
                                                            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter))

    transformation = result.transformation
    metrics = [result.fitness, result.inlier_rmse, result.correspondence_set]

    return transformation, metrics

"""
Open3D helper functions
"""

def remove_outlier_pts(points3d, nb_neighbors=20, std_ratio=10., vis=False):

    points3d_np = points3d.cpu().detach().numpy() if torch.is_tensor(points3d) else points3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = copy.deepcopy(o3d.utility.Vector3dVector(points3d_np.transpose()))
    cloud_filt, ind_filt = cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    if vis: vis_utils.visualize_inlier_outlier(cloud, ind_filt)   

    points3d_filt = np.asarray(cloud_filt.points).transpose()
    points3d_filt = torch.tensor(points3d_filt) if torch.is_tensor(points3d) else points3d_filt

    return points3d_filt

def remove_background_pts(pts, bg_mask=None):

    if bg_mask is not None:
        fg_mask_pts = ~bg_mask.view(-1)
        points3d_x = pts[0, fg_mask_pts].view(1, -1)
        points3d_y = pts[1, fg_mask_pts].view(1, -1)
        points3d_z = pts[2, fg_mask_pts].view(1, -1)
        pts_fg = torch.cat((points3d_x, points3d_y, points3d_z), dim=0)
    else:
        pts_fg = pts

    return pts_fg

def init_points_to_clouds(clouds, points3d, colors=None):

    for idx, cloud in enumerate(clouds):
        points3d_np = points3d[idx].cpu().detach().numpy() if torch.is_tensor(points3d[idx]) else points3d[idx]
        cloud.points = copy.deepcopy(o3d.utility.Vector3dVector(points3d_np.transpose()))
        if colors is not None: cloud.paint_uniform_color(colors[idx])

    return clouds
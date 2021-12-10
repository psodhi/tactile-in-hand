
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import math

import torch
import pytorch3d.transforms as p3d_t

"""
dataloader, logger helpers
"""

def pandas_col_to_numpy(df_col):

    df_col = df_col.apply(lambda x: np.fromstring(x.replace('\n', '').replace(
        '[', '').replace(']', '').replace('  ', ' '), sep=', '))
    df_col = np.stack(df_col)

    return df_col

def pandas_string_to_numpy(arr_str):
    arr_npy = np.fromstring(arr_str.replace('\n', '').replace(
        '[', '').replace(']', '').replace('  ', ' '), sep=', ')
    return arr_npy

def interpolate_img(img, rows, cols):
    '''
    img: C x H x W
    '''
    
    img = torch.nn.functional.interpolate(img, size=cols)
    img = torch.nn.functional.interpolate(img.permute(0, 2, 1), size=rows)
    img = img.permute(0, 2, 1)
    
    return img

"""
common functions
"""

def flip(x): 
    return torch.flip(x, dims=[0])

def min_clip(x, min_val): 
    return torch.max(x, min_val)

def max_clip(x, max_val): 
    return torch.min(x, max_val)

def normalize(x, min_val, max_val): 
    return (x - torch.min(x)) * (max_val - min_val) / (torch.max(x) - torch.min(x)) + min_val

def mask_background(x, bg_mask, bg_val=0.):
    
    if bg_mask is not None: x[bg_mask] = bg_val
    
    return x 

"""
quant metrics 
"""

def wrap_to_pi(arr):
    arr_wrap = (arr + math.pi) % (2 * math.pi) - math.pi
    return arr_wrap

def traj_error_trans(xyz_gt, xyz_est):

    diff = xyz_gt - xyz_est
    diff_sq = diff**2

    rmse_trans = np.sqrt(np.mean(diff_sq.flatten()))
    error = rmse_trans

    return error

def traj_error_rot(rot_mat_gt, rot_mat_est, convention="XYZ"):

    if (torch.is_tensor(rot_mat_gt) & torch.is_tensor(rot_mat_est)):
        rot_rpy_gt = p3d_t.matrix_to_euler_angles(rot_mat_gt, convention)
        rot_rpy_est = p3d_t.matrix_to_euler_angles(rot_mat_est, convention)
    else:
        rot_rpy_gt = (p3d_t.matrix_to_euler_angles(torch.tensor(rot_mat_gt), convention)).cpu().detach().numpy()
        rot_rpy_est = (p3d_t.matrix_to_euler_angles(torch.tensor(rot_mat_est), convention)).cpu().detach().numpy()

    diff = wrap_to_pi(rot_rpy_gt - rot_rpy_est)
    diff_sq = diff**2

    rmse_rot = np.sqrt(np.mean(diff_sq.flatten()))
    error = rmse_rot

    return error

"""
transform helper functions
"""

def Rt_to_T(R=None, t=None, device=None):
    """
    :param R: rotation, (B, 3, 3) or (3, 3)
    :param t: translation, (B, 3) or  (3)
    :return: T, (B, 4, 4) or  (4, 4)
    """

    T = torch.eye(4, device=device)
    
    if ((len(R.shape) > 2) & (len(t.shape) > 1)): # batch version
        B = R.shape[0]
        T = T.repeat(B, 1, 1)
        if R is not None: T[:, 0:3, 0:3] = R
        if t is not None: T[:, 0:3, -1] = t

    else:
        if R is not None: T[0:3, 0:3] = R
        if t is not None: T[0:3, -1] = t

    return T

def transform_pts3d(T, pts):
    """
    T: 4x4
    pts: 3xN

    returns 3xN
    """
    D, N = pts.shape
        
    if (D == 3): 
        pts_tf = torch.cat((pts, torch.ones(1, N)), dim=0) if torch.is_tensor(pts) else np.concatenate((pts, torch.ones(1, N)), axis=0)

    pts_tf = torch.matmul(T, pts_tf) if torch.is_tensor(pts) else np.matmul(T, pts_tf)
    pts_tf = pts_tf[0:3, :]

    return pts_tf
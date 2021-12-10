# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch

import open3d as o3d
import copy

from scipy import ndimage
from inhandpy.thirdparty import poisson
from inhandpy.utils import vis_utils

"""
3D-2D projection / conversion functions
OpenGL transforms reference: http://www.songho.ca/opengl/gl_transform.html
"""

def vectorize_pixel_coords(rows, cols, device=None):

    y_range = torch.arange(rows, device=device)
    x_range = torch.arange(cols, device=device)
    grid_x, grid_y = torch.meshgrid(x_range, y_range)
    pixel_pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=0) # 2 x N

    return pixel_pos

def clip_to_pixel(clip_pos, img_shape, params):
    
    H, W = img_shape
    
    # clip -> ndc coords
    x_ndc = clip_pos[0, :] / clip_pos[3, :]
    y_ndc = clip_pos[1, :] / clip_pos[3, :]
    # z_ndc = clip_pos[2, :] / clip_pos[3, :]

    # ndc -> pixel coords
    x_pix = (W-1) / 2 * (x_ndc + 1) # [-1, 1] -> [0, W-1]
    y_pix = (H-1) / 2 * (y_ndc + 1) # [-1, 1] -> [0, H-1]
    # z_pix = (f-n) / 2 *  z_ndc + (f+n) / 2 

    pixel_pos = torch.stack((x_pix, y_pix), dim=0)

    return pixel_pos

def pixel_to_clip(pix_pos, depth_map, params):
    """
    :param pix_pos: position in pixel space, (2, N)
    :param depth_map: depth map, (H, W)
    :return: clip_pos position in clip space, (4, N)
    """

    x_pix = pix_pos[0, :]
    y_pix = pix_pos[1, :]

    H, W = depth_map.shape
    f = params.z_far
    n = params.z_near

    # pixel -> ndc coords
    x_ndc = 2/(W-1) * x_pix - 1 # [0, W-1] -> [-1, 1]
    y_ndc = 2/(H-1) * y_pix - 1 # [0, H-1] -> [-1, 1]
    z_buf = depth_map[y_pix, x_pix]

    # ndc -> clip coords
    z_eye = -z_buf
    w_c = -z_eye
    x_c = x_ndc * w_c
    y_c = y_ndc * w_c
    z_c = -(f+n)/(f-n) * z_eye - 2*f*n/(f-n) * 1.

    clip_pos = torch.stack([x_c, y_c, z_c, w_c], dim=0)
    
    return clip_pos

def clip_to_eye(clip_pos, P):

    P_inv = torch.inverse(P)
    eye_pos = torch.matmul(P_inv, clip_pos)

    return eye_pos

def eye_to_clip(eye_pos, P):

    clip_pos = torch.matmul(P, eye_pos)

    return clip_pos

def eye_to_world(eye_pos, V):

    V_inv = torch.inverse(V)
    world_pos = torch.matmul(V_inv, eye_pos)

    world_pos = world_pos / world_pos[3, :]

    return world_pos

def world_to_eye(world_pos, V):

    eye_pos = torch.matmul(V, world_pos)

    return eye_pos

def world_to_object(world_pos, M):

    M_inv = torch.inverse(M)
    obj_pos = torch.matmul(M_inv, world_pos)

    obj_pos = obj_pos / obj_pos[3, :]

    return obj_pos

def object_to_world(obj_pos, M):

    world_pos = torch.matmul(M, obj_pos)

    world_pos = world_pos / world_pos[3, :]

    return world_pos

def depth_to_pts3d(depth, P, V, params=None, ordered_pts=False):
    """
    :param depth: depth map, (C, H, W) or (H, W)
    :param P: projection matrix, (4, 4)
    :param V: view matrix, (4, 4)
    :return: world_pos position in 3d world coordinates, (3, H, W) or (3, N)
    """
    
    assert (2 <= len(depth.shape) <= 3)
    assert (P.shape == (4, 4))
    assert (V.shape == (4, 4))

    depth_map = depth.squeeze(0) if (len(depth.shape) == 3) else depth
    H, W = depth_map.shape
    pixel_pos = vectorize_pixel_coords(rows=H, cols=W)

    clip_pos = pixel_to_clip(pixel_pos, depth_map, params)
    eye_pos = clip_to_eye(clip_pos, P)
    world_pos = eye_to_world(eye_pos, V)

    world_pos = world_pos[0:3, :] / world_pos[3, :]

    if ordered_pts:
        H, W = depth_map.shape
        world_pos = world_pos.reshape(world_pos.shape[0], H, W)
    
    return world_pos

def analytic_flow(img1, depth1, P, V1, V2, M1, M2, gel_depth, params):
            
    C, H, W = img1.shape
    depth_map = depth1.squeeze(0) if (len(depth1.shape) == 3) else depth1
    pixel_pos = vectorize_pixel_coords(rows=H, cols=W, device=img1.device)

    clip_pos = pixel_to_clip(pixel_pos, depth_map, params)
    eye_pos = clip_to_eye(clip_pos, P)
    world_pos = eye_to_world(eye_pos, V1)
    obj_pos = world_to_object(world_pos, M1)
    
    world_pos = object_to_world(obj_pos, M2)
    eye_pos = world_to_eye(world_pos, V2)
    clip_pos = eye_to_clip(eye_pos, P)
    pixel_pos_proj = clip_to_pixel(clip_pos, (H, W), params)
    
    pixel_flow = pixel_pos - pixel_pos_proj
    flow_map = pixel_flow.reshape(pixel_flow.shape[0], H, W)
    
    # mask out background gel pixels
    mask_idxs = (depth_map >= gel_depth)
    flow_map[:, mask_idxs] = 0.

    return flow_map
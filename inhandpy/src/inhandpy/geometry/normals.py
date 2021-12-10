# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
from scipy import ndimage
from inhandpy.thirdparty import poisson
from inhandpy.utils import utils

"""
Normal to depth conversion / integration functions
"""

def preproc_depth(img_depth, bg_mask=None):

    if bg_mask is not None:
        img_depth = utils.utils.mask_background(img_depth, bg_mask=bg_mask, bg_val=0.)

    return img_depth

def preproc_normal(img_normal, bg_mask=None):
    '''
    img_normal: lies in range [0, 1]
    '''

    # 0.5 corresponds to 0
    img_normal = img_normal - 0.5

    # normalize
    img_normal = img_normal / torch.linalg.norm(img_normal, dim=0)

    # set background to have only z normals (flat, facing camera)
    if bg_mask is not None:
        img_normal[0:2, bg_mask] = 0.
        img_normal[2, bg_mask] = 1.0

    return img_normal

def depth_to_grad_depth(img_depth, bg_mask=None):
    gradx = ndimage.sobel(img_depth.cpu().detach().numpy(), axis=1, mode='constant')
    grady = ndimage.sobel(img_depth.cpu().detach().numpy(), axis=0, mode='constant')

    gradx = torch.FloatTensor(gradx, device=img_depth.device)
    grady = torch.FloatTensor(grady, device=img_depth.device)

    if bg_mask is not None:
        gradx = utils.mask_background(gradx, bg_mask=bg_mask, bg_val=0.)
        grady = utils.mask_background(grady, bg_mask=bg_mask, bg_val=0.)

    return gradx, grady

def normal_to_grad_depth(img_normal, gel_width=1., gel_height=1., bg_mask=None):

    # Ref: https://stackoverflow.com/questions/34644101/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-produc/34644939#34644939

    EPS = 1e-1
    nz = torch.max(torch.tensor(EPS), img_normal[2, :])

    dzdx = -(img_normal[0, :] / nz).squeeze()
    dzdy = -(img_normal[1, :] / nz).squeeze()

    # taking out negative sign as we are computing gradient of depth not z
    # since z is pointed towards sensor, increase in z corresponds to decrease in depth
    # i.e., dz/dx = -ddepth/dx
    ddepthdx = -dzdx
    ddepthdy = -dzdy

    # pixel axis v points in opposite dxn of camera axis y
    ddepthdu = ddepthdx
    ddepthdv = -ddepthdy

    gradx = ddepthdu # cols
    grady = ddepthdv # rows

    # convert units from pixel to meters
    C, H, W = img_normal.shape
    gradx = gradx * (gel_width / W)
    grady = grady * (gel_height / H)

    if bg_mask is not None:
        gradx = utils.mask_background(gradx, bg_mask=bg_mask, bg_val=0.)
        grady = utils.mask_background(grady, bg_mask=bg_mask, bg_val=0.)

    return gradx, grady

def integrate_grad_depth(gradx, grady, boundary=None, bg_mask=None, max_depth=0.0):
    
    if boundary is None:
        boundary = torch.zeros((gradx.shape[0], gradx.shape[1]))

    img_depth_recon = poisson.poisson_reconstruct(grady.cpu().detach(
    ).numpy(), gradx.cpu().detach().numpy(), boundary.cpu().detach().numpy())
    img_depth_recon = torch.FloatTensor(img_depth_recon, device=gradx.device)

    if bg_mask is not None:
        img_depth_recon = utils.mask_background(img_depth_recon, bg_mask)

    # after integration, img_depth_recon lies between 0. (bdry) and a -ve val (obj depth)
    # rescale to make max depth as gel depth and obj depth as +ve values
    img_depth_recon = utils.max_clip(img_depth_recon, max_val=torch.tensor(0.)) + max_depth

    return img_depth_recon

def depth_to_depth(img_depth, bg_mask=None, boundary=None, params=None):

    # preproc depth img
    img_depth = preproc_depth(img_depth=img_depth.squeeze(), bg_mask=bg_mask)

    # get grad depth
    gradx, grady = depth_to_grad_depth(img_depth=img_depth.squeeze(), bg_mask=bg_mask)

    # integrate grad depth
    img_depth_recon = integrate_grad_depth(gradx, grady, boundary=boundary, bg_mask=bg_mask)

    return img_depth_recon

def normal_to_depth(img_normal, bg_mask=None, boundary=None, gel_width=0.02, gel_height=0.03, max_depth=0.02):

    # preproc normal img
    img_normal = preproc_normal(img_normal=img_normal, bg_mask=bg_mask)
    
    # get grad depth
    gradx, grady = normal_to_grad_depth(img_normal=img_normal, gel_width=gel_width,
                                         gel_height=gel_height, bg_mask=bg_mask)

    # integrate grad depth
    img_depth_recon = integrate_grad_depth(gradx, grady, boundary=boundary, bg_mask=bg_mask, max_depth=max_depth)

    return img_depth_recon

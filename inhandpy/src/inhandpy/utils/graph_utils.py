
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import gtsam
import torch

def vec3_to_pose2(vec):
    return gtsam.Pose2(vec[0], vec[1], vec[2])

def pose2_to_vec3(pose2):
    return [pose2.x(), pose2.y(), pose2.theta()]

def T_to_pose3(T):

    T_local = T
    if torch.is_tensor(T): T_local = T.cpu().detach().numpy() 

    return gtsam.Pose3(T_local)

def pose3_to_T(pose3):
    return pose3.matrix()

def T_to_rpy_xyz(T):

    T_local = T
    if torch.is_tensor(T): T_local = T.cpu().detach().numpy() 

    pose3 = gtsam.Pose3(T_local)
    rot_rpy = (pose3.rotation().rpy()).tolist()
    trans_xyz = [pose3.x(), pose3.y(), pose3.z()]

    return (rot_rpy, trans_xyz)

def add_gaussian_noise(pose, noisevec, add_noise=True):
    if (add_noise):
        pose_noisy = pose.retract(noisevec)
    else:
        pose_noisy = pose

    return pose_noisy
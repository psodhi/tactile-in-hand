
# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch

import pytorch3d.transforms as p3d_t
# from lietorch import SO3, SE3
import gtsam

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

def tf2d_between(pose1, pose2, device=None):
    """
    Relative transform of pose2 in pose1 frame, i.e. T12 = T1^{1}*T2
    :param pose1: n x 3 tensor [x,y,yaw]
    :param pose2: n x 3 tensor [x,y,yaw]
    :return: pose12 n x 3 tensor [x,y,yaw]
    """

    num_data = pose1.shape[0]

    rot1 = torch.cat([torch.zeros(num_data, 1, device=device), torch.zeros(
        num_data, 1, device=device), pose1[:, 2][:, None]], 1)
    rot2 = torch.cat([torch.zeros(num_data, 1, device=device), torch.zeros(
        num_data, 1, device=device), pose2[:, 2][:, None]], 1)
    t1 = torch.cat([pose1[:, 0][:, None], pose1[:, 1]
                    [:, None], torch.zeros(num_data, 1, device=device)], 1)
    t2 = torch.cat([pose2[:, 0][:, None], pose2[:, 1]
                    [:, None], torch.zeros(num_data, 1, device=device)], 1)

    R1 = p3d_t.euler_angles_to_matrix(rot1, "XYZ")
    R2 = p3d_t.euler_angles_to_matrix(rot2, "XYZ")
    R1t = torch.inverse(R1)

    R12 = torch.matmul(R1t, R2)
    rot12 = p3d_t.matrix_to_euler_angles(R12, "XYZ")
    t12 = torch.matmul(R1t, (t2-t1)[:, :, None])
    t12 = t12[:, :, 0]

    tx = t12[:, 0][:, None]
    ty = t12[:, 1][:, None]
    yaw = rot12[:, 2][:, None]
    pose12 = torch.cat([tx, ty, yaw], 1)

    return pose12

def tf2d_compose(pose1, pose12):
    """
    Composing pose1 with pose12, i.e. T2 = T1*T12
    :param pose1: n x 3 tensor [x,y,yaw]
    :param pose12: n x 3 tensor [x,y,yaw]
    :return: pose2 n x 3 tensor [x,y,yaw]
    """

    num_data = pose1.shape[0]

    rot1 = torch.cat([torch.zeros(num_data, 1), torch.zeros(
        num_data, 1), pose1[:, 2][:, None]], 1)
    rot12 = torch.cat([torch.zeros(num_data, 1), torch.zeros(
        num_data, 1), pose12[:, 2][:, None]], 1)
    t1 = torch.cat([pose1[:, 0][:, None], pose1[:, 1]
                    [:, None], torch.zeros(num_data, 1)], 1)
    t12 = torch.cat([pose12[:, 0][:, None], pose12[:, 1]
                    [:, None], torch.zeros(num_data, 1)], 1)

    R1 = p3d_t.euler_angles_to_matrix(rot1, "XYZ")
    R12 = p3d_t.euler_angles_to_matrix(rot12, "XYZ")

    R2 = torch.matmul(R1, R12)
    rot2 = p3d_t.matrix_to_euler_angles(R2, "XYZ")
    t2 = torch.matmul(R1, t12[:, :, None]) + t1[:, :, None]
    t2 = t2[:, :, 0]

    tx = t2[:, 0][:, None]
    ty = t2[:, 1][:, None]
    yaw = rot2[:, 2][:, None]
    pose2 = torch.cat([tx, ty, yaw], 1)

    return pose2

def tf3d_between(pose1, pose2, device=None, rot_order="XYZ"):
    """
    Relative transform of pose2 in pose1 frame, i.e. T12 = T1^{1}*T2
    :param pose1: n x 6 tensor
    :param pose2: n x 6 tensor
    :return: pose12 n x 6 tensor
    """

    num_data = pose1.shape[0]

    t1 = pose1[:, 0:3]
    t2 = pose2[:, 0:3]

    ang1 = pose1[:, 3:6]
    ang2 = pose2[:, 3:6]

    R1 = p3d_t.euler_angles_to_matrix(ang1, rot_order)
    R2 = p3d_t.euler_angles_to_matrix(ang2, rot_order)

    R1t = torch.inverse(R1)
    R12 = torch.matmul(R1t, R2)
    t12 = torch.matmul(R1t, (t2-t1).unsqueeze(2)).squeeze(2)

    # ang12 = p3d_t.matrix_to_euler_angles(R12, rot_order)    
    # pose12 = torch.cat([t12, ang12], 1)

    T12 = torch.eye(4)
    T12[0:3, 0:3] = R12[0,:]
    T12[0:3, -1] = t12[0,:]

    return T12

# def tf3d_compose(pose1, pose12, rot_order="ZYX"):
#     """
#     Composing pose1 with pose12, i.e. T2 = T1*T12
#     :param pose1: n x 6 tensor [x,y,yaw]
#     :param pose12: n x 6 tensor [x,y,yaw]
#     :return: pose2 n x 6 tensor [x,y,yaw]
#     """

#     num_data = pose1.shape[0]

#     rot1 = pose1[:, 3:6]
#     rot12 = pose12[:, 3:6]

#     t1 = pose1[:, 0:3]
#     t12 = pose12[:, 0:3]

#     R1 = p3d_t.euler_angles_to_matrix(rot1, rot_order)
#     R12 = p3d_t.euler_angles_to_matrix(rot12, rot_order)

#     R2 = torch.matmul(R1, R12)

#     rot2 = p3d_t.matrix_to_euler_angles(R2, rot_order)
#     t2 = torch.matmul(R1, t12) + t1
#     t2 = t2[:, :, 0]

#     tx = t2[:, 0][:, None]
#     ty = t2[:, 1][:, None]
#     yaw = rot2[:, 2][:, None]
#     pose2 = torch.cat([tx, ty, yaw], 1)

#     return pose2

def tf3d_between_lie(xi1, xi2):
    """
    Relative transform of pose2 in pose1 frame, i.e. T12 = T1^{1}*T2
    :param xi1: n x 6 tensor [x,y,z,roll,pitch,yaw]
    :param xi2: n x 6 tensor [x,y,z,roll,pitch,yaw]
    :return: xi12 n x 6 tensor [x,y,z,roll,pitch,yaw]
    """
    
    pose1 = SE3.exp(xi1)
    pose2 = SE3.exp(xi2)

    pose12 = pose1.inv() * pose2
    xi12 = pose12.log() 

    return xi12

def tf3d_compose_lie(xi1, xi12):
    """
    Composing pose1 with pose12, i.e. T2 = T1*T12
    :param xi1: n x 6 tensor [x,y,z,roll,pitch,yaw]
    :param xi12: n x 6 tensor [x,y,z,roll,pitch,yaw]
    :return: xi2 n x 6 tensor [x,y,z,roll,pitch,yaw]
    """
    
    pose1 = SE3.exp(xi1)
    pose12 = SE3.exp(xi12)

    pose2 = pose1 * pose12
    xi2 = pose2.log() 

    return xi2

def tf2d_net_input(pose_rel):
    tx = pose_rel[:, 0][:, None]  # N x 1
    ty = pose_rel[:, 1][:, None]  # N x 1
    yaw = pose_rel[:, 2][:, None]  # N x 1
    pose_rel_net = torch.cat(
        [tx*1000, ty*1000, torch.cos(yaw), torch.sin(yaw)], 1)  # N x 4
    return pose_rel_net
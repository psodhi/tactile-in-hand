# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python

import sys
sys.path.append("/usr/local/cython/")

import numpy as np
import os

import logging

import gtsam

import open3d as o3d
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from inhandpy.utils import tf_utils

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 36})
plt.ion()

log = logging.getLogger(__name__)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class GraphOpt():
    def __init__(self):

        self.graph = gtsam.NonlinearFactorGraph()
        self.init_vals = gtsam.Values()

        self.optimizer = self.init_isam2()
        self.est_vals = gtsam.Values()

    def init_isam2(self):
        params_isam2 = gtsam.ISAM2Params()
        params_isam2.setRelinearizeThreshold(0.01)
        params_isam2.setRelinearizeSkip(10)

        return gtsam.ISAM2(params_isam2)

def reset_graph(graphopt):    
    graphopt.graph.resize(0)
    graphopt.init_vals.clear()

    return graphopt

def optimizer_update(graphopt):
    graphopt.optimizer.update(graphopt.graph, graphopt.init_vals)
    graphopt.est_vals = graphopt.optimizer.calculateEstimate()

    return graphopt

def init_vars_step(graphopt, step):

    key_tm1 = gtsam.symbol(ord('o'), step-1)
    key_t = gtsam.symbol(ord('o'), step)
    graphopt.init_vals.insert(key_t, graphopt.est_vals.atPose3(key_tm1))

    key_tm1 = gtsam.symbol(ord('e'), step-1)
    key_t = gtsam.symbol(ord('e'), step)
    graphopt.init_vals.insert(key_t, graphopt.est_vals.atPose3(key_tm1))

    return graphopt

def log_step_curr_pose(logger, step, est_pose, gt_pose):

    rot_est = (est_pose.rotation().rpy()).tolist()
    trans_est = [est_pose.x(), est_pose.y(), est_pose.z()]

    rot_gt = (gt_pose.rotation().rpy()).tolist()
    trans_gt = [gt_pose.x(), gt_pose.y(), gt_pose.z()]

    logger.log_val(names=['graphopt/est/rot', 'graphopt/est/trans', 'graphopt/gt/rot', 'graphopt/gt/trans'],
                   vals=[rot_est, trans_est, rot_gt, trans_gt], index_val=step, index_name='step')

    return logger

def log_step(logger, step, graph_poses, gt_poses):

    nposes = step + 1

    graph_obj_rot_list, graph_obj_trans_list = [], []
    graph_ee_rot_list, graph_ee_trans_list = [], []
    
    gt_obj_rot_list, gt_obj_trans_list = [], []
    gt_ee_rot_list, gt_ee_trans_list = [], []

    for i in range(1, nposes):
        obj_key = gtsam.symbol(ord('o'), i)
        ee_key = gtsam.symbol(ord('e'), i)

        # read graph obj poses
        obj_pose3d = graph_poses.atPose3(obj_key)        
        graph_obj_rot_list.append((obj_pose3d.rotation().matrix()).tolist())
        graph_obj_trans_list.append([obj_pose3d.x(), obj_pose3d.y(), obj_pose3d.z()]) 

        # read graph ee poses
        ee_pose3d = graph_poses.atPose3(ee_key)
        graph_ee_rot_list.append((ee_pose3d.rotation().matrix()).tolist())
        graph_ee_trans_list.append([ee_pose3d.x(), ee_pose3d.y(), ee_pose3d.z()]) 

        # read gt obj poses
        obj_pose3d = gt_poses.atPose3(obj_key)        
        gt_obj_rot_list.append((obj_pose3d.rotation().matrix()).tolist())
        gt_obj_trans_list.append([obj_pose3d.x(), obj_pose3d.y(), obj_pose3d.z()]) 

        # read gt ee poses
        ee_pose3d = graph_poses.atPose3(ee_key)
        gt_ee_rot_list.append((ee_pose3d.rotation().matrix()).tolist())
        gt_ee_trans_list.append([ee_pose3d.x(), ee_pose3d.y(), ee_pose3d.z()])

    logger.log_val(names=['graph/obj/rot', 'graph/obj/trans', 'graph/ee/rot', 'graph/ee/trans'],
                   vals=[graph_obj_rot_list, graph_obj_trans_list, graph_ee_rot_list, graph_ee_trans_list], 
                   index_val=step, index_name='step')

    logger.log_val(names=['gt/obj/rot', 'gt/obj/trans', 'gt/ee/rot', 'gt/ee/trans'],
                   vals=[gt_obj_rot_list, gt_obj_trans_list, gt_ee_rot_list, gt_ee_trans_list], 
                   index_val=step, index_name='step')

    return logger

def print_step(est_pose, gt_pose, type='', decimals=3):

    rot_est = np.round(est_pose.rotation().rpy(), decimals=decimals)
    tran_est = np.round([est_pose.x(), est_pose.y(), est_pose.z()], decimals=decimals)

    rot_gt = np.round(gt_pose.rotation().rpy(), decimals=decimals)
    tran_gt = np.round([gt_pose.x(), gt_pose.y(), gt_pose.z()], decimals=decimals)

    err_pose = gt_pose.between(est_pose)
    rot_err = np.round(err_pose.rotation().rpy(), decimals=decimals)
    tran_err = np.round([err_pose.x(), err_pose.y(), err_pose.z()], decimals=decimals)

    print(f'ESTIMATED {type} pose\n rot_est: {rot_est}, tran_est: {tran_est}')
    print(f'GROUNDTRUTH {type} pose\n rot_gt: {rot_gt}, tran_gt: {tran_gt}')
    print(f'ERROR {type} pose\n rot_err: {rot_err}, tran_err: {tran_err}')

# Copyright (c) Facebook, Inc. and its affiliates.

#%%

import numpy as np
import math

import glob
import pandas as pd

import matplotlib.pyplot as plt
from inhandpy.utils import vis_utils
import open3d as o3d

import time

plt.rcParams.update({'font.size': 28})
plt.ion()

#%% Set source directories

BASE_PATH = '/home/paloma/code/fair_ws/tactile-in-hand/inhandpy'
logdir = f'{BASE_PATH}/local/datalogs/'

# dataset = 'tacto/cube/'
dataset = 'real/cube/'
label = 'const_vel' # [const_vel, scan2scan, proposed, global]

#%% helper functions

def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(5.0, -0.0)

def pan_scene(vis, max=720):
    for i in range(0, max):
        rotate_view(vis)
        render(vis)

def render(vis):
    vis.poll_events()
    vis.update_renderer()
    
    time.sleep(0.01)


#%% visualize cloud

cloudfiles = sorted(glob.glob(f'{logdir}/{label}/{dataset}/**/*.ply', recursive=True))
print(cloudfiles)


vis3d = o3d.visualization.Visualizer()
vis3d.create_window()

for idx in range(2, len(cloudfiles)):

    cloud = o3d.io.read_point_cloud(cloudfiles[idx])
    # o3d.visualization.draw_geometries([cloud])

    vis3d.add_geometry(cloud)
    vis3d.run()
    pan_scene(vis3d)
    vis3d.destroy_window()

    break
# %%

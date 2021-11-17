# Copyright (c) Facebook, Inc. and its affiliates.

#%%

import numpy as np
import math
import glob
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 28})
plt.ion()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib import animation

plt.rcParams.update({'font.size': 28})
plt.ion()

BASE_PATH = '/home/paloma/code/fair_ws/tactile-in-hand/inhandpy'
logdir = f'{BASE_PATH}/local/datalogs/'

#%% Helper functions

def pandas_string_to_numpy(arr_str):
    arr_npy = np.fromstring(arr_str.replace('\n', '').replace(
        '[', '').replace(']', '').replace('  ', ' '), sep=', ')
    return arr_npy

def transform_axes_tips(R, t=None, scale=1):
    """
    R: Rotation matrix of shape (N,3,3)
    t: Translation matrix of shape (N,3)
    scale: Length of axes
    Returns (N, 4, 3) coordinate of 4 axes tips (origin, X, Y, Z)
    """
    if R.ndim is 2:
        R = np.expand_dims(R, axis=0)
    
    if t is None:
        t = np.zeros((R.shape[0],3))
    elif t.ndim is 1:
        t = np.expand_dims(t, axis=0)
    
    T = np.concatenate((R,  np.expand_dims(t, axis=-1)), axis=-1)
    origin = np.matmul(T, np.expand_dims(np.repeat([[0,0,0,1]], T.shape[0], axis=0), axis=-1)).squeeze(axis=-1) 
    Xtip = np.matmul(T, np.expand_dims(np.repeat([[scale,0,0,1]], T.shape[0], axis=0), axis=-1)).squeeze(axis=-1)
    Ytip = np.matmul(T, np.expand_dims(np.repeat([[0,scale,0,1]], T.shape[0], axis=0), axis=-1)).squeeze(axis=-1)
    Ztip = np.matmul(T, np.expand_dims(np.repeat([[0,0,scale,1]], T.shape[0], axis=0), axis=-1)).squeeze(axis=-1)

    return np.stack([origin, Xtip, Ytip, Ztip], axis=1)

class Arrow3D(FancyArrowPatch):
    def __init__(self, verts3d, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = verts3d[:,0], verts3d[:,1], verts3d[:,2]

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_axes(ax, axes_tips, alpha=1.0, linewidth=1.0):
    arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', shrinkA=0, shrinkB=0, linewidth=linewidth, alpha=alpha)

    for axes_tip in axes_tips:
        a = Arrow3D(axes_tip[[0,1],:], **arrow_prop_dict, color='#fc8d62')
        ax.add_artist(a)
        a = Arrow3D(axes_tip[[0,2],:], **arrow_prop_dict, color='#66c2a5')
        ax.add_artist(a)
        a = Arrow3D(axes_tip[[0,3],:], **arrow_prop_dict, color='#8da0cb')
        ax.add_artist(a)
        
def load_dataset(dataset, label, seq_idx):
    csvfiles = sorted(glob.glob(f'{logdir}/{label}/{dataset}/**/*.csv', recursive=True))
    print(f"Loading sequences: {csvfiles}")
    num_seqs = len(csvfiles)

    print(f"Reading sequence: {csvfiles[seq_idx]}")

    dataframe = pd.read_csv(csvfiles[seq_idx])
    num_steps = dataframe.shape[0]

    step = num_steps-1
    graph_obj_trans = pandas_string_to_numpy(dataframe['graph/obj/trans'][step]).reshape((step+1,3))
    graph_obj_rot = pandas_string_to_numpy(dataframe['graph/obj/rot'][step]).reshape((step+1,3,3))

    gt_obj_trans = pandas_string_to_numpy(dataframe['gt/obj/trans'][step]).reshape((step+1,3))
    gt_obj_rot = pandas_string_to_numpy(dataframe['gt/obj/rot'][step]).reshape(step+1,3,3)
    
    return graph_obj_trans, graph_obj_rot, gt_obj_trans, gt_obj_rot

#%% CUBE 

dataset = 'tacto/cube/'
label = 'proposed'
seq_idx = 5

graph_obj_trans, graph_obj_rot, gt_obj_trans, gt_obj_rot = load_dataset(dataset, label, seq_idx)

num_frames = 10
FPS = 15.0
fig = plt.figure(figsize=(num_frames*1.2, 2))
idx = np.round(np.linspace(0, gt_obj_rot.shape[0] - 1, num_frames)).astype(int)
for i in range(num_frames):
    ax = fig.add_subplot(1, num_frames, i+1, projection='3d')
    ax.plot3D(gt_obj_trans[0:idx[i],0], gt_obj_trans[0:idx[i],1], gt_obj_trans[0:idx[i],2], 'gray',linewidth=4,alpha=1.0)
    ax.plot3D(graph_obj_trans[0:idx[i],0], graph_obj_trans[0:idx[i],1], graph_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=0.3)
    plot_axes(ax, transform_axes_tips(graph_obj_rot[idx[i],:,:], t=graph_obj_trans[idx[i],:], scale=1), alpha=0.3, linewidth=4.0)
    plot_axes(ax, transform_axes_tips(gt_obj_rot[idx[i],:,:], t=gt_obj_trans[idx[i],:], scale=1), alpha=1.0, linewidth=2.0)
    ax.set_title('t: {:.2f}s'.format(float(idx[i])/FPS), y = 0.0) # increase or decrease y as needed
    ax.title.set_size(8)
    ax.view_init(azim=37.5, elev=30)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    ax._axis3don = False
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

plt.subplots_adjust(wspace=-0.1, hspace=0)
plt.savefig(f'{logdir}/{label}/{dataset}/qual_traj_episode_{seq_idx:03d}.png')
plt.show()

#%% SPHERE 

# Sphere has mainly translation and no rotation tracking. 
# To show translation, one has to plot the traced trajectory and set scale accordingly.

dataset = 'tacto/sphere/'
label = 'proposed'
seq_idx = 0

graph_obj_trans, graph_obj_rot, gt_obj_trans, gt_obj_rot = load_dataset(dataset, label, seq_idx)

num_frames = 10
FPS = 15.0
fig = plt.figure(figsize=(num_frames*1.2, 2))
idx = np.round(np.linspace(0, gt_obj_rot.shape[0] - 1, num_frames)).astype(int)
for i in range(num_frames):
    ax = fig.add_subplot(1, num_frames, i+1, projection='3d')
    ax.plot3D(gt_obj_trans[0:idx[i],0], gt_obj_trans[0:idx[i],1], gt_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=1.0)
    ax.plot3D(graph_obj_trans[0:idx[i],0], graph_obj_trans[0:idx[i],1], graph_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=0.3)
    plot_axes(ax, transform_axes_tips(graph_obj_rot[idx[i],:,:], t=graph_obj_trans[idx[i],:], scale=0.01), alpha=0.3, linewidth=1.5)
    plot_axes(ax, transform_axes_tips(gt_obj_rot[idx[i],:,:], t=gt_obj_trans[idx[i],:], scale=0.01), alpha=1.0, linewidth=1.5)
    ax.set_title('t: {:.2f}s'.format(float(idx[i])/FPS), y = 0.0) # increase or decrease y as needed
    ax.title.set_size(8)
    ax.view_init(azim=37.5, elev=30)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlim(-0.025, -0.007)
    ax.set_ylim(-0.009, 0.009)
    ax.set_zlim(0.026, 0.044)
    ax._axis3don = False
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

plt.subplots_adjust(wspace=0.1, hspace=0)
plt.savefig(f'{logdir}/{label}/{dataset}/qual_traj_episode_{seq_idx:03d}.png')
plt.show()

#%% TOY BRICK 

dataset = 'tacto/toy_brick/'
label = 'proposed'
seq_idx = 0

graph_obj_trans, graph_obj_rot, gt_obj_trans, gt_obj_rot = load_dataset(dataset, label, seq_idx)

num_frames = 10
FPS = 15.0
fig = plt.figure(figsize=(num_frames*1.2, 2))
idx = np.round(np.linspace(0, gt_obj_rot.shape[0] - 1, num_frames)).astype(int)
for i in range(num_frames):
    ax = fig.add_subplot(1, num_frames, i+1, projection='3d')
    ax.plot3D(gt_obj_trans[0:idx[i],0], gt_obj_trans[0:idx[i],1], gt_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=1.0)
    ax.plot3D(graph_obj_trans[0:idx[i],0], graph_obj_trans[0:idx[i],1], graph_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=0.3)
    plot_axes(ax, transform_axes_tips(graph_obj_rot[idx[i],:,:], t=graph_obj_trans[idx[i],:], scale=1), alpha=0.3, linewidth=4.0)
    plot_axes(ax, transform_axes_tips(gt_obj_rot[idx[i],:,:], t=gt_obj_trans[idx[i],:], scale=1), alpha=1.0, linewidth=2.0)
    ax.set_title('t: {:.2f}s'.format(float(idx[i])/FPS), y = 0.0) # increase or decrease y as needed
    ax.title.set_size(8)
    ax.view_init(azim=37.5, elev=30)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    ax._axis3don = False
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

plt.subplots_adjust(wspace=-0.1, hspace=0)
plt.savefig(f'{logdir}/{label}/{dataset}/qual_traj_episode_{seq_idx:03d}.png')
plt.show()

#%% TOY HUMAN

dataset = 'tacto/toy_human/'
label = 'proposed'
seq_idx = 0

graph_obj_trans, graph_obj_rot, gt_obj_trans, gt_obj_rot = load_dataset(dataset, label, seq_idx)

num_frames = 9
FPS = 15.0
fig = plt.figure(figsize=(num_frames*1.2, 2))
idx = np.round(np.linspace(0, gt_obj_rot.shape[0] - 1, num_frames)).astype(int)
for i in range(num_frames):
    ax = fig.add_subplot(1, num_frames, i+1, projection='3d')
    ax.plot3D(gt_obj_trans[0:idx[i],0], gt_obj_trans[0:idx[i],1], gt_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=1.0)
    ax.plot3D(graph_obj_trans[0:idx[i],0], graph_obj_trans[0:idx[i],1], graph_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=0.3)
    plot_axes(ax, transform_axes_tips(graph_obj_rot[idx[i],:,:], t=graph_obj_trans[idx[i],:], scale=1), alpha=0.3, linewidth=4.0)
    plot_axes(ax, transform_axes_tips(gt_obj_rot[idx[i],:,:], t=gt_obj_trans[idx[i],:], scale=1), alpha=1.0, linewidth=2.0)
    ax.set_title('t: {:.2f}s'.format(float(idx[i])/FPS), y = 0.0) # increase or decrease y as needed
    ax.title.set_size(8)
    ax.view_init(azim=37.5, elev=30)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.7, 0.7)
    ax._axis3don = False
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

plt.subplots_adjust(wspace=-0.1, hspace=0)
plt.savefig(f'{logdir}/{label}/{dataset}/qual_traj_episode_{seq_idx:03d}.png')
plt.show()

#%% REAL SPHERE 

# Sphere has mainly translation and no rotation tracking. 
# To show translation, one has to plot the traced trajectory and set scale accordingly.

dataset = 'real/sphere/'
label = 'proposed'
seq_idx=5

graph_obj_trans, graph_obj_rot, gt_obj_trans, gt_obj_rot = load_dataset(dataset, label, seq_idx)

num_frames = 10
FPS = 15.0
fig = plt.figure(figsize=(num_frames*1.2, 2))
max_frame = 100 # gt_obj_rot.shape[0] - 1
idx = np.round(np.linspace(0, max_frame, num_frames)).astype(int)
for i in range(num_frames):
    ax = fig.add_subplot(1, num_frames, i+1, projection='3d')
    ax.plot3D(gt_obj_trans[0:idx[i],0], gt_obj_trans[0:idx[i],1], gt_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=1.0)
    ax.plot3D(graph_obj_trans[0:idx[i],0], graph_obj_trans[0:idx[i],1], graph_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=0.4)

    # ARROW_SCALE = 0.1
    # plot_axes(ax, transform_axes_tips(graph_obj_rot[idx[i],:,:], t=graph_obj_trans[idx[i],:], scale=ARROW_SCALE), alpha=0.5, linewidth=1.5)
    # plot_axes(ax, transform_axes_tips(gt_obj_rot[idx[i],:,:], t=gt_obj_trans[idx[i],:], scale=ARROW_SCALE), alpha=1.0, linewidth=1.5)


    ax.set_title('t: {:.2f}s'.format(float(idx[i])/FPS), y = 0.0) # increase or decrease y as needed
    ax.title.set_size(8)
    ax.view_init(azim=37.5, elev=30)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # ax.set_xlim(-0.07, 0.07)
    # ax.set_ylim(-0.07, 0.07)
    # ax.set_zlim(-0.07, 0.07)

    # print(ax.get_xlim())
    # print(ax.get_ylim())
    # print(ax.get_zlim())

    ax._axis3don = False
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

plt.subplots_adjust(wspace=0.1, hspace=0)
filename = f'{logdir}/{label}/{dataset}/qual_traj_episode_{seq_idx:03d}.png'
plt.savefig(filename)
print(f"Saved figure to: {filename}")
plt.show()

#%% REAL CUBE 

dataset = 'real/cube/'
label = 'proposed'
seq_idx = 8

graph_obj_trans, graph_obj_rot, gt_obj_trans, gt_obj_rot = load_dataset(dataset, label, seq_idx)

num_frames = 10
FPS = 15.0
fig = plt.figure(figsize=(num_frames*1.2, 2))
max_frame = 100 # gt_obj_rot.shape[0] - 1
idx = np.round(np.linspace(0, max_frame, num_frames)).astype(int)
for i in range(num_frames):
    ax = fig.add_subplot(1, num_frames, i+1, projection='3d')
    ax.plot3D(gt_obj_trans[0:idx[i],0], gt_obj_trans[0:idx[i],1], gt_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=1.0)
    ax.plot3D(graph_obj_trans[0:idx[i],0], graph_obj_trans[0:idx[i],1], graph_obj_trans[0:idx[i],2], 'gray',linewidth=2,alpha=0.3)
    # plot_axes(ax, transform_axes_tips(graph_obj_rot[idx[i],:,:], t=graph_obj_trans[idx[i],:], scale=1), alpha=0.4, linewidth=4.0)
    # plot_axes(ax, transform_axes_tips(gt_obj_rot[idx[i],:,:], t=gt_obj_trans[idx[i],:], scale=1), alpha=1.0, linewidth=2.0)
    ax.set_title('t: {:.2f}s'.format(float(idx[i])/FPS), y = 0.0) # increase or decrease y as needed
    ax.title.set_size(8)
    ax.view_init(azim=37.5, elev=30)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(-0.5, 0.5)
    ax._axis3don = False
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

plt.subplots_adjust(wspace=-0.1, hspace=0)
plt.savefig(f'{logdir}/{label}/{dataset}/qual_traj_episode_{seq_idx:03d}.png')
plt.show()

# %%

# Copyright (c) Facebook, Inc. and its affiliates.

#%%

import numpy as np
import math
import os

import glob
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from inhandpy.utils import data_utils

plt.rcParams.update({'font.size': 36})
plt.ion()

#%% Read dataframe from csvfile

BASE_PATH = '/home/paloma/code/fair_ws/tactile-in-hand/inhandpy'
logdir = f'{BASE_PATH}/local/datalogs'

## Sim tracker results ##
dataset = 'tacto/toy_human' # 'cube', 'toy_brick', 'toy_human', 'sphere'
labels = ['const_vel', 'scan2scan', 'proposed', 'global']

## Real tracker results ##
# dataset = 'real/sphere' # 'sphere', 'cube'
# labels = ['const_vel', 'scan2scan', 'proposed']

colors = [[0.55, 0.63, 0.80], [0.89, 0.47, 0.76], [0.99, 0.55, 0.38], [0.4, 0.76, 0.65]]
# matplotlib.colors.to_rgba('tab:pink', alpha=None)

#%% Compute translational, rotational errors

max_steps = 200
num_seqs = 10
num_labels = len(labels)

error_mat_trans = np.zeros((num_seqs, num_labels))
error_mat_rot = np.zeros((num_seqs, num_labels))

for label_idx, label in enumerate(labels):

    csvfiles = sorted(glob.glob(f'{logdir}/{label}/{dataset}/**/*.csv', recursive=True))
    
    for seq_idx in range(0, num_seqs):

        dataframe = pd.read_csv(csvfiles[seq_idx])
        num_steps = np.minimum(dataframe.shape[0], max_steps)

        error_vec_trans = np.zeros((max_steps))
        error_vec_rot = np.zeros((max_steps))

        print(f"Processing {num_steps} step sequence {csvfiles[seq_idx]} ...")

        for step in range(0, num_steps):
            
            graph_obj_trans = data_utils.pandas_string_to_numpy(dataframe['graph/obj/trans'][step]).reshape((step+1,3))
            graph_obj_rot = (data_utils.pandas_string_to_numpy(dataframe['graph/obj/rot'][step])).reshape((step+1,3,3))

            gt_obj_trans = data_utils.pandas_string_to_numpy(dataframe['gt/obj/trans'][step]).reshape((step+1,3))
            gt_obj_rot = (data_utils.pandas_string_to_numpy(dataframe['gt/obj/rot'][step])).reshape(step+1,3,3)

            error_vec_trans[step] = data_utils.traj_error_trans(graph_obj_trans, gt_obj_trans)
            error_vec_rot[step] = data_utils.traj_error_rot(graph_obj_rot, gt_obj_rot)
        
        error_vec_trans[num_steps:max_steps] = error_vec_trans[num_steps-1]
        error_vec_rot[num_steps:max_steps] = error_vec_rot[num_steps-1]

        error_mat_trans[seq_idx, label_idx] = np.mean(error_vec_trans)
        error_mat_rot[seq_idx, label_idx] = np.mean(error_vec_rot)
    
# convert units
error_mat_trans = 1e3 * error_mat_trans
# error_mat_rot = 180/math.pi * error_mat_rot

# %% Box plots for rotational, translational errors

fig1, axs1 = plt.subplots(nrows=1, ncols=2, num=1, clear=True, figsize=(24, 8))

line_props = dict(color="black", alpha=1., linewidth=2)
kwargs = {'vert': True, 'notch': False, 'patch_artist': True,
            'medianprops': line_props, 'whiskerprops': line_props}

bplot1 = axs1[0].boxplot(error_mat_rot, widths=0.8, labels=None, **kwargs)

for patch, color in zip(bplot1['boxes'], colors):
    color = np.array([color[0], color[1], color[2], 0.8])
    patch.set_facecolor(color)
    patch.set_linewidth(2)

axs1[0].set_xticks([])
# axs1[0].set_ylim((0., 1.))
axs1[0].set_yscale('log')
axs1[0].set_ylim((10**-2, 2 *10**0))
# axs1[0].set_title("RMSE rotational error (radians)")

bplot2 = axs1[1].boxplot(error_mat_trans, widths=0.8, labels=None, **kwargs)

for patch, color in zip(bplot2['boxes'], colors):
    color = np.array([color[0], color[1], color[2], 0.8])
    patch.set_facecolor(color)
    patch.set_linewidth(2)

axs1[1].set_xticks([])
# axs1[1].set_ylim((0., 5.))
axs1[1].set_yscale('log')
axs1[1].set_ylim((10**-1, 10**1))
# axs1[1].set_title("RMSE translational error")

plotdir = f'{logdir}/plots/{dataset}'
os.makedirs(plotdir, exist_ok=True)

plt.savefig(f'{plotdir}/quant_errors_boxplot.png')
print(f'Saved figure to {plotdir}/quant_errors_boxplot.png')

# %%

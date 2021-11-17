# Copyright (c) Facebook, Inc. and its affiliates.

#%%

import pandas as pd
import glob
import os

import matplotlib.pyplot as plt
from inhandpy.utils import data_utils

plt.rcParams.update({'font.size': 32})
plt.ion()

#%%

BASE_PATH = '/home/paloma/code/fair_ws/tactile-in-hand/inhandpy/'
dirname = f'{BASE_PATH}/local/datalogs/20210831/tacto/die/dataset_0002/'
csvfiles = sorted(glob.glob(f'{dirname}/*.csv'))
csvfile = csvfiles[-1]

print(f"Reading {csvfile}")
dataframe = pd.read_csv(csvfile)

#%% Pairwise estimates

reg_type = 's2s' # s2s, s2m
reg_gt_rot = data_utils.pandas_col_to_numpy(dataframe[f'reg/{reg_type}/gt/rot'].dropna())
reg_gt_trans = data_utils.pandas_col_to_numpy(dataframe[f'reg/{reg_type}/gt/trans'].dropna())
reg_icp_rot = data_utils.pandas_col_to_numpy(dataframe[f'reg/{reg_type}/rot'].dropna())
reg_icp_trans = data_utils.pandas_col_to_numpy(dataframe[f'reg/{reg_type}/trans'].dropna())

fig1, axs1 = plt.subplots(nrows=1, ncols=3, num=1, clear=True, figsize=(42, 8))

ylabels = ['rot_x', 'rot_y', 'rot_z']
for dim in range(0, 3):
    axs1[dim].plot(reg_icp_rot[:, dim], color='tab:orange', linewidth=3, label='registration')
    axs1[dim].plot(reg_gt_rot[:, dim], color='tab:green', linewidth=3, label='groundtruth')
    
    axs1[dim].set_xlabel('step')
    axs1[dim].set_ylabel(ylabels[dim])
    axs1[dim].set_title('Pairwise values')

    axs1[dim].set_ylim([-0.2, 0.2])

    axs1[dim].legend()


plt.savefig(f"{csvfile.split('.')[0]}_registration.png")

#%% Integrated estimates

graphopt_gt_rot = data_utils.pandas_col_to_numpy(dataframe['graphopt/gt/rot'].dropna())
graphopt_gt_trans = data_utils.pandas_col_to_numpy(dataframe['graphopt/gt/trans'].dropna())
graphopt_est_rot = data_utils.pandas_col_to_numpy(dataframe['graphopt/est/rot'].dropna())
graphopt_est_trans = data_utils.pandas_col_to_numpy(dataframe['graphopt/est/trans'].dropna())

fig2, axs2 = plt.subplots(nrows=1, ncols=3, num=1, clear=True, figsize=(42, 8))

ylabels = ['rot_x', 'rot_y', 'rot_z']
for dim in range(0, 3):
    axs2[dim].plot(graphopt_est_rot[:, dim], color='tab:orange', linewidth=4, label='graphopt')
    axs2[dim].plot(graphopt_gt_rot[:, dim], color='tab:green', linewidth=4, label='groundtruth')
    
    axs2[dim].set_xlabel('step')
    axs2[dim].set_ylabel(ylabels[dim])
    axs2[dim].set_title('Integrated values')

    axs2[dim].set_ylim([-2., 2.])

    axs2[dim].legend()

plt.savefig(f"{csvfile.split('.')[0]}_graphopt.png")

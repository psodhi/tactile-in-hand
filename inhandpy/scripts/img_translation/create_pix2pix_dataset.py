# Copyright (c) Facebook, Inc. and its affiliates.

import os
import glob
import pandas as pd

import numpy as np
from PIL import Image
import imageio

import logging
log = logging.getLogger(__name__)

def load_seq_from_file(csvfile):

    seq_data = pd.read_csv(csvfile) if os.path.exists(csvfile) else None

    if seq_data is not None:
        seq_data = None if (len(seq_data) < 2) else seq_data

    return seq_data

def main():

    base_path = '/home/paloma/code/fair_ws/tactile-in-hand/inhandpy/'

    # update src + dst dirs
    # src_dataset_dir = f'{base_path}/local/datasets/tacto/sphere/dataset_0000/'
    # src_dataset_dir = f'{base_path}/local/datasets/real/sphere/digit-0.5mm-ball-bearing-zup_2021-08-31-21-37-35/'
    src_dataset_dir = f'{base_path}/local/datasets/real/cube/digit-flatcorner_2021-09-10-21-13-17/'

    dst_dataset_dir = f'{base_path}/local/datasets/pix2pix/real_pyramid'
    
    dataset_types = ['train', 'test']

    dir_A = f"{dst_dataset_dir}/A/"
    dir_B = f"{dst_dataset_dir}/B/"
    dir_AB = f"{dst_dataset_dir}/AB/"

    # iterate over train/test data
    for dataset_type in dataset_types:

        os.makedirs(f"{dir_A}/{dataset_type}", exist_ok=True)
        os.makedirs(f"{dir_B}/{dataset_type}", exist_ok=True)
        os.makedirs(f"{dir_AB}/{dataset_type}", exist_ok=True)

        csvfiles = sorted(glob.glob(f"{src_dataset_dir}/{dataset_type}/**/*.csv"))
        dst_img_idx = 0

        # iterate over contact sequences
        for eps_idx in range(9, len(csvfiles)):

            csvfile = csvfiles[eps_idx]
            seq_data = load_seq_from_file(csvfile)

            if seq_data is None:
                continue

            print(f"Reading contact sequence from {csvfile}")

            img_color_locs, img_normal_locs = None, None

            if 'img_bot_color_loc' in seq_data: img_color_locs = seq_data[f"img_bot_color_loc"]
            if 'img_color_loc' in seq_data: img_color_locs = seq_data[f"img_color_loc"]
            if 'img_bot_normal_loc' in seq_data: img_normal_locs = seq_data[f"img_bot_normal_loc"]
            # if 'img_normal_loc' in seq_data: img_normal_locs = seq_data[f"img_normal_loc"]

            # iterate over images within each sequence
            for img_idx in range(0, len(img_color_locs)):

                img_color = Image.open(f"{src_dataset_dir}/{dataset_type}/{img_color_locs[img_idx]}")
                
                if img_normal_locs is not None: 
                    img_normal = Image.open(f"{src_dataset_dir}/{dataset_type}/{img_normal_locs[img_idx]}")
                else:
                    img_normal = img_color
                    img_normal = np.zeros_like(img_color)
                    img_normal[:, :, 2] = 255

                imageio.imwrite(f"{dir_A}/{dataset_type}/{dst_img_idx:04d}.png", img_color)
                imageio.imwrite(f"{dir_B}/{dataset_type}/{dst_img_idx:04d}.png", img_normal)

                dst_img_idx = dst_img_idx + 1
            
            break # single episode
            
    os.system(f"python scripts/img_translation/combine_A_and_B.py --fold_A {dir_A} --fold_B {dir_B} --fold_AB {dir_AB}")

    log.info(f"Created tactile dataset of {dst_img_idx} images at {dst_dataset_dir}.")

if __name__=='__main__':
    main()
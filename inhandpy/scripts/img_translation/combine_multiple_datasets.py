# Copyright (c) Facebook, Inc. and its affiliates.

import os
import glob
import pandas as pd

import numpy as np
from PIL import Image
import random

import logging

log = logging.getLogger(__name__)

def main():

    base_path= '/home/paloma/code/fair_ws/tactile-in-hand/inhandpy/local/datasets/pix2pix'

    # dataset_names = ['sim_sphere', 'sim_cube', 'sim_toy_human', 'sim_toy_brick']
    dataset_names = ['sim_sphere', 'sim_cube', 'sim_toy_human_2', 'sim_toy_brick', 'real_sphere', 'real_cube']
    comb_dataset_name = '_'.join(dataset_names)
    # max_files = [75, 75, 125]
    max_files = len(dataset_names) * [10]

    dataset_types = ['train', 'test']

    for dataset_type in dataset_types:

        combdir = f"{base_path}/{comb_dataset_name}/AB/{dataset_type}/"
        os.makedirs(combdir, exist_ok=True)

        for didx, dataset_name in enumerate(dataset_names):

            imgfiles = (glob.glob(f"{base_path}/{dataset_name}/AB/{dataset_type}/**/*.png", recursive=True))
            random.shuffle(imgfiles)

            n_files = np.minimum(max_files[didx], len(imgfiles))

            for fidx in range(0, n_files):

                imgfile = imgfiles[fidx]

                img_name_old = imgfile.split('/')[-1]
                img_name_new = f"{dataset_name}_{img_name_old}"

                print(f"cp {imgfile} {base_path}/{comb_dataset_name}/AB/{dataset_type}/{img_name_new}")

                os.system(f"cp {imgfile} {combdir}/{img_name_new}")

if __name__=='__main__':
    main()
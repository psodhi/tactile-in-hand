# Copyright (c) Facebook, Inc. and its affiliates.

import os
import numpy as np
import imageio

import torch
import torchvision.transforms as transforms

from inhandpy.thirdparty.pix2pix.options.test_options import TestOptions
from inhandpy.thirdparty.pix2pix.data import create_dataset
from inhandpy.thirdparty.pix2pix.models import create_model
from inhandpy.thirdparty.pix2pix.util.visualizer import save_images
from inhandpy.thirdparty.pix2pix.util import html

from inhandpy.utils import data_utils

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

loss_fn = torch.nn.MSELoss()

def compute_loss(actual, expected):
    error = loss_fn(actual, expected)
    return error

def update_img_dict(img_dict, visuals):
    
    visuals_keys = ['real_A', 'real_B', 'fake_B']

    idx = 0
    for key in img_dict:
        vkey = visuals_keys[idx]
        img = ((visuals[vkey]).squeeze(0) + 1) / 2.0
        img = data_utils.interpolate_img(img=img, rows=160, cols=120)
        img_dict[key].append(transforms.ToPILImage()(img))
        idx = idx + 1

    return img_dict

def save_video(img_dict, dstdir, dataset_name, fps=15):

    print("Writing img dict outputs as videos to: {0}".format(dstdir))

    for key in img_dict:
        vidfile = f"{dstdir}/{dataset_name}_{key}.mp4"
        imageio.mimwrite(vidfile, img_dict[key], fps=fps)

if __name__ == '__main__':
    
    opt = TestOptions().parse()  # get test options

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    n_data = len(dataset) # np.minimum(100, len(dataset))
    loss_vec = torch.zeros(n_data)

    img_dict = {}
    img_dict['color'], img_dict['normal_gt'], img_dict['normal_pred'] = [], [], []

    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results

        loss_vec[i] = compute_loss(actual=visuals['fake_B'], expected=visuals['real_B'])
        img_dict = update_img_dict(img_dict, visuals)
        
        # save images to an HTML file
        img_path = model.get_image_paths()
        if i % 2 == 0: 
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    # save img lists as video to file
    dstdir_vid=f"{BASE_PATH}/local/pix2pix/qual_videos"
    os.makedirs(dstdir_vid, exist_ok=True)
    dataset_name = opt.dataroot.split('/')[-2]
    save_video(img_dict, dstdir=dstdir_vid, dataset_name=dataset_name)

    webpage.save()  # save the HTML

    print(f"test mse reconstruction loss: {torch.mean(loss_vec)}")

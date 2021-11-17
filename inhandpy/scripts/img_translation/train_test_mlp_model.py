# Copyright (c) Facebook, Inc. and its affiliates.

import os
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from inhandpy.thirdparty.pix2pix.options.test_options import TestOptions
from inhandpy.thirdparty.pix2pix.options.train_options import TrainOptions
from inhandpy.thirdparty.pix2pix.data import create_dataset
from inhandpy.utils import vis_utils, data_utils

import matplotlib.pyplot as plt
plt.ion()

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class MLPNetwork(nn.Module):
    def __init__(
            self, input_size=5, output_size=3, hidden_size=32):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def visualize_vector(img_vec, title=''):

    img_mat = img_vec.permute(1, 0) # n x 3 -> 3 x n
    img_mat = img_mat.reshape(3, 256, 256) # 3 x n -> 3 x H x W
    plt.imshow(img_mat.permute(1,2,0).cpu().detach().numpy())
    plt.title(title)

    plt.show()
    plt.pause(1e-1)

def vector_to_img(img_vec, img_numpy=False):

    img_mat = img_vec.permute(1, 0) # n x 3 -> 3 x n
    img_mat = img_mat.reshape(3, 256, 256) # 3 x n -> 3 x H x W

    if img_numpy: img_mat = img_mat.permute(1,2,0).cpu().detach().numpy()

    return img_mat

def update_img_dict(img_dict, inputs, outputs, labels):
    
    img_color = (vector_to_img(img_vec=inputs[:, 0:3]) + 1) / 2.0
    img_normal_gt = (vector_to_img(img_vec=labels) + 1) / 2.0
    img_normal_pred = (vector_to_img(img_vec=outputs) + 1) / 2.0

    img_list = [img_color, img_normal_gt, img_normal_pred]

    idx = 0
    for key in img_dict:
        img = img_list[idx]
        img = data_utils.interpolate_img(img=img, rows=160, cols=120)
        img_dict[key].append(transforms.ToPILImage()(img))
        idx = idx + 1

    return img_dict

def save_video(img_dict, dstdir, dataset_name, fps=15):

    print("Writing img dict outputs as videos to: {0}".format(dstdir))

    for key in img_dict:
        vidfile = f"{dstdir}/{dataset_name}_{key}.mp4"
        imageio.mimwrite(vidfile, img_dict[key], fps=fps)

def preproc_data(data):

    # read data
    color = data['A'].squeeze(0)
    normal = data['B'].squeeze(0)

    # compute network input
    ch, rows, cols = color.shape
    row_loc_mat = (torch.arange(0, cols)).repeat(rows, 1) / cols
    col_loc_mat = (torch.arange(0, rows).unsqueeze(-1)).repeat(1, cols) / rows

    inputs = torch.zeros((5, rows, cols))
    inputs[0:3, :] = color
    inputs[3, :] = row_loc_mat
    inputs[4, :] = col_loc_mat

    # reshape input / labels
    inputs = inputs.reshape((5, -1))
    labels = normal.reshape((3, -1))

    inputs = inputs.permute(1,0) # 5 x n -> n x 5
    labels = labels.permute(1,0) # 3 x n -> n x 3

    return inputs, labels

def train_model(dataset, n_epochs=500, opt=None):

    model = MLPNetwork()
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):

        loss_vec = torch.zeros(len(dataset))
        
        for i, data in enumerate(dataset):

            inputs, labels = preproc_data(data)

            # clear gradients
            optimizer.zero_grad()
            
            # forward, backward, optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_vec[i] = loss.item()
            
        print(f'epoch: {epoch}, loss: {torch.mean(loss_vec)}')

        # save model
        torch.save(model.state_dict(), "{BASE_PATH}/local/pix2pix/checkpoints/mlp")

def test_model(dataset, opt=None):

    model = MLPNetwork()
    criterion = torch.nn.MSELoss()

    model.load_state_dict(torch.load("{BASE_PATH}/local/pix2pix/checkpoints/mlp"))
    model.eval()

    loss_vec = torch.zeros(len(dataset))

    img_dict = {}
    img_dict['color'], img_dict['normal_gt'], img_dict['normal_pred'] = [], [], []

    for i, data in enumerate(dataset):
        inputs, labels = preproc_data(data) # n x 5, n x 3
        outputs = model.forward(inputs) # n x 3

        ## collect imgs to be saved to file as a video
        img_dict = update_img_dict(img_dict, inputs, outputs, labels)

        ## visualize predictions
        # fig1, axs1 = plt.subplots(nrows=1, ncols=3, num=1, clear=True, figsize=(22, 8))
        # img_color_np = (vector_to_img(img_vec=inputs[:, 0:3]) + 1) / 2.0
        # img_normal_gt_np = (vector_to_img(img_vec=labels) + 1) / 2.0
        # img_normal_pred_np = (vector_to_img(img_vec=outputs) + 1) / 2.0
        # vis_utils.visualize_imgs(fig=fig1, axs=[axs1[0], axs1[1], axs1[2]],
        #                 img_list=[img_color_np, img_normal_gt_np, img_normal_pred_np], 
        #                 titles=['img_color', 'img_normal_gt', 'img_normal_pred'], cmap='coolwarm')
        # plt.pause(1e-3)

        loss_vec[i] = criterion(outputs, labels)
    
    # save img lists as video to file
    dstdir_vid=f"{BASE_PATH}/local/pix2pix/qual_videos/mlp"
    os.makedirs(dstdir_vid, exist_ok=True)
    dataset_name = opt.dataroot.split('/')[-2]
    save_video(img_dict, dstdir=dstdir_vid, dataset_name=dataset_name)

    print(f"test mse reconstruction loss: {torch.mean(loss_vec)}")

def main():

    mode = 'test' # 'train', 'test'

    if (mode == 'train'): opt = TrainOptions().parse()  
    if (mode == 'test'): opt = TestOptions().parse() 

    opt.num_threads = 0  
    opt.batch_size = 1   
    opt.serial_batches = True 
    opt.no_flip = True   
    opt.display_id = -1  
    dataset = create_dataset(opt) 

    if (mode == 'train'): train_model(dataset, opt=opt)
    if (mode == 'test'): test_model(dataset, opt=opt)

if __name__ == '__main__':
    main()
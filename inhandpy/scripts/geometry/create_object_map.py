# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import os

import hydra
import logging
from attrdict import AttrDict
import copy

import open3d as o3d
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

from inhandpy.dataio.sim_digit_data_loaders import DigitTactoImageTfSeqDataset
from inhandpy.utils import geom_utils, vis_utils

log = logging.getLogger(__name__)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONFIG_PATH = os.path.join(BASE_PATH, "config/depth_normals_cloud.yaml")  

def data_loader(dataset_names, datatype, params):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_list = []
    for (ds_idx, dataset_name) in enumerate(dataset_names):
        srcdir_dataset = f"{BASE_PATH}/local/datasets/{dataset_name}"
        dataset_list.append(DigitTactoImageTfSeqDataset(
            dir_dataset=f"{srcdir_dataset}/{datatype}", dataset_name=dataset_name, transform=transform,
            downsample_imgs=params.downsample_imgs, img_types=params.img_types, digit_id=params.digit_id))

    dataset = ConcatDataset(dataset_list)
    dataloader = DataLoader(dataset, batch_size=params.batch_size,
                            shuffle=params.shuffle, num_workers=params.num_workers)

    return dataloader, dataset

def save_map_to_file(cloud_map, ds_name, map_type='voxel', voxel_size=1e-5):

    dstdir = f"{BASE_PATH}/local/maps/{ds_name}/"
    os.makedirs(dstdir, exist_ok=True)

    dstfile = f"{dstdir}/prior_map.ply"
    map = cloud_map

    if (map_type == 'cloud'):
        o3d.io.write_point_cloud(dstfile, map)
    elif (map_type == 'voxel'):
        map = o3d.geometry.VoxelGrid.create_from_point_cloud(cloud_map, voxel_size=voxel_size)
        num_voxels = len(o3d.geometry.VoxelGrid.get_voxels(map))
        o3d.io.write_voxel_grid(dstfile, map)
        # print(f"[save_map_to_file] Voxel map containts {num_voxels} voxels of size {voxel_size}")
    else:
        log.info(f"[save_map_to_file] map_type: {map_type} not found")

    print(f"[save_map_to_file] Saved {map_type} map to {dstfile}")

    return map

@hydra.main(config_path=CONFIG_PATH)
def main(cfg):

    # data loader
    train_dataloader, train_dataset = data_loader(
        dataset_names=cfg.dataset_names, datatype="train", params=cfg.dataloader)

    dataset_uids = [item.get_uids() for item in train_dataset.datasets]
    dataset_uids = [item for sublist in dataset_uids for item in sublist]

    # init values
    view_params = AttrDict({'fov': 60, 'front': [-0.56, 0.81, 0.14], 'lookat': [
                           -0.006, -0.0117, 0.033], 'up': [0.0816, -0.112, 0.990], 'zoom': 2.0})
    # view_params = AttrDict({'fov': 60, 'front': [0.2, 0.95, 0.14], 'lookat': [
    #                        -0.00, -0.0, 0.025], 'up': [0.0, 0.0, -0.4], 'zoom': 1.0})
    vis3d = vis_utils.Visualizer3d(base_path=BASE_PATH, view_params=view_params)

    T_cam_offset = torch.tensor(cfg.sensor.T_cam_offset)
    proj_mat = torch.tensor(cfg.sensor.P)
    gel_depth = torch.tensor((np.loadtxt(
        f"{BASE_PATH}/{cfg.sensor.gel_depth_map_loc}", delimiter=',') - cfg.sensor.gel_depth_offset).astype(np.float32))

    # initialize open3d geometries
    colors = [[252/255., 141/255., 98/255.], [102/255., 194/255., 165/255.], [141/255., 160/255., 203/255.], [0.3, 0.3, 0.3]] # Or, Gr, Pu
    frames = vis3d.init_geometry(geom_type="frame", num_items=1)
    meshes = vis3d.init_geometry(geom_type="mesh", num_items=1, file_names=[
                                 cfg.object.mesh_name], colors=[colors[0]], wireframes=[True])
    meshes[0].scale(cfg.object.global_scaling * cfg.object.mesh_scaling, center=(0, 0, 0))

    # iterate over dataset
    for ds_idx in tqdm(range(0, len(train_dataset))):

        log.info(f"[run_tracker] Running dataset {dataset_uids[ds_idx]}")

        dataset = train_dataset[ds_idx]

        # load imgs: S x C x H x W , poses: (S x 3 x 3, S x 3) <-> (R, t)
        img_seq_color, img_seq_depth, img_seq_normal, pose_seq_obj, pose_seq_digit_top, pose_seq_digit_bot = dataset
        pose_seq_sensor = pose_seq_digit_bot if (cfg.dataloader.digit_id == "bot") else pose_seq_digit_top

        S, C, H, W = img_seq_color.shape

        # initialize open3d geometries
        clouds = vis3d.init_geometry(geom_type="cloud", num_items=1)

        start_step, nsteps = 5, S
        points3d_map = None

        for step in tqdm(range(start_step, nsteps)):

            # get current images
            img_normal = copy.deepcopy(img_seq_normal[step, :, :, :])
            img_depth_gt = copy.deepcopy(img_seq_depth[step, :, :]).squeeze(0)
            bg_mask = (img_depth_gt > gel_depth).squeeze()

            # convert to depth
            use_gt_depth = False
            if not use_gt_depth: # normal to depth
                boundary = torch.zeros((img_normal.shape[-2], img_normal.shape[-1]))
                img_depth = geom_utils.normal_to_depth(img_normal, bg_mask=bg_mask, boundary=boundary,
                                                       gel_width=cfg.sensor.gel_width, gel_height=cfg.sensor.gel_height)
                img_depth = geom_utils.mask_background(img_depth, bg_mask=bg_mask, bg_val=0.)
            else:
                img_depth = geom_utils.mask_background(img_depth_gt, bg_mask=bg_mask, bg_val=0.)

            img_depth = geom_utils.flip(img_depth)

            # get current transforms
            T_obj_gt = geom_utils.Rt_to_T(R=pose_seq_obj[0][step, :], t=pose_seq_obj[1][step, :])
            T_sensor = geom_utils.Rt_to_T(R=pose_seq_sensor[0][step, :], t=pose_seq_sensor[1][step, :])
            T_camera = torch.matmul(T_sensor, T_cam_offset)

            # inverse projection
            points3d = geom_utils.depth_to_pts3d(depth=img_depth, P=proj_mat, V=torch.inverse(T_camera), 
                                                       params=cfg.sensor, ordered_pts=False)
            
            # transform to object frame, remove outliers
            points3d = geom_utils.transform_pts3d(torch.inverse(T_obj_gt), points3d)
            points3d = geom_utils.remove_outlier_pts(points3d, nb_neighbors=20, std_ratio=10.)

            # concatenate pts into a single global cloud
            points3d_map = np.concatenate((points3d_map, points3d), axis=1) if (points3d_map is not None) else points3d
            clouds = geom_utils.init_points_to_clouds(clouds=clouds, points3d=[points3d_map])

            if False:
                vis_utils.visualize_geometries_o3d(vis3d=vis3d, clouds=clouds, frames=frames,
                                                meshes=meshes, transforms=[T_obj_gt])
        
        clouds = geom_utils.init_points_to_clouds(clouds=clouds, points3d=[points3d_map])
        object_map = save_map_to_file(cloud_map=clouds[0], ds_name=dataset_uids[ds_idx],
                                      map_type='voxel', voxel_size=1e-5)

        if False:
            o3d.visualization.draw_geometries([object_map]) # blocking vis
            # vis_utils.visualize_geometries_o3d(vis3d=vis3d, clouds=[object_map])        

        vis3d.clear_geometries()

    vis3d.destroy()

if __name__ == '__main__':
    main()

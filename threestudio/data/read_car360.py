import copy
import os,math
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import NamedTuple
from threestudio.data.utils import qvec2rotmat,  read_extrinsics_binary, read_intrinsics_binary

import numpy as np
import json
from pathlib import Path
import torch
from jaxtyping import Float
from torch import Tensor
import glob



def generate_box(img):
    # generate bbox
    input_mask = img[..., 3:] > 125
    rows = np.any(input_mask, axis=1)
    cols = np.any(input_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    # Create the bounding box (top-left and bottom-right coordinates)
    bbox = [col_min, row_min, col_max, row_max]

    return bbox


def look_at_opencv(eye):
    down = np.array([0, 0, -1])
    # right = np.array([0,1,0])
    direction = -eye
    direction /= np.linalg.norm(direction)

    right = np.cross(down, direction)
    right /= np.linalg.norm(right)

    new_down = np.cross(direction, right)

    lookat_matrix = np.identity(4)
    lookat_matrix[:3, 0] = right
    lookat_matrix[:3, 1] = new_down
    lookat_matrix[:3, 2] = direction
    lookat_matrix[:3, 3] = eye

    # lookat_matrix[:3,:3] = lookat_matrix[:3,:3].T

    return lookat_matrix

def readColmap(root, ref_width, ref_height, radius, flip, num=5):
    cameras_extrinsic_file = os.path.join(root, "sparse/0", "images.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)

    imgs = []
    masks = []
    normals = []
    depths = []
    poses = []
    filenames = []

    hq_radius=0
    max_box_w=0

    img_dir = os.path.join(root, "images")
    ## extract hq radius
    img_list = glob.glob(os.path.join(img_dir, "*.jpg"))
    img_list.sort()

    hq_idx = []
    tan_value = 90
    hq_img_path=None
    for i, img_path in enumerate(img_list):
        basename = os.path.basename(img_path)
        if "rgba" in basename or "normal" in basename or "depth" in basename:
            continue

        extr = None
        for key in cam_extrinsics:
            extr_tmp = cam_extrinsics[key]
            if extr_tmp.name == basename:
                extr = extr_tmp



        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        T = -R @ T

        pose = np.eye(4)
        pose[:3, :3] = R[:3, :3]
        pose[:3, -1] = T

        img_path = os.path.join(root, "images", extr.name[:-4] + "_rgba.png")
        img = cv2.cvtColor(
            cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )

        tan = abs(np.arctan2(T[0], T[1]))
        if tan < tan_value:
            tan_value = tan
            hq_img_path = img_path
            pose_hq = pose
            hq_radius = np.linalg.norm(T)
            hq_idx=[i]


    for i, img_path in enumerate(img_list):
        basename = os.path.basename(img_path)
        if "rgba" in basename or "normal" in basename or "depth" in basename:
            continue

        extr = None
        for key in cam_extrinsics:
            extr_tmp = cam_extrinsics[key]
            if extr_tmp.name == basename:
                extr = extr_tmp

        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        T = -R @ T

        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, -1] = T


        img_path = os.path.join(root, "images", extr.name[:-4]+"_rgba.png")
        depth_path = img_path.replace("_rgba.png", "_depth.png")
        normal_path = img_path.replace("_rgba.png", "_normal.png")


        image_name = os.path.basename(img_path)
        filenames.append(image_name)


        img = cv2.cvtColor(
            cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)

        # resize and normalize
        img = cv2.resize(img, (ref_width, ref_height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        input_mask = img[..., 3:] > 0.5
        # white bg
        input_img = img[..., :3] * input_mask + (1 - input_mask)

        normal = cv2.resize(
            normal, (ref_width, ref_height), interpolation=cv2.INTER_AREA
        )

        depth = cv2.resize(
            depth, (ref_width, ref_height), interpolation=cv2.INTER_AREA
        )


        pose[:3, -1] = pose[:3, -1] / hq_radius * radius
        pose = pose.astype(np.float32)
        pose = look_at_opencv(pose[:3, -1])

        pose[:3, 1:3] *= -1  # OPENCV to OPENGL
        pose = pose.astype(np.float32)

        # plt.imshow(input_img)
        # plt.show()
        # plt.imshow(depth)
        # plt.show()
        # plt.imshow(input_mask)
        # plt.show()
        # plt.imshow(normal)
        # plt.show()





        rgb_torch: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(input_img.astype(np.float32)).unsqueeze(0).contiguous()
        )
        mask_torch: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(input_mask).unsqueeze(0)
        )

        normal_torch: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(normal.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        )
        depth_torch: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(depth.astype(np.float32) / 255.0).unsqueeze(0)
        )

        imgs.append(rgb_torch)
        masks.append(mask_torch)
        normals.append(normal_torch)
        depths.append(depth_torch)
        poses.append(pose)




    if flip: # save y>0 and flip then
        origin_len = len(imgs)
        for i in range(len(imgs)):
            imgs.append(copy.deepcopy(torch.flip(imgs[i], dims=[2])))
            masks.append(copy.deepcopy(torch.flip(masks[i], dims=[2])))
            depths.append(copy.deepcopy(torch.flip(depths[i], dims=[2])))
            normals.append(copy.deepcopy(torch.flip(normals[i], dims=[2])))
            filenames.append(filenames[i])

            x_factor = [-1, 1, -1]
            y_factor = [1, -1, 1]
            z_factor = [1, -1, 1]

            sym_pose = copy.deepcopy(poses[i])
            sym_pose[:3, 0] = sym_pose[:3, 0] * x_factor
            sym_pose[:3, 1] = sym_pose[:3, 1] * y_factor
            sym_pose[:3, 2] = sym_pose[:3, 2] * z_factor
            sym_pose[1,-1] =  sym_pose[1, -1] * -1


            poses.append(sym_pose)

        hq_idx.append(hq_idx[0]+origin_len)


    return imgs, masks, depths, normals,  poses, filenames, hq_idx

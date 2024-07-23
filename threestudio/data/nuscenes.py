import bisect
import copy
import math
import os
import random
from dataclasses import dataclass, field

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable,BaseModule
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
from threestudio.data.read_nuscenes import nuscenes_loader, cartesian_to_spherical
from threestudio.data.normal_depth import predict_normal_depth
from threestudio.data.read_car360 import readColmap
from threestudio.data.read_li import lidata_loader



@dataclass
class NuscenesImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    rays_noise_scale: float = 0
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False
    rays_d_normalize: bool = True
    use_mixed_camera_config: bool = False
    flip: bool = False
    mv: bool = False




class NuscenesImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: NuscenesImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            # FIXME:
            if self.cfg.use_mixed_camera_config:
                if self.rank % 2 == 0:
                    random_camera_cfg.camera_distance_range=[self.cfg.default_camera_distance, self.cfg.default_camera_distance]
                    random_camera_cfg.fovy_range=[self.cfg.default_fovy_deg, self.cfg.default_fovy_deg]
                    self.fixed_camera_intrinsic = True
                else:
                    self.fixed_camera_intrinsic = False
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )



        ##### intrinsics
        elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180
        camera_position: Float[Tensor, "1 3"] = torch.stack(
            [
                camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                camera_distance * torch.sin(elevation),
            ],
            dim=-1,
        )

        center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
        up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[None]

        light_position: Float[Tensor, "1 3"] = camera_position
        lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
        right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        self.c2w: Float[Tensor, "1 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_position[:, :, None]],
            dim=-1,
        )
        self.c2w4x4: Float[Tensor, "B 4 4"] = torch.cat(
            [self.c2w, torch.zeros_like(self.c2w[:, :1])], dim=1
        )
        self.c2w4x4[:, 3, 3] = 1.0

        self.camera_position = camera_position
        self.light_position = light_position
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.focal_length = self.focal_lengths[0]
        self.prev_height = self.height

        self.rays_o, self.rays_d, self.mvp_mtx = self.set_rays(self.c2w.cpu())



        ## loding nuscenes dataset
        if os.path.exists(os.path.join(self.cfg.image_path, "extrinsics.npy")):
            load_data= (
                nuscenes_loader(self.cfg.image_path, self.width, self.height,
                                radius=self.cfg.default_camera_distance, flip=self.cfg.flip))

        elif os.path.exists(os.path.join(self.cfg.image_path, "sparse/0")):
            load_data = (
                readColmap(self.cfg.image_path, self.width, self.height,
                           radius=self.cfg.default_camera_distance, flip=self.cfg.flip))

        elif os.path.exists(os.path.join(self.cfg.image_path, "cam_pose.npy")):
            load_data = (
                lidata_loader(self.cfg.image_path, self.width, self.height,
                                radius=self.cfg.default_camera_distance, flip=self.cfg.flip))


        self.imgs, self.masks, self.depths, self.normals, self.poses, filenames, self.hq_idx = load_data
        ## initlalize delta pose
        # len_pose = len(self.poses)
        # self.delta_T = torch.nn.Parameter(torch.zeros([len_pose, 3], device="cuda")).requires_grad_(True)
        # detal_R_init = torch.zeros([len_pose, 3, 2], device="cuda")
        # for i in range(len_pose):
        #     detal_R_init.data[i] = torch.eye(3, 2, device="cuda")
        # self.delta_R = torch.nn.Parameter(detal_R_init).requires_grad_(True)

        self.batches=[]
        poses = []
        rgbs = []
        for i in range(len(self.imgs)):
            rgb = self.imgs[i].to(self.rank)
            mask =  self.masks[i].to(self.rank)
            depth = self.depths[i].to(self.rank)
            normal = self.normals[i].to(self.rank)
            rgbs.append(rgb)


            c2w4x4 =  self.poses[i]
            c2w4x4: Float[Tensor, "B 4 4"] = torch.unsqueeze(torch.from_numpy(c2w4x4), 0).cuda()
            poses.append(c2w4x4)
            batch = {
                "c2w4x4": c2w4x4,
                "light_positions": self.light_position,
                "rgb": rgb,
                "ref_depth": depth,
                "ref_normal": normal,
                "mask": mask,
                "height": self.cfg.height,
                "width": self.cfg.width,
                "idx": i,
                "filename": filenames[i],
                "hq_idx":self.hq_idx
                # "delta_T": self.delta_T,
                # "delta_R": self.delta_R,

            }

            self.batches.append(batch)

        self.rgb = torch.concatenate(rgbs, 0)
        self.poses = torch.cat(poses, dim=0)


    def safe_normalize(self, v, epsilon=1e-6):
        norm = torch.norm(v, dim=-1)
        return v / (norm + epsilon)

    def sixD_to_mtx(self, r):
        b1 = r[..., 0]
        # b1 = b1 / torch.norm(b1, dim=-1)
        b1 = self.safe_normalize(b1)
        b2 = r[..., 1] - torch.sum(b1 * r[..., 1], dim=-1)* b1
        # b2 = b2 / torch.norm(b2, dim=-1)
        b2 = self.safe_normalize(b2)
        b3 = torch.cross(b1, b2)

        return torch.stack([b1, b2, b3], dim=-1)



    def set_rays(self, c2w):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = copy.deepcopy(self.directions_unit_focal[None])
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

        rays_o, rays_d = get_rays(
            directions,
            c2w,
            keepdim=True,
            noise_scale=self.cfg.rays_noise_scale,
            normalize=self.cfg.rays_d_normalize,
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.01, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return rays_o, rays_d, mvp_mtx



    def get_all_images(self):
        return self.rgb



    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")

        self.rays_o, self.rays_d, self.mvp_mtx = self.set_rays(self.c2w.cpu())


class NuscenesImageIterableDataset(IterableDataset, NuscenesImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)


    def collate(self, batch) -> Dict[str, Any]:
        if len(self.batches) > 1:
            if self.cfg.mv:
                i = random.randint(0, len(self.batches) - 1)
            else:
                i = self.hq_idx[0]
            batch = copy.deepcopy(self.batches[i])
            # c2w4x4 = batch['c2w4x4']

            # c2w4x4 = batch['c2w4x4'].clone().detach()
            # c2w4x4.retain_graph = True
            # c2w4x4.retain_grad = True

            ## optimze pose
            # delta_T = self.delta_T[i].clone()
            # delta_R = self.delta_R[i].clone()
            # delta_R_mtx = self.sixD_to_mtx(delta_R)
            #
            # c2w4x4[0, :3, :3] = torch.matmul(c2w4x4[0, :3, :3].clone(), delta_R_mtx)
            # c2w4x4[0, :3, 3] = c2w4x4[0, :3, -1].clone() + delta_T
            # c2w = c2w4x4[:, :3].clone()

            # posemlp optimize
    #------------------------------------------
        else:

            batch = self.batches[0]
            batch.update({
                "rays_o": self.rays_o,
                "rays_d": self.rays_d,
                "mvp_mtx": self.mvp_mtx,
                "camera_positions": self.camera_position,
                "light_positions": self.light_position,
                "elevation": self.elevation_deg,
                "azimuth": self.azimuth_deg,
                "camera_distances": self.camera_distance,
                "height": self.cfg.height,
                "width": self.cfg.width,
                "c2w": self.c2w,
                "c2w4x4": self.c2w4x4,
                "idx":None
            })

        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}






class NuscenesTestDataset(Dataset, NuscenesImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)
        self.split  = split
    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        batch = self.random_pose_generator[index]
        batch.update(
            {
            "height": self.random_pose_generator.cfg.eval_height,
            "width": self.random_pose_generator.cfg.eval_width,
            # "mvp_mtx_ref": self.mvp_mtx[0],
            # "c2w_ref": self.c2w4x4,
            }
        )
        # else:
        #     batch = self.batches[0]
        #     batch.update(
        #         {
        #             "light_positions": self.light_position[0],
        #             "camera_distances": torch.norm(self.c2w4x4[0,:3,-1]),
        #             "height": self.random_pose_generator.cfg.eval_height,
        #             "width": self.random_pose_generator.cfg.eval_width,
        #             "mvp_mtx_ref": self.mvp_mtx[0],
        #             "c2w_ref": self.c2w4x4,
        #             "index": 0,
        #         }
        #     )
        #
        #     for k, v in batch.items():
        #         if  isinstance(v, torch.Tensor) :
        #             if len(v.shape) > 1:
        #                 batch[k] = v[0]

        return batch



@register("nuscenes-image-datamodule")
class NuscenesImageDataModule(pl.LightningDataModule):
    cfg: NuscenesImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(NuscenesImageDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = NuscenesImageIterableDataset(self.cfg, "train")
            # self.train_dataset = NuscenesImageDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = NuscenesTestDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = NuscenesTestDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

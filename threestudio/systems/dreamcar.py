import os
import random
import shutil
from dataclasses import dataclass, field
import cv2
import copy
import clip
import torch
import shutil
import numpy as np
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.utils.misc import get_rank, get_device, load_module_weights
from threestudio.utils.perceptual import PerceptualLoss

from threestudio.data.read_nuscenes import nuscenes_loader, cartesian_to_spherical
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)

import pytorch_lightning as pl

from kornia.geometry.conversions import axis_angle_to_rotation_matrix
from kornia.geometry.conversions import rotation_matrix_to_axis_angle
import torch.nn as nn
import torch





from threestudio.data.read_nuscenes import nuscenes_loader, cartesian_to_spherical
from threestudio.data.read_car360 import readColmap
from threestudio.data.read_li import lidata_loader




def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    c2w = torch.eye(4).type_as(r)
    R = axis_angle_to_rotation_matrix(r.unsqueeze(0))[0]  # (3, 3)
    c2w[:3, :3] = R
    c2w[:3, 3] = t

    return c2w

class PoseMLP(pl.LightningModule):
    def __init__(self, num_cams, init_c2w=None, hidden_dim=256, mode='porf', scale=1e-6):
        """
        :param num_cams:
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(PoseMLP, self).__init__()
        self.num_cams = num_cams
        self.scale = scale
        self.mode = mode

        self.init_c2w = init_c2w.clone().detach()
        self.init_r = []
        self.init_t = []
        for idx in range(num_cams):
            r_init = rotation_matrix_to_axis_angle(self.init_c2w[idx][:3, :3].reshape([1, 3, 3])).reshape(-1)
            t_init = self.init_c2w[idx][:3, 3].reshape(-1)
            self.init_r.append(r_init)
            self.init_t.append(t_init)
        self.init_r = torch.stack(self.init_r)  # nx3
        self.init_t = torch.stack(self.init_t)  # nx3


        d_in = 7  # 1 cam_id + 6 pose

        self.elu = nn.ELU(inplace=False)

        self.layer1 = nn.Linear(d_in, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, 6)

        print('init_r range: ', [self.init_r.min(), self.init_r.max()])
        print('init_t range: ', [self.init_t.min(), self.init_t.max()])

    def get_init_pose(self, cam_id):
        return self.init_c2w[cam_id]

    def forward(self, cam_id):
        cam_id_tensor = torch.tensor([cam_id]).type_as(self.init_c2w)
        cam_id_tensor = (cam_id_tensor / self.num_cams) * 2 - 1  # range [-1, +1]

        init_r = self.init_r[cam_id]
        init_t = self.init_t[cam_id]

        if self.mode == 'porf':
            inputs = torch.cat([cam_id_tensor, init_r, init_t], dim=-1)
        elif self.mode == 'pose_only':
            inputs = torch.cat([torch.zeros_like(cam_id_tensor),  init_r, init_t], dim=-1)

        output1 = self.elu(self.layer1(inputs))
        output2 = self.elu(self.layer2(output1))
        output3 = self.elu(self.layer3(output2))
        out = self.layer4(output3)


        # out = self.layers(inputs)
        out = out * self.scale

        # cat pose
        r = out[:3] + self.init_r[cam_id]
        t = out[3:] + self.init_t[cam_id]
        c2w = make_c2w(r, t)  # (4, 4)
        return c2w


@threestudio.register("dreamcar-system")
class ImageConditionDreamFusion(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture'].
        # Note that in the paper we consolidate 'coarse' and 'geometry' into a single phase called 'geometry-sculpting'.
        stage: str = "coarse"
        freq: dict = field(default_factory=dict)
        guidance_3d_type: str = ""
        guidance_3d: dict = field(default_factory=dict)
        use_mixed_camera_config: bool = False
        control_guidance_type: str = ""
        control_guidance: dict = field(default_factory=dict)
        control_prompt_processor_type: str = ""
        control_prompt_processor: dict = field(default_factory=dict)
        visualize_samples: bool = False

        ## for posemlp
        image_path: Optional[str] = None
        flip: bool = False
        default_camera_distance: float = None
        width: int = None
        height: int = None
        fovy: int=None
        pose_optim: Optional[bool] = False


    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.guidance_3d_type != "":
            self.guidance_3d = threestudio.find(self.cfg.guidance_3d_type)(
                self.cfg.guidance_3d
            )
        else:
            self.guidance_3d = None
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        p_config = {}
        self.perceptual_loss = threestudio.find("perceptual-loss")(p_config)

        if not (self.cfg.control_guidance_type == ""):
            self.control_guidance = threestudio.find(self.cfg.control_guidance_type)(self.cfg.control_guidance)
            self.control_prompt_processor = threestudio.find(self.cfg.control_prompt_processor_type)(
                self.cfg.control_prompt_processor
            )
            self.control_prompt_utils = self.control_prompt_processor()



        ## data preprocessing
        self.height = self.cfg.height
        self.width = self.cfg.width
        self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.fovy]))
        self.directions_unit_focals = [
            get_ray_directions(H=self.height, W=self.width, focal=1.0)

        ]

        self.focal_length = 0.5 * self.height / torch.tan(0.5 * self.fovy)

        self.directions_unit_focal = self.directions_unit_focals[0]



        if os.path.exists(os.path.join(self.cfg.image_path, "extrinsics.npy")):
            load_data = (
                nuscenes_loader(self.cfg.image_path, self.width, self.height,
                                radius=self.cfg.default_camera_distance, flip=self.cfg.flip))
        elif os.path.exists(os.path.join(self.cfg.image_path, "cam_pose.npy")):
            load_data = (
                lidata_loader(self.cfg.image_path, self.width, self.height,
                              radius=self.cfg.default_camera_distance, flip=self.cfg.flip))
        else:
            load_data = (
                readColmap(self.cfg.image_path, self.width, self.height,
                           radius=self.cfg.default_camera_distance, flip=self.cfg.flip))
        self.imgs, self.masks, self.depths, self.normals, self.poses, self.filenames, self.hq_idx = load_data

        self.single_view = False
        if len(self.imgs) // 2 == 1:
            self.single_view = True

        poses=[]
        for i in range(len(self.imgs)):
            c2w4x4 =  self.poses[i]
            c2w4x4: Float[Tensor, "B 4 4"] = torch.unsqueeze(torch.from_numpy(c2w4x4), 0).cuda()
            poses.append(c2w4x4)
        self.poses = torch.cat(poses, dim=0)


        self.posemlp = PoseMLP(num_cams=len(self.poses), init_c2w=self.poses)
        self.posemlp = self.posemlp.cuda()
        if self.cfg.pose_optim:   ## for nerf stage
            self.posemlp.train()

        else:
            posemlp_state_dict = {}
            if self.cfg.stage != "coarse":
                ckpt = torch.load(self.cfg.geometry_convert_from)["state_dict"]
            else:
                ckpt = torch.load(self.cfg.weights)["state_dict"]

            for k, v in ckpt.items():
                if "posemlp" in k:
                    posemlp_state_dict[k[8:]] = v
            result = self.posemlp.load_state_dict(posemlp_state_dict, strict=False)
            print(f"Missing posemlp keys: {result.missing_keys}")
            print(f"Unexpected posemlp keys: {result.unexpected_keys}")
            self.posemlp.eval()


        ## view parameters
        lambda_views = torch.zeros(len(self.imgs))
        lambda_views[self.hq_idx[0]] = 1
        lambda_views[self.hq_idx[1]] = 1
        self.lambda_views = nn.Parameter(lambda_views).requires_grad_(True)
        # self.optimizer_views = torch.optim.Adam([self.lambda_views], lr=1e-4 )




    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "texture":
            render_out = self.renderer(**batch, render_mask=True)
        else:
            render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

        # visualize all training images
        all_images = self.trainer.datamodule.train_dataloader().dataset.get_all_images()
        self.save_image_grid(
            "all_training_images.png",
            [
                {"type": "rgb", "img": image, "kwargs": {"data_format": "HWC"}}
                for image in all_images
            ],
            name="on_fit_start",
            step=self.true_global_step,
        )

        self.pearson = PearsonCorrCoef().to(self.device)

    def training_substep(self, batch, batch_idx, guidance: str, render_type="rgb"):
        """
        Args:
            guidance: one of "ref" (reference image supervision), "guidance"
        """

        gt_mask = batch["mask"]
        gt_rgb = batch["rgb"]
        gt_depth = batch["ref_depth"]
        gt_normal = batch["ref_normal"]
        mvp_mtx_ref = batch["mvp_mtx"]
        c2w_ref = batch["c2w4x4"]

        if guidance == "guidance":
            batch = batch["random_camera"]

        # Support rendering visibility mask
        batch["mvp_mtx_ref"] = mvp_mtx_ref
        batch["c2w_ref"] = c2w_ref

        out = self(batch)
        loss_prefix = f"loss_{guidance}_"

        loss_terms = {}

        def set_loss(name, value):
            loss_terms[f"{loss_prefix}{name}"] = value

        guidance_eval = (
            guidance == "guidance"
            and self.cfg.freq.guidance_eval > 0
            and self.true_global_step % self.cfg.freq.guidance_eval == 0
        )

        prompt_utils = self.prompt_processor()

        if guidance == "ref":
            if render_type == "rgb":
                # color loss. Use l2 loss in coarse and geometry satge; use l1 loss in texture stage.
                if self.C(self.cfg.loss.lambda_rgb) > 0:
                    gt_rgb = gt_rgb * gt_mask.float() + out["comp_rgb_bg"] * (
                        1 - gt_mask.float()
                    )
                    pred_rgb = out["comp_rgb"]
                    if self.cfg.stage in ["coarse", "geometry"]:
                        set_loss("rgb", F.mse_loss(gt_rgb, pred_rgb))
                    else:
                        if self.cfg.stage == "texture":
                            grow_mask = F.max_pool2d(1 - gt_mask.float().permute(0, 3, 1, 2), (9, 9), 1, 4)
                            grow_mask = (1 - grow_mask).permute(0, 2, 3, 1)
                            set_loss("rgb", F.l1_loss(gt_rgb*grow_mask, pred_rgb*grow_mask))
                        else:
                            set_loss("rgb", F.l1_loss(gt_rgb, pred_rgb))

                    # grow_mask = F.max_pool2d(1 - gt_mask.float().permute(0, 3, 1, 2), (9, 9), 1, 4)
                    # grow_mask = (1 - grow_mask).permute(0, 2, 3, 1)
                    # set_loss("rgb", F.l1_loss(gt_rgb * grow_mask, pred_rgb * grow_mask))



                # mask loss
                if self.C(self.cfg.loss.lambda_mask) > 0:
                    set_loss("mask", F.mse_loss(gt_mask.float(), out["opacity"]))

                # mask binary cross loss
                if self.C(self.cfg.loss.lambda_mask_binary) > 0:
                    set_loss("mask_binary", F.binary_cross_entropy(
                    out["opacity"].clamp(1.0e-5, 1.0 - 1.0e-5),
                    batch["mask"].float(),))

                # depth loss
                if self.C(self.cfg.loss.lambda_depth) > 0 and batch["ref_depth"] is not None:
                # if self.C(self.cfg.loss.lambda_depth) > 0:
                    valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)].unsqueeze(1)
                    valid_pred_depth = out["depth"][gt_mask].unsqueeze(1)
                    with torch.no_grad():
                        A = torch.cat(
                            [valid_gt_depth, torch.ones_like(valid_gt_depth)], dim=-1
                        )  # [B, 2]
                        X = torch.linalg.lstsq(A, valid_pred_depth).solution  # [2, 1]
                        valid_gt_depth = A @ X  # [B, 1]
                    set_loss("depth", F.mse_loss(valid_gt_depth, valid_pred_depth))

                # relative depth loss
                if self.C(self.cfg.loss.lambda_depth_rel) > 0 and batch["ref_depth"] is not None:
                # if self.C(self.cfg.loss.lambda_depth_rel) > 0:
                    valid_gt_depth = batch["ref_depth"][gt_mask.squeeze(-1)]  # [B,]
                    valid_pred_depth = out["depth"][gt_mask]  # [B,]
                    set_loss(
                        "depth_rel", 1 - self.pearson(valid_pred_depth, valid_gt_depth)
                    )

            # normal loss
            if self.C(self.cfg.loss.lambda_normal) > 0 and gt_normal is not None:
            # if self.C(self.cfg.loss.lambda_normal) > 0:
                valid_gt_normal = (
                    1 - 2 * gt_normal[gt_mask.squeeze(-1)]
                )  # [B, 3]
                # FIXME: reverse x axis
                pred_normal = out["comp_normal_viewspace"]
                pred_normal[..., 0] = 1 - pred_normal[..., 0]
                valid_pred_normal = (
                    2 * pred_normal[gt_mask.squeeze(-1)] - 1
                )  # [B, 3]
                set_loss(
                    "normal",
                    1 - F.cosine_similarity(valid_pred_normal, valid_gt_normal).mean(),
                )

        elif guidance == "guidance" and self.true_global_step > self.cfg.freq.no_diff_steps:
            if self.cfg.stage == "geometry" and render_type == "normal":
                guidance_inp = out["comp_normal"]
            else:
                guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp,
                prompt_utils,
                **batch,
                rgb_as_latents=False,
                guidance_eval=guidance_eval,
                mask=out["mask"] if "mask" in out else None,
            )
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    set_loss(name.split("_")[-1], value)

            if self.guidance_3d is not None:

                # FIXME: use mixed camera config
                if not self.cfg.use_mixed_camera_config or get_rank() % 2 == 0:
                    guidance_3d_out = self.guidance_3d(
                        out["comp_rgb"],
                        **batch,
                        rgb_as_latents=False,
                        guidance_eval=guidance_eval,
                    )
                    for name, value in guidance_3d_out.items():
                        if not (isinstance(value, torch.Tensor) and len(value.shape) > 0):
                            self.log(f"train/{name}_3d", value)
                        if name.startswith("loss_"):
                           set_loss("3d_"+name.split("_")[-1], value)
                    # set_loss("3d_sd", guidance_out["loss_sd"])

        # Regularization
        if self.C(self.cfg.loss.lambda_normal_smooth) > 0:
            if "comp_normal" not in out:
                raise ValueError(
                    "comp_normal is required for 2D normal smooth loss, no comp_normal is found in the output."
                )
            normal = out["comp_normal"]
            set_loss(
                "normal_smooth",
                (normal[:, 1:, :, :] - normal[:, :-1, :, :]).square().mean()
                + (normal[:, :, 1:, :] - normal[:, :, :-1, :]).square().mean(),
            )

        if self.C(self.cfg.loss.lambda_3d_normal_smooth) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for normal smooth loss, no normal is found in the output."
                )
            if "normal_perturb" not in out:
                raise ValueError(
                    "normal_perturb is required for normal smooth loss, no normal_perturb is found in the output."
                )
            normals = out["normal"]
            normals_perturb = out["normal_perturb"]
            set_loss("3d_normal_smooth", (normals - normals_perturb).abs().mean())

        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                set_loss(
                    "orient",
                    (
                        out["weights"].detach()
                        * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
                    ).sum()
                    / (out["opacity"] > 0).sum(),
                )

            if guidance != "ref" and self.C(self.cfg.loss.lambda_sparsity) > 0:
                set_loss("sparsity", (out["opacity"] ** 2 + 0.01).sqrt().mean())

            if self.C(self.cfg.loss.lambda_opaque) > 0:
                opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
                set_loss(
                    "opaque", binary_cross_entropy(opacity_clamped, opacity_clamped)
                )

            if "lambda_eikonal" in self.cfg.loss and self.C(self.cfg.loss.lambda_eikonal) > 0:
                if "sdf_grad" not in out:
                    raise ValueError(
                        "SDF grad is required for eikonal loss, no normal is found in the output."
                    )
                set_loss(
                    "eikonal", (
                        (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
                    ).mean()
                )

            if "lambda_z_variance"in self.cfg.loss and self.C(self.cfg.loss.lambda_z_variance) > 0:
                # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
                # helps reduce floaters and produce solid geometry
                loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
                set_loss("z_variance", loss_z_variance)

        elif self.cfg.stage == "geometry":
            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                set_loss("normal_consistency", out["mesh"].normal_consistency())
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                set_loss("laplacian_smoothness", out["mesh"].laplacian())
        elif self.cfg.stage == "texture":
            if self.C(self.cfg.loss.lambda_reg) > 0 and guidance == "guidance" and self.true_global_step % 5 == 0:

                rgb = out["comp_rgb"]
                rgb = F.interpolate(rgb.permute(0, 3, 1, 2), (512, 512), mode='bilinear').permute(0, 2, 3, 1)
                control_prompt_utils = self.control_prompt_processor()
                with torch.no_grad():
                    control_dict = self.control_guidance(
                        rgb=rgb,
                        cond_rgb=rgb,
                        prompt_utils=control_prompt_utils,
                        mask=out["mask"] if "mask" in out else None,
                    )

                    edit_images = control_dict["edit_images"]
                    temp = (edit_images.detach().cpu()[0].numpy() * 255).astype(np.uint8)
                    cv2.imwrite(".threestudio_cache/control_debug.jpg", temp[:, :, ::-1])

                loss_reg = (rgb.shape[1] // 8) * (rgb.shape[2] // 8) * self.perceptual_loss(edit_images.permute(0, 3, 1, 2), rgb.permute(0, 3, 1, 2)).mean()
                set_loss("reg", loss_reg)
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        loss = 0.0
        for name, value in loss_terms.items():
            self.log(f"train/{name}", value)
            if name.startswith(loss_prefix):
                loss_weighted = value * self.C(
                    self.cfg.loss[name.replace(loss_prefix, "lambda_")]
                )
                self.log(f"train/{name}_w", loss_weighted)
                loss = loss + loss_weighted

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        self.log(f"train/loss_{guidance}", loss)

        if guidance_eval:
            self.guidance_evaluation_save(
                out["comp_rgb"].detach()[: guidance_out["eval"]["bs"]],
                guidance_out["eval"],
            )

        return {"loss": loss}


    def set_rays(self, c2w, noise_scale=0):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = copy.deepcopy(self.directions_unit_focal[None])
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

        rays_o, rays_d = get_rays(
            directions,
            c2w,
            keepdim=True,
            noise_scale=noise_scale,
            normalize=True,
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.01, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return rays_o, rays_d, mvp_mtx



    def training_step(self, batch, batch_idx):

        ## update pose
        idx = batch["idx"]
        if idx is not None:
            if self.cfg.pose_optim:  ## for nerf stage
                c2w4x4 = self.posemlp(idx)[None, :, :]
            else:
                with torch.no_grad():
                    c2w4x4 = self.posemlp(idx)[None, :, :]
            ## pose optimization
            camera_distance = torch.norm(c2w4x4[0, :3, -1])
            camera_position: Float[Tensor, "1 3"] = c2w4x4[:, :3, -1]
            polar_system = cartesian_to_spherical(camera_position.cpu().detach().numpy())
            rays_o, rays_d, mvp_mtx = self.set_rays(c2w4x4)

            ## only hq normal and depth for supervision
            # if idx not in self.hq_idx :
            #     batch["ref_depth"] = None
                # batch["ref_normal"] = None

            batch.update({
                "rays_o": rays_o,
                "rays_d": rays_d,
                "mvp_mtx": mvp_mtx,
                "camera_positions": camera_position,
                "elevation": polar_system[0],
                "azimuth": polar_system[1],
                "camera_distances": camera_distance,
                "c2w": c2w4x4[:3,:].clone(),
                "c2w4x4": c2w4x4.clone(),
            })





        if self.cfg.freq.ref_or_guidance == "accumulate":
            do_ref = True
            do_guidance = True
        elif self.cfg.freq.ref_or_guidance == "alternate":
            do_ref = (
                self.true_global_step < self.cfg.freq.ref_only_steps
                or self.true_global_step % self.cfg.freq.n_ref == 0
            )
            do_guidance = not do_ref
            if hasattr(self.guidance.cfg, "only_pretrain_step"):
                if (self.guidance.cfg.only_pretrain_step > 0) and (self.global_step % self.guidance.cfg.only_pretrain_step) < (self.guidance.cfg.only_pretrain_step // 5):
                    do_guidance = True
                    do_ref = False

        if self.cfg.stage == "geometry":
            render_type = "rgb" if self.true_global_step % self.cfg.freq.n_rgb == 0 else "normal"
        else:
            render_type = "rgb"



        total_loss = 0.0

        if do_guidance:
            out = self.training_substep(batch, batch_idx, guidance="guidance", render_type=render_type)
            total_loss = total_loss + out["loss"]

        if do_ref:
            out = self.training_substep(batch, batch_idx, guidance="ref", render_type=render_type)

            if not self.single_view:
                # total_loss = total_loss +  out["loss"] * self.cfg.loss.lambda_mv
                out["loss"] = out["loss"] * self.lambda_views[batch["idx"]]
                total_loss = total_loss +  out["loss"]
            else:
                total_loss = total_loss +  out["loss"]


        self.log("train/loss", total_loss, prog_bar=True)


        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-val/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale",
                        "img": out["depth"][0],
                        "kwargs": {}
                    }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],

            name="validation_step",
            step=self.true_global_step,
        )

        if self.cfg.stage=="texture" and self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        filestem = f"it{self.true_global_step}-val"

        try:
            self.save_img_sequence(
                filestem,
                filestem,
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="validation_epoch_end",
                step=self.true_global_step,
            )
            shutil.rmtree(
                os.path.join(self.get_save_dir(), f"it{self.true_global_step}-val")
            )
        except:
            pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal_viewspace"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal_viewspace" in out
                else []
            )
            + (
                [
                    {
                        "type": "grayscale", "img": out["depth"][0], "kwargs": {}
                        }
                ]
                if "depth" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ]
            + (
                [
                    {
                        "type": "grayscale", "img": out["opacity_vis"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)}
                        }
                ]
                if "opacity_vis" in out
                else []
            )
            ,
            name="test_step",
            step=self.true_global_step,
        )

        out_rgb = out["comp_rgb"][0].cpu().detach().numpy() * 255
        out_rgb = out_rgb.astype("uint8")
        self.save_image(f"test/it{self.true_global_step}-test/{batch['index'][0]}.png", out_rgb)





        # FIXME: save camera extrinsics
        c2w_360 = batch["c2w"]
        save_path = os.path.join(self.get_save_dir(), f"it{self.true_global_step}-test/{batch['index'][0]}.npy")
        np.save(save_path, c2w_360.detach().cpu().numpy()[0])



        ## test mirror
        with torch.no_grad():
            c2w4x4 = self.posemlp(len(self.poses)-1)[None, :, :]
            c2w = c2w4x4[:, :3]

            camera_distance = torch.norm(c2w4x4[0, :3, -1])
            camera_position: Float[Tensor, "1 3"] = c2w4x4[:, :3, -1]
            polar_system = cartesian_to_spherical(camera_position.cpu().detach().numpy())
            rays_o, rays_d, mvp_mtx = self.set_rays(c2w)

        batch.update({
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_position,
            "elevation": polar_system[0],
            "azimuth": polar_system[1],
            "camera_distances": camera_distance,
            "c2w": c2w,
            "c2w4x4": c2w4x4.clone(),
        })
        out = self(batch)
        out_rgb = out["comp_rgb"][0].cpu().detach().numpy() * 255
        out_normal = out["comp_normal"][0].cpu().detach().numpy() * 255
        out_rgb = out_rgb.astype("uint8")
        out_normal = out_normal.astype("uint8")

        self.save_image(os.path.join("test", self.filenames[-1][:-4]+  "_mirror.png"), out_rgb)
        self.save_image(os.path.join("test", self.filenames[-1][:-4]+  "_mirrornormal.png"), out_normal)
        # self.save_image("test/test_mask.png", ~mask*255)
        # self.save_uv_image("test/uv.png", out_rgb, data_range=[0,255], cmap = "color")


        ## test
        with torch.no_grad():
            c2w4x4 = self.posemlp((len(self.poses) - 1) // 2)[None, :, :]
            c2w = c2w4x4[:, :3]

            camera_distance = torch.norm(c2w4x4[0, :3, -1])
            camera_position: Float[Tensor, "1 3"] = c2w4x4[:, :3, -1]
            polar_system = cartesian_to_spherical(camera_position.cpu().detach().numpy())
            rays_o, rays_d, mvp_mtx = self.set_rays(c2w)

        batch.update({
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_position,
            "elevation": polar_system[0],
            "azimuth": polar_system[1],
            "camera_distances": camera_distance,
            "c2w": c2w,
            "c2w4x4": c2w4x4.clone(),
        })
        out = self(batch)
        out_rgb = out["comp_rgb"][0].cpu().detach().numpy() * 255
        out_normal = out["comp_normal"][0].cpu().detach().numpy() * 255
        out_rgb = out_rgb.astype("uint8")
        out_normal = out_normal.astype("uint8")

        self.save_image(os.path.join("test", self.filenames[-1]), out_rgb)
        self.save_image(os.path.join("test", self.filenames[-1][:-4]+ "_normal.png"), out_normal)

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )


        self.save_img_sequence(
            f"test/it{self.true_global_step}-test",
            f"test/it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test_rgb",
            step=self.true_global_step,
        )


    def on_before_optimizer_step(self, optimizer) -> None:
        # print("on_before_opt enter")
        # for n, p in self.geometry.named_parameters():
        #     if p.grad is None:
        #         print(n)
        # print("on_before_opt exit")

        pass

    def on_load_checkpoint(self, checkpoint):
        # raise RuntimeError
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {"guidance."+k : v for (k,v) in self.guidance.state_dict().items()}
        checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
        return

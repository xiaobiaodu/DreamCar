import glob
import importlib
import os
from dataclasses import dataclass, field

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from tqdm import tqdm

import threestudio
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *
from threestudio.data.read_nuscenes import cartesian_to_spherical
from threestudio.data.utils import read_extrinsics_binary, qvec2rotmat

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# load model
def load_model_from_config(config, ckpt, device, vram_O=True, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")

    if "global_step" in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print("[INFO] missing keys: \n", m)
    if len(u) > 0 and verbose:
        print("[INFO] unexpected keys: \n", u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print("[INFO] loading EMA...")
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    model.eval().to(device)

    return model


@threestudio.register("stable-zero123-guidance")
class StableZero123Guidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "load/zero123/stable-zero123.ckpt"
        pretrained_config: str = "load/zero123/sd-objaverse-finetune-c_concat-256.yaml"
        vram_O: bool = True

        cond_image_path: str = "load/images/hamburger_rgba.png"
        cond_elevation_deg: float = 0.0
        cond_azimuth_deg: float = 0.0
        cond_camera_distance: float = 1.2

        guidance_scale: float = 5.0

        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

    cfg: Config



    def configure(self) -> None:
        threestudio.info(f"Loading DreamCar123 ...")

        self.config = OmegaConf.load(self.cfg.pretrained_config)
        # TODO: seems it cannot load into fp16...
        self.weights_dtype = torch.float32
        self.model = load_model_from_config(
            self.config,
            self.cfg.pretrained_model_name_or_path,
            device=self.device,
            vram_O=self.cfg.vram_O,
        )

        for p in self.model.parameters():
            p.requires_grad_(False)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.config.model.params.timesteps

        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        img_dir = os.path.join(self.cfg.cond_image_path, "sam")
        img_dir2 = os.path.join(self.cfg.cond_image_path, "images")

        if os.path.exists(os.path.join(self.cfg.cond_image_path, "masks/sam")):
            img_dir = os.path.join(self.cfg.cond_image_path, "images")


            tan_value = 90

            cameras_extrinsic_file = os.path.join(self.cfg.cond_image_path, "sparse/0", "images.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)


            hq_img_path = None
            pose_hq = None

            img_list = glob.glob(os.path.join(img_dir, "*.jpg"))
            img_list.sort()
            for i, img_path in enumerate(img_list):
                basename = os.path.basename(img_path)
                if "rgba" in basename or "normal" in basename or "depth" in basename:
                    continue

                extr=None
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

                img_path = os.path.join(self.cfg.cond_image_path, "images", basename[:-4] + "_rgba.png")
                img = cv2.cvtColor(
                    cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
                )
                tan = abs(np.arctan2(T[0], T[1]))
                if tan < tan_value:
                    tan_value = tan
                    hq_img_path = img_path
                    pose_hq = pose

            pose_hq[:3, -1] = pose_hq[:3, -1] / np.linalg.norm(pose_hq[:3, -1]) * 3.8
            degrees = cartesian_to_spherical(pose_hq[None, :3, -1])
            theta_deg = degrees[0]
            azimuth_deg = degrees[1]


            self.cfg.cond_elevation_deg = float(theta_deg)
            self.cfg.cond_azimuth_deg = float(azimuth_deg)
            self.prepare_embeddings(hq_img_path)


        elif os.path.exists(img_dir):
            max_radius = 0
            max_box_idx = None
            max_box_w = 0
            pose_hq = None
            tan_value = 90

            img_list = glob.glob(os.path.join(img_dir, "*rgba.png"))
            img_list.sort()
            c2ws = np.load(os.path.join(self.cfg.cond_image_path, "extrinsics.npy"))
            obj_poses = np.load(os.path.join(self.cfg.cond_image_path, "obj_poses.npy"))

            hq_img_path = None
            for i, img_path in enumerate(img_list):
                basename = os.path.basename(img_path)
                img_id = int(basename.split("_")[0])  # img id

                # loading imgs
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                pose = np.linalg.inv(obj_poses[img_id]) @ c2ws[img_id]
                T = pose[:3, -1]

                tan = abs(np.arctan2(T[0], T[1]))
                if tan < tan_value:
                    tan_value = tan
                    hq_img_path = img_path
                    pose_hq = pose

                # bbox = self.generate_box(img)
                # bbox_w = bbox[2] - bbox[0]
                #
                # if bbox_w > max_box_w:
                #     max_box_w = bbox_w
                #     max_box_idx = img_id
                #     hq_img_path = img_path
            pose = pose_hq

            pose[:3, -1] = pose[:3,-1] / np.linalg.norm(pose[:3, -1]) * 3.8
            degrees = cartesian_to_spherical(pose[None,:3, -1])
            theta_deg = degrees[0]
            azimuth_deg = degrees[1]




            self.cfg.cond_elevation_deg = float(theta_deg)
            self.cfg.cond_azimuth_deg = float(azimuth_deg)
            self.prepare_embeddings(hq_img_path)


        elif os.path.exists(img_dir2):
            max_radius = 0
            max_box_idx = None
            max_box_w = 0
            pose_hq = None
            tan_value = 90

            img_list = glob.glob(os.path.join(img_dir2, "*rgba.png"))
            img_list.sort()
            c2ws = np.load(os.path.join(self.cfg.cond_image_path, "cam_pose.npy"))
            obj_poses = np.load(os.path.join(self.cfg.cond_image_path, "obj_pose.npy"))

            hq_img_path = None
            for i, img_path in enumerate(img_list):
                basename = os.path.basename(img_path)
                img_id = int(basename[:-9])  # img id

                # loading imgs
                # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                pose = np.linalg.inv(obj_poses[img_id]) @ c2ws[img_id]
                T = pose[:3, -1]

                tan = abs(np.arctan2(T[0], T[1]))
                if tan < tan_value:
                    tan_value = tan
                    hq_img_path = img_path
                    pose_hq = pose

                # bbox = self.generate_box(img)
                # bbox_w = bbox[2] - bbox[0]
                #
                # if bbox_w > max_box_w:
                #     max_box_w = bbox_w
                #     max_box_idx = img_id
                #     hq_img_path = img_path
            pose = pose_hq

            pose[:3, -1] = pose[:3, -1] / np.linalg.norm(pose[:3, -1]) * 3.8
            degrees = cartesian_to_spherical(pose[None, :3, -1])
            theta_deg = degrees[0]
            azimuth_deg = degrees[1]

            self.cfg.cond_elevation_deg = float(theta_deg)
            self.cfg.cond_azimuth_deg = float(azimuth_deg)
            self.prepare_embeddings(hq_img_path)
        else:
            self.prepare_embeddings(self.cfg.cond_image_path)



        if "stable" in self.cfg.pretrained_model_name_or_path:
            threestudio.info(f"Loaded Stable Zero123! ckpt: {self.cfg.pretrained_model_name_or_path}" )
        else:
            threestudio.info(f"Loaded Zero123! ckpt: {self.cfg.pretrained_model_name_or_path}", )

    def generate_box(self, img):
        # generate bbox
        input_mask = img[..., 3:]>125
        rows = np.any(input_mask, axis=1)
        cols = np.any(input_mask, axis=0)
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]

        # Create the bounding box (top-left and bottom-right coordinates)
        bbox = [col_min, row_min, col_max, row_max]

        return bbox
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def prepare_embeddings(self, image_path: str) -> None:
        # load cond image for zero123
        assert os.path.exists(image_path)
        rgba = cv2.cvtColor(
            cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )

        # plt.imshow(rgba)
        # plt.show()

        rgba = (
            cv2.resize(rgba, (256, 256), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            / 255.0
        )
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])
        self.rgb_256: Float[Tensor, "1 3 H W"] = (
            torch.from_numpy(rgb)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
            .contiguous()
            .to(self.device)
        )
        self.c_crossattn, self.c_concat = self.get_img_embeds(self.rgb_256)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_img_embeds(
        self,
        img: Float[Tensor, "B 3 256 256"],
    ) -> Tuple[Float[Tensor, "B 1 768"], Float[Tensor, "B 4 32 32"]]:
        img = img * 2.0 - 1.0
        c_crossattn = self.model.get_learned_conditioning(img.to(self.weights_dtype))
        c_concat = self.model.encode_first_stage(img.to(self.weights_dtype)).mode()
        return c_crossattn, c_concat

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(imgs.to(self.weights_dtype))
        )
        return latents.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        image = self.model.decode_first_stage(latents)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_cond(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c_crossattn=None,
        c_concat=None,
        **kwargs,
    ) -> dict:
        if "stable"  in self.cfg.pretrained_model_name_or_path:
            T = torch.stack(
                [
                    torch.deg2rad(
                        (90-elevation)  - (90-self.cfg.cond_elevation_deg)
                    ),  # Zero123 polar is 90-elevation
                    torch.sin(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                    torch.cos(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                    torch.deg2rad(
                        90 - torch.full_like(elevation, 0)
                    ),
                ],
                dim=-1,
            )[:, None, :].to(self.device)
        else:
            T = torch.stack(
                [
                    torch.deg2rad(
                        (90 - elevation) - (90 - self.cfg.cond_elevation_deg)
                    ),  # Zero123 polar is 90-elevation
                    torch.sin(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                    torch.cos(torch.deg2rad(azimuth - self.cfg.cond_azimuth_deg)),
                    torch.deg2rad(
                        0 - torch.full_like(elevation, 0)
                    ),
                ],
                dim=-1,
            )[:, None, :].to(self.device)


        cond = {}
        clip_emb = self.model.cc_projection(
            torch.cat(
                [
                    (self.c_crossattn if c_crossattn is None else c_crossattn).repeat(
                        len(T), 1, 1
                    ),
                    T,
                ],
                dim=-1,
            )
        )
        cond["c_crossattn"] = [
            torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)
        ]
        cond["c_concat"] = [
            torch.cat(
                [
                    torch.zeros_like(self.c_concat)
                    .repeat(len(T), 1, 1, 1)
                    .to(self.device),
                    (self.c_concat if c_concat is None else c_concat).repeat(
                        len(T), 1, 1, 1
                    ),
                ],
                dim=0,
            )
        ]
        return cond

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        if rgb_as_latents:
            latents = (
                F.interpolate(rgb_BCHW, (32, 32), mode="bilinear", align_corners=False)
                * 2
                - 1
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (256, 256), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        cond = self.get_cond(elevation, azimuth, camera_distances)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)
            noise_pred = self.model.apply_model(x_in, t_in, cond)

        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        w = (1 - self.alphas[t]).reshape(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # loss = SpecifyGradient.apply(latents, grad)
        # SpecifyGradient is not straghtforward, use a reparameterization trick instead
        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        guidance_out = {
            "loss_sd": loss_sds,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        return guidance_out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

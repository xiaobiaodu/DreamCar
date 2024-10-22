import random
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.utils.import_utils import is_xformers_available

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.typing import *


class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return self.module(x).to(self.dtype)


@threestudio.register("stable-diffusion-bsd-guidance")
class StableDiffusionBSDGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        cache_dir: Optional[str] = None
        local_files_only: Optional[bool] = False
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base"
        pretrained_model_name_or_path_lora: str = "stabilityai/stable-diffusion-2-1"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        guidance_scale_lora: float = 1.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True
        lora_cfg_training: bool = True
        lora_n_timestamp_samples: int = 1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        view_dependent_prompting: bool = True
        camera_condition_type: str = "extrinsics"

        use_du: bool = False
        per_du_step: int = 10
        start_du_step: int = 0
        du_diffusion_steps: int = 20

        lora_pretrain_cfg_training: bool = True
        lora_pretrain_n_timestamp_samples: int = 1
        per_update_pretrain_step: int = 25
        only_pretrain_step: int = 1000




    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
            "local_files_only": self.cfg.local_files_only
        }

        pipe_lora_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
            "local_files_only": self.cfg.local_files_only
        }

        @dataclass
        class SubModules:
            pipe: StableDiffusionPipeline
            pipe_lora: StableDiffusionPipeline
            pipe_fix: StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)
        self.single_model = False
        pipe_lora = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            **pipe_lora_kwargs,
        ).to(self.device)
        del pipe_lora.vae
        cleanup()
        pipe_lora.vae = pipe.vae

        pipe_fix = pipe



        # if self.cfg.lora_weights_path is not None:
        #     pipe.load_lora_weights(self.cfg.lora_weights_path)
        #     pipe_lora.load_lora_weights(self.cfg.lora_weights_path)
        #     pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
        #     pipe_lora.scheduler = pipe_lora.scheduler.__class__.from_config(pipe_lora.scheduler.config, variance_type="fixed_small")
        #
        #     print("loading lora wights", self.cfg.lora_weights_path)

        self.submodules = SubModules(pipe=pipe, pipe_lora=pipe_lora, pipe_fix=pipe_fix)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()
                self.pipe_lora.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            self.pipe_lora.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)
            self.pipe_lora.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe_lora.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        if not self.single_model:
            del self.pipe_lora.text_encoder
        cleanup()

        for p in self.vae.parameters():
            p.requires_grad_(False)

        for p in self.vae_fix.parameters():
            p.requires_grad_(False)
        for p in self.unet_fix.parameters():
            p.requires_grad_(False)

        # FIXME: hard-coded dims
        self.camera_embedding = ToWeightsDType(
            TimestepEmbedding(16, 1280), self.weights_dtype
        ).to(self.device)
        # self.unet_lora.class_embedding = self.camera_embedding

        # set up LoRA layers
        # self.set_up_lora_layers(self.unet_lora)
        # self.lora_layers = AttnProcsLayers(self.unet_lora.attn_processors).to(
        #     self.device
        # )
        # self.lora_layers._load_state_dict_pre_hooks.clear()
        # self.lora_layers._state_dict_hooks.clear()

        # set up LoRA layers for pretrain
        # self.set_up_lora_layers(self.unet)
        # self.lora_layers_pretrain = AttnProcsLayers(self.unet.attn_processors).to(
        #     self.device
        # )
        # self.lora_layers_pretrain._load_state_dict_pre_hooks.clear()
        # self.lora_layers_pretrain._state_dict_hooks.clear()

        self.train_unet = UNet2DConditionModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="unet",
            torch_dtype=self.weights_dtype
        )
        self.train_unet.enable_xformers_memory_efficient_attention()
        self.train_unet.enable_gradient_checkpointing()

        self.train_unet_lora = UNet2DConditionModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora, subfolder="unet",
            torch_dtype=self.weights_dtype
        )
        self.train_unet_lora.enable_xformers_memory_efficient_attention()
        self.train_unet_lora.enable_gradient_checkpointing()

        for p in self.train_unet.parameters():
            p.requires_grad_(True)
        for p in self.train_unet_lora.parameters():
            p.requires_grad_(True)
        # for p in self.lora_layers.parameters():
        #     p.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_pretrained( # DDPM
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
            local_files_only=self.cfg.local_files_only,
        )

        self.scheduler_lora = DDPMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path_lora,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
            local_files_only=self.cfg.local_files_only,
        )

        self.scheduler_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.scheduler_lora_sample = DPMSolverMultistepScheduler.from_config(
            self.pipe_lora.scheduler.config
        )

        self.pipe.scheduler = self.scheduler
        self.pipe_lora.scheduler = self.scheduler_lora

        self.pipe_fix.scheduler = self.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)

        self.grad_clip_val: Optional[float] = None

        if self.cfg.use_du:
            self.perceptual_loss = PerceptualLoss().eval().to(self.device)
            for p in self.perceptual_loss.parameters():
                p.requires_grad_(False)

        self.cache_frames = []

        threestudio.info(f"Loaded Stable Diffusion!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @property
    def pipe(self):
        return self.submodules.pipe

    @property
    def pipe_lora(self):
        return self.submodules.pipe_lora

    @property
    def unet(self):
        return self.train_unet

    @property
    def unet_lora(self):
        return self.train_unet_lora

    @property
    def vae(self):
        return self.submodules.pipe.vae

    @property
    def vae_lora(self):
        return self.submodules.pipe_lora.vae

    @property
    def pipe_fix(self):
        return self.submodules.pipe_fix

    @property
    def unet_fix(self):
        return self.submodules.pipe_fix.unet

    @property
    def vae_fix(self):
        return self.submodules.pipe_fix.vae

    def set_up_lora_layers(self, unet):
        # set up LoRA layers
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size, cross_attention_dim=cross_attention_dim
            )

        unet.set_attn_processor(lora_attn_procs)

        return lora_attn_procs

    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        pipe: StableDiffusionPipeline,
        sample_scheduler: DPMSolverMultistepScheduler,
        text_embeddings: Float[Tensor, "BB N Nf"],
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        class_labels: Optional[Float[Tensor, "BB 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents_inp: Optional[Float[Tensor, "..."]] = None,
    ) -> Float[Tensor, "B H W 3"]:
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        height = height or pipe.unet.config.sample_size * vae_scale_factor
        width = width or pipe.unet.config.sample_size * vae_scale_factor
        batch_size = text_embeddings.shape[0] // 2
        device = self.device

        sample_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = sample_scheduler.timesteps
        num_channels_latents = pipe.unet.config.in_channels

        if latents_inp is not None:
            B = latents_inp.shape[0]
            t = torch.randint(
                self.max_step,
                self.max_step+1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            noise = torch.randn_like(latents_inp)
            # latents = sample_scheduler.add_noise(latents_inp, noise, t).to(self.weights_dtype)

            init_timestep = max(1, min(int(num_inference_steps * t[0].item() / self.num_train_timesteps), num_inference_steps))
            t_start = max(num_inference_steps - init_timestep, 0)
            latent_timestep = sample_scheduler.timesteps[t_start : t_start + 1].repeat(batch_size)
            latents = sample_scheduler.add_noise(latents_inp, noise, latent_timestep).to(self.weights_dtype)

        else:
            latents = pipe.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                self.weights_dtype,
                device,
                generator,
            )
            t_start = 0

        for i, t in enumerate(timesteps[t_start:]):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = sample_scheduler.scale_model_input(
                latent_model_input, t
            )
            t_start = 0

            # predict the noise residual
            if class_labels is None:
                with self.disable_unet_class_embedding(pipe.unet) as unet:
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
            else:
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings.to(self.weights_dtype),
                    class_labels=class_labels,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = sample_scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / pipe.vae.config.scaling_factor * latents
        images = pipe.vae.decode(latents).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        images = images.permute(0, 2, 3, 1).float()

        return images

    def sample(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
        generator = torch.Generator(device=self.device).manual_seed(seed)

        return self._sample(
            pipe=self.pipe,
            sample_scheduler=self.scheduler_sample,
            text_embeddings=text_embeddings_vd,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale,
            cross_attention_kwargs=cross_attention_kwargs,
            generator=generator,
        )

    def sample_img2img(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        seed: int = 0,
        mask = None,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        mask_BCHW = mask.permute(0, 3, 1, 2)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=False) # TODO: 有部分概率是du或者ref image

        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # return self._sample(
        #     pipe=self.pipe,
        #     sample_scheduler=self.scheduler_sample,
        #     text_embeddings=text_embeddings_vd,
        #     num_inference_steps=25,
        #     guidance_scale=self.cfg.guidance_scale,
        #     cross_attention_kwargs=cross_attention_kwargs,
        #     generator=generator,
        #     latents_inp=latents
        # )

        return self.compute_grad_du(latents, rgb_BCHW, text_embeddings_vd, mask=mask_BCHW)

    def sample_lora(
        self,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        seed: int = 0,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        # input text embeddings, view-independent
        text_embeddings = prompt_utils.get_text_embeddings(
            elevation, azimuth, camera_distances, view_dependent_prompting=False
        )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        B = elevation.shape[0]
        camera_condition_cfg = torch.cat(
            [
                camera_condition.view(B, -1),
                torch.zeros_like(camera_condition.view(B, -1)),
            ],
            dim=0,
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self._sample(
            sample_scheduler=self.scheduler_lora_sample,
            pipe=self.pipe_lora,
            text_embeddings=text_embeddings,
            num_inference_steps=25,
            guidance_scale=self.cfg.guidance_scale_lora,
            class_labels=camera_condition_cfg,
            cross_attention_kwargs={"scale": 1.0},
            generator=generator,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
        class_labels: Optional[Float[Tensor, "B 16"]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_grad_du(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        rgb_BCHW_512: Float[Tensor, "B 3 512 512"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        mask = None,
        **kwargs,
    ):
        batch_size, _, _, _ = latents.shape
        rgb_BCHW_512 = F.interpolate(rgb_BCHW_512, (512, 512), mode="bilinear")
        assert batch_size == 1
        need_diffusion = (
            self.global_step % self.cfg.per_du_step == 0
            and self.global_step > self.cfg.start_du_step
        )
        guidance_out = {}

        if need_diffusion:
            t = torch.randint(
                self.min_step,
                self.max_step,
                [1],
                dtype=torch.long,
                device=self.device,
            )
            self.scheduler.config.num_train_timesteps = t.item()
            self.scheduler.set_timesteps(self.cfg.du_diffusion_steps)

            if mask is not None:
                mask = F.interpolate(mask, (64, 64), mode="bilinear", antialias=True)
            with torch.no_grad():
                # add noise
                noise = torch.randn_like(latents)
                latents = self.scheduler.add_noise(latents, noise, t)  # type: ignore
                for i, timestep in enumerate(self.scheduler.timesteps):
                    # predict the noise residual with unet, NO grad!
                    with torch.no_grad():
                        latent_model_input = torch.cat([latents] * 2)
                        with self.disable_unet_class_embedding(self.unet) as unet:
                            cross_attention_kwargs = (
                                {"scale": 0.0} if self.single_model else None
                            )
                            noise_pred = self.forward_unet(
                                unet,
                                latent_model_input,
                                timestep,
                                encoder_hidden_states=text_embeddings,
                                cross_attention_kwargs=cross_attention_kwargs,
                            )
                    # perform classifier-free guidance
                    noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    if mask is not None:
                        noise_pred = mask * noise_pred + (1 - mask) * noise
                    # get previous sample, continue loop
                    latents = self.scheduler.step(
                        noise_pred, timestep, latents
                    ).prev_sample
            edit_images = self.decode_latents(latents)
            edit_images = F.interpolate(
                edit_images, (512, 512), mode="bilinear"
            ).permute(0, 2, 3, 1)
            gt_rgb = edit_images
            # import cv2
            # import numpy as np
            # mask_temp = mask_BCHW_512.permute(0,2,3,1)
            # # edit_images = edit_images * mask_temp + torch.rand(3)[None, None, None].to(self.device).repeat(*edit_images.shape[:-1],1) * (1 - mask_temp)
            # temp = (edit_images.detach().cpu()[0].numpy() * 255).astype(np.uint8)
            # cv2.imwrite(f".threestudio_cache/pig_sd_noise_500/test_{kwargs.get('name', 'none')}.jpg", temp[:, :, ::-1])

            guidance_out.update(
                {
                    "loss_l1": torch.nn.functional.l1_loss(
                        rgb_BCHW_512, gt_rgb.permute(0, 3, 1, 2), reduction="sum"
                    ),
                    "loss_p": self.perceptual_loss(
                        rgb_BCHW_512.contiguous(),
                        gt_rgb.permute(0, 3, 1, 2).contiguous(),
                    ).sum(),
                    "edit_image": edit_images.detach()
                }
            )

        return guidance_out

    def compute_grad_vsd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B, C, H, W = latents.shape

        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            cross_attention_kwargs = {"scale": 0.0}
            noise_pred_pretrain = self.forward_unet(
                self.train_unet,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings_vd,
                cross_attention_kwargs=cross_attention_kwargs,
            )

            # use view-independent text embeddings in LoRA
            text_embeddings_cond, _ = text_embeddings.chunk(2)
            noise_pred_est = self.forward_unet(
                self.train_unet_lora,
                latent_model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=torch.cat([text_embeddings_cond] * 2),
                # class_labels=torch.cat(
                #     [
                #         camera_condition.view(B, -1),
                #         torch.zeros_like(camera_condition.view(B, -1)),
                #     ],
                #     dim=0,
                # ),
                cross_attention_kwargs={"scale": 0.0},
            )


        # TODO: more general cases
        assert self.scheduler.config.prediction_type == "epsilon"
        if self.scheduler_lora.config.prediction_type == "v_prediction":
            alphas_cumprod = self.scheduler_lora.alphas_cumprod.to(
                device=latents_noisy.device, dtype=latents_noisy.dtype
            )
            alpha_t = alphas_cumprod[t] ** 0.5
            sigma_t = (1 - alphas_cumprod[t]) ** 0.5

            noise_pred_est = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(
                -1, 1, 1, 1
            ) + noise_pred_est * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)

        (
            noise_pred_est_camera,
            noise_pred_est_uncond,
        ) = noise_pred_est.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
            noise_pred_est_camera - noise_pred_est_uncond
        )

        (
            noise_pred_pretrain_text,
            noise_pred_pretrain_uncond,
        ) = noise_pred_pretrain.chunk(2)

        # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
        noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
            noise_pred_pretrain_text - noise_pred_pretrain_uncond
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_pretrain - noise_pred_est)
        return grad

    def compute_grad_vsd_hifa(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings_vd: Float[Tensor, "BB 77 768"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
        mask=None,
    ):
        B, _, DH, DW = latents.shape
        rgb = self.decode_latents(latents)
        self.name = "hifa"

        if mask is not None:
            mask = F.interpolate(mask, (DH, DW), mode="bilinear", antialias=True)
        with torch.no_grad():
            # random timestamp
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler_sample.add_noise(latents, noise, t)
            latents_noisy_lora = self.scheduler_lora_sample.add_noise(latents, noise, t)
            # pred noise

            self.scheduler_sample.config.num_train_timesteps = t.item()
            self.scheduler_sample.set_timesteps(t.item() // 50 + 1)
            self.scheduler_lora_sample.config.num_train_timesteps = t.item()
            self.scheduler_lora_sample.set_timesteps(t.item() // 50 + 1)

            for i, timestep in enumerate(self.scheduler_sample.timesteps):
            # for i, timestep in tqdm(enumerate(self.scheduler.timesteps)):
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                latent_model_input_lora = torch.cat([latents_noisy_lora] * 2, dim=0)

                # print(latent_model_input.shape)
                with self.disable_unet_class_embedding(self.unet) as unet:
                    cross_attention_kwargs = {"scale": 0.0} if self.single_model else None
                    noise_pred_pretrain = self.forward_unet(
                        unet,
                        latent_model_input,
                        timestep,
                        encoder_hidden_states=text_embeddings_vd,
                        cross_attention_kwargs=cross_attention_kwargs,
                    )

                # use view-independent text embeddings in LoRA
                noise_pred_est = self.forward_unet(
                    self.unet_lora,
                    latent_model_input_lora,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                    class_labels=torch.cat(
                        [
                            camera_condition.view(B, -1),
                            torch.zeros_like(camera_condition.view(B, -1)),
                        ],
                        dim=0,
                    ),
                    cross_attention_kwargs={"scale": 1.0},
                )

                (
                    noise_pred_pretrain_text,
                    noise_pred_pretrain_uncond,
                ) = noise_pred_pretrain.chunk(2)

                # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
                noise_pred_pretrain = noise_pred_pretrain_uncond + self.cfg.guidance_scale * (
                    noise_pred_pretrain_text - noise_pred_pretrain_uncond
                )
                if mask is not None:
                    noise_pred_pretrain = mask * noise_pred_pretrain + (1 - mask) * noise

                (
                    noise_pred_est_text,
                    noise_pred_est_uncond,
                ) = noise_pred_est.chunk(2)

                # NOTE: guidance scale definition here is aligned with diffusers, but different from other guidance
                # noise_pred_est = noise_pred_est_uncond + self.cfg.guidance_scale_lora * (
                #     noise_pred_est_text - noise_pred_est_uncond
                # )
                noise_pred_est = noise_pred_est_text
                if mask is not None:
                    noise_pred_est = mask * noise_pred_est + (1 - mask) * noise

                latents_noisy = self.scheduler_sample.step(noise_pred_pretrain, timestep, latents_noisy).prev_sample
                latents_noisy_lora = self.scheduler_lora_sample.step(noise_pred_est, timestep, latents_noisy_lora).prev_sample

                # noise = torch.randn_like(latents)
                # latents_noisy = self.scheduler.step(noise_pred_pretrain, timestep, latents_noisy).prev_sample
                # latents_noisy = mask * latents_noisy + (1-mask) * latents
                # latents_noisy = self.scheduler_sample.add_noise(latents_noisy, noise, timestep)

                # latents_noisy_lora = self.scheduler_lora.step(noise_pred_est, timestep, latents_noisy_lora).prev_sample
                # latents_noisy_lora = mask * latents_noisy_lora + (1-mask) * latents
                # latents_noisy_lora = self.scheduler_lora_sample.add_noise(latents_noisy_lora, noise, timestep)

            hifa_images = self.decode_latents(latents_noisy)
            hifa_lora_images = self.decode_latents(latents_noisy_lora)

            import cv2
            import numpy as np
            if mask is not None:
                print('hifa mask!')
                prefix = 'vsd_mask'
            else:
                prefix = ''
            temp = (hifa_images.permute(0, 2, 3, 1).detach().cpu()[0].numpy() * 255).astype(np.uint8)
            cv2.imwrite(".threestudio_cache/%s%s_test.jpg" % (prefix, self.name), temp[:, :, ::-1])
            temp = (hifa_lora_images.permute(0, 2, 3, 1).detach().cpu()[0].numpy() * 255).astype(np.uint8)
            cv2.imwrite(".threestudio_cache/%s%s_test_lora.jpg" %  (prefix, self.name), temp[:, :, ::-1])

        target = (latents_noisy - latents_noisy_lora + latents).detach()
        # target = latents_noisy.detach()
        targets_rgb = self.decode_latents(target)
        # targets_rgb = (hifa_images - hifa_lora_images + rgb).detach()
        temp = (targets_rgb.permute(0, 2, 3, 1).detach().cpu()[0].numpy() * 255).astype(np.uint8)
        cv2.imwrite(".threestudio_cache/%s_target.jpg" % self.name, temp[:, :, ::-1])

        return w * 0.5 * F.mse_loss(target, latents, reduction='sum')

    def train_lora(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
    ):
        B = latents.shape[0]
        latents = latents.detach().repeat(self.cfg.lora_n_timestamp_samples, 1, 1, 1)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * self.cfg.lora_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler_lora.add_noise(latents, noise, t)
        if self.scheduler_lora.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler_lora.config.prediction_type == "v_prediction":
            target = self.scheduler_lora.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler_lora.config.prediction_type}"
            )
        # use view-independent text embeddings in LoRA
        text_embeddings_cond, _ = text_embeddings.chunk(2)
        if self.cfg.lora_cfg_training and random.random() < 0.1:
            camera_condition = torch.zeros_like(camera_condition)
        noise_pred = self.forward_unet(
            self.train_unet_lora,
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings_cond.repeat(
                self.cfg.lora_n_timestamp_samples, 1, 1
            ),
            # class_labels=camera_condition.view(B, -1).repeat(
            #     self.cfg.lora_n_timestamp_samples, 1
            # ),
            cross_attention_kwargs={"scale": 0.0},
        )
        return F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    def train_pretrain(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        text_embeddings: Float[Tensor, "BB 77 768"],
        camera_condition: Float[Tensor, "B 4 4"],
        sample_new_img=False,
    ):
        B = latents.shape[0]
        if sample_new_img or len(self.cache_frames) == 0:
            latents = latents.detach().repeat(self.cfg.lora_pretrain_n_timestamp_samples, 1, 1, 1)
            images_sample = self._sample(
                pipe=self.pipe_fix,
                sample_scheduler=self.scheduler_sample,
                text_embeddings=text_embeddings,
                num_inference_steps=25,
                guidance_scale=7.5,
                cross_attention_kwargs = {"scale": 0.0},
                latents_inp=latents,
            ).permute(0,3,1,2)
            from torchvision.utils import save_image
            save_image(images_sample, f".threestudio_cache/test_sample.jpg")
            self.cache_frames.append(images_sample)

            self.pipe.unet = self.train_unet
            pretrain_images_sample = self._sample(
                pipe=self.pipe,
                sample_scheduler=self.scheduler_sample,
                text_embeddings=text_embeddings,
                num_inference_steps=25,
                guidance_scale=1.0,
                cross_attention_kwargs = {"scale": 0.0},
                latents_inp=latents,
            ).permute(0,3,1,2)
            save_image(pretrain_images_sample, f".threestudio_cache/test_pretrain.jpg")
        if len(self.cache_frames) > 10:
            self.cache_frames.pop(0)
        random_idx = torch.randint(0, len(self.cache_frames), [1]).item()
        images_sample = self.cache_frames[random_idx]

        with torch.no_grad():
            latents_sample = self.get_latents(images_sample, rgb_as_latents=False)

        t = torch.randint(
            int(self.num_train_timesteps * 0.0),
            int(self.num_train_timesteps * 1.0),
            [B * self.cfg.lora_pretrain_n_timestamp_samples],
            dtype=torch.long,
            device=self.device,
        )

        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents_sample, noise, t)
        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents_sample, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )
        # FIXME: use view-independent or dependent embeddings?
        text_embeddings_cond, _ = text_embeddings.chunk(2)
        if self.cfg.lora_pretrain_cfg_training and random.random() < 0.1:
            text_embeddings_cond = torch.zeros_like(text_embeddings_cond)
        noise_pred = self.forward_unet(
            self.train_unet,
            noisy_latents,
            t,
            encoder_hidden_states=text_embeddings_cond.repeat(
                self.cfg.lora_pretrain_n_timestamp_samples, 1, 1
            )
        )
        loss_pretrain = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        return loss_pretrain

    def get_latents(
        self, rgb_BCHW: Float[Tensor, "B C H W"], rgb_as_latents=False
    ) -> Float[Tensor, "B 4 64 64"]:
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            rgb_BCHW_512 = F.interpolate(
                rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
            )
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)
        return latents

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        mvp_mtx: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents=False,
        mask: Float[Tensor, "B H W 1"] = None,
        lora_prompt_utils = None,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents = self.get_latents(rgb_BCHW, rgb_as_latents=rgb_as_latents)

        if mask is not None: mask = mask.permute(0, 3, 1, 2)

        # view-dependent text embeddings
        text_embeddings_vd = prompt_utils.get_text_embeddings(
            elevation,
            azimuth,
            camera_distances,
            view_dependent_prompting=self.cfg.view_dependent_prompting,
        )
        if lora_prompt_utils is not None:
            # input text embeddings, view-independent
            text_embeddings = lora_prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, view_dependent_prompting=False
            )
        else:
            # input text embeddings, view-independent
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, view_dependent_prompting=False
            )

        if self.cfg.camera_condition_type == "extrinsics":
            camera_condition = c2w
        elif self.cfg.camera_condition_type == "mvp":
            camera_condition = mvp_mtx
        else:
            raise ValueError(
                f"Unknown camera_condition_type {self.cfg.camera_condition_type}"
            )

        do_update_pretrain = (self.cfg.only_pretrain_step > 0) and (
            (self.global_step % self.cfg.only_pretrain_step) < (self.cfg.only_pretrain_step // 5)
        )

        guidance_out = {}
        if do_update_pretrain:
            sample_new_img = self.global_step % self.cfg.per_update_pretrain_step == 0
            loss_pretrain = self.train_pretrain(latents, text_embeddings_vd, camera_condition, sample_new_img=sample_new_img)
            guidance_out.update({
                "loss_pretrain": loss_pretrain,
                "min_step": self.min_step,
                "max_step": self.max_step,
            })
            return guidance_out

        grad = self.compute_grad_vsd(
            latents, text_embeddings_vd, text_embeddings, camera_condition
        )

        grad = torch.nan_to_num(grad)
        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

        # reparameterization trick
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        target = (latents - grad).detach()
        loss_vsd = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size

        loss_lora = self.train_lora(latents, text_embeddings, camera_condition)

        guidance_out.update({
            "loss_sd": loss_vsd,
            "loss_lora": loss_lora,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        })

        if self.cfg.use_du:
            du_out = self.compute_grad_du(latents, rgb_BCHW, text_embeddings_vd, mask=mask)
            guidance_out.update(du_out)

        return guidance_out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)
        self.global_step = global_step
        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )

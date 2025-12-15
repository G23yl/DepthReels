import gc
import math
import os
import shutil
import wandb

import torch
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from loguru import logger
from safetensors.torch import load_file
from torch.amp.autocast_mode import autocast
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

from skyreels_v2.modules import get_image_encoder, get_text_encoder, get_vae
from skyreels_v2.modules.transformer import WanModel
from skyreels_v2.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
from skyreels_v2.trainer.robodataset import RoboDepth

PRECISION2TYPE = {
    "bf16": torch.bfloat16,
}

class Trainer:
    def __init__(self, model_path: str, config):
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        if self.accelerator.mixed_precision in PRECISION2TYPE:
            self.dtype = PRECISION2TYPE[self.accelerator.mixed_precision]
        else:
            raise ValueError(f"Unknown dtype, {self.accelerator.mixed_precision}")
        self.device = self.accelerator.device
        self.config = config
        if self.accelerator.is_local_main_process:
            logger.info(f"[Config] {config}")

        vae_model_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        self.vae = get_vae(vae_model_path, "cpu", weight_dtype=torch.bfloat16)
        self.text_encoder = get_text_encoder(model_path, "cpu")
        self.clip = get_image_encoder(model_path, "cpu")
        self.transformer = self.get_transformer(model_path)
        if config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        if self.accelerator.is_local_main_process:
            logger.info("[Model] Loaded vae | text_encoder | clip | transformer")

        self.scheduler = self.get_scheduler()
        if self.accelerator.is_local_main_process:
            logger.info(f"[Scheduler] Train timesteps: {self.scheduler.timesteps}")

        dataset = RoboDepth(data_root=config.data_root, max_num_frames=49)
        self.dataloader = DataLoader(dataset, batch_size=1)
        if self.accelerator.is_local_main_process:
            logger.info(f"[Dataset] Size: {len(dataset)}")

        self.optimizer = AdamW(
            self.transformer.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        self.lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=config.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=config.max_training_steps
            * self.accelerator.num_processes,
            num_cycles=config.lr_num_cycles,
        )

        self.transformer, self.dataloader, self.optimizer, self.lr_scheduler = (
            self.accelerator.prepare(
                self.transformer, self.dataloader, self.optimizer, self.lr_scheduler
            )
        )

        # recalculate update times in an epoch
        num_update_steps_per_epoch = math.ceil(
            len(self.dataloader) / config.gradient_accumulation_steps
        )
        # real training epoch
        self.num_train_epoch = math.ceil(
            config.max_training_steps / num_update_steps_per_epoch
        )

        self.global_step = 0

        if self.accelerator.is_local_main_process and not config.disable_wandb:
            wandb.login(key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                mode="online",
                name="wan_depth",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
            )

    def get_transformer(self, model_path: str):
        cfg = WanModel.load_config(model_path, local_files_only=True)
        assert isinstance(cfg, dict), "Model config type is not dict"
        transformer = WanModel.from_config(cfg)

        assert isinstance(transformer, WanModel), "Loading transformer failed"

        for file in os.listdir(model_path):
            if file.endswith(".safetensors"):
                file_path = os.path.join(model_path, file)
                state_dict = load_file(file_path)
                transformer.load_state_dict(state_dict, strict=False)
                del state_dict
                gc.collect()
                torch.cuda.empty_cache()

        transformer.requires_grad_(True)
        transformer.to(dtype=torch.bfloat16)
        return transformer

    def get_scheduler(self):
        scheduler = FlowUniPCMultistepScheduler()
        scheduler.set_timesteps(
            self.config.num_train_steps, shift=self.config.time_shift
        )
        return scheduler

    def train(self):
        if self.accelerator.is_local_main_process:
            logger.info("Start training!")

        progress_bar = tqdm(
            range(self.config.max_training_steps),
            desc="Steps",
            disable=not self.accelerator.is_local_main_process,
        )

        for _ in range(0, self.num_train_epoch):
            self.transformer.train()
            for _, batch in enumerate(self.dataloader):
                with self.accelerator.accumulate(self.transformer):
                    prompts = batch["prompts"]
                    video = batch["frames"]
                    depth = batch["depth"]
                    first_img = batch["first_image"]
                    first_depth = batch["first_depth"]

                    context = self.text_encoder.encode(prompts)
                    clip_context = self.clip.encode_video(first_img)

                    B, C, F, H, W = video.shape

                    # img condition
                    padding_img = torch.zeros(B, C, F - 1, H, W)
                    img_cond = torch.cat([first_img, padding_img], dim=2)
                    img_cond = self.vae.encode(img_cond)
                    mask = torch.ones_like(img_cond)
                    mask[:, :, 1:] = 0
                    img_y = torch.cat([mask[:, :4], img_cond], dim=1)
                    # depth condition
                    padding_depth = torch.zeros(B, C, F - 1, H, W)
                    depth_cond = torch.cat([first_depth, padding_depth], dim=2)
                    depth_cond = self.vae.encode(depth_cond)
                    mask = torch.ones_like(depth_cond)
                    mask[:, :, 1:] = 0
                    depth_y = torch.cat([mask[:, :4], depth_cond], dim=1)

                    # cat img_y and depth_y along frame dim -- [B, 16+4, 13*2, h, w]
                    y = torch.cat([img_y, depth_y], dim=2)

                    # encode video and depth to get latent
                    latent_video = self.vae.encode(video)
                    latent_depth = self.vae.encode(depth)
                    # concat them -- [B, 16, 13*2, h, w]
                    latent = torch.cat([latent_video, latent_depth], dim=2)

                    # noise ready
                    noise = torch.randn_like(latent)
                    timestep_idx = torch.randint(
                        0,
                        self.scheduler.config.num_train_timesteps,
                        dtype=torch.int64,
                    )
                    timesteps = self.scheduler.timesteps[timestep_idx].expand(B)
                    noisy_latent = self.scheduler.add_noise(
                        latent, noise, timesteps
                    )

                    transformer_input_kwargs = {
                        "context": context,
                        "clip_fea": clip_context,
                        "y": y,
                    }

                    noise_pred = self.transformer(
                        noisy_latent, t=timesteps, **transformer_input_kwargs
                    )
                    # x0 sample
                    model_pred = self.scheduler.convert_model_output(
                        noise_pred, sample=noisy_latent
                    )
                    # clean sample
                    target = latent

                    rgb_pred, depth_pred = model_pred.chunk(2, dim=2)
                    rgb_target, depth_target = target.chunk(2, dim=2)

                    def loss_fn(pred, target):
                        return torch.mean(
                            ((pred - target) ** 2).reshape(B, -1),
                            dim=1,
                        ).mean()

                    rgb_loss = loss_fn(rgb_pred, rgb_target)
                    depth_loss = loss_fn(depth_pred, depth_target)
                    loss = rgb_loss + depth_loss

                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.transformer.parameters(), self.config.max_grad_norm
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.lr_scheduler.step()

                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1

                    # wait for all processes
                    self.accelerator.wait_for_everyone()

                    if self.accelerator.is_local_main_process:
                        if self.global_step % self.config.save_steps == 0:
                            # guarantee only 5 files, delete the oldest ckpt
                            folders = os.listdir(self.config.save_dir)
                            if len(folders) >= 5:
                                folders = sorted(folders)
                                shutil.rmtree(
                                    os.path.join(self.config.save_dir, folders[0])
                                )
                            out_dir = os.path.join(
                                self.config.save_dir, f"step_{self.global_step:08d}"
                            )
                            if not os.path.exists(out_dir):
                                os.makedirs(out_dir, exist_ok=True)
                            logger.info(f"Saving model to {out_dir}")
                            self.accelerator.save_model(
                                self.transformer,
                                out_dir,
                            )

                last_lr = self.lr_scheduler.get_last_lr()[0]

                logs = {"loss": loss.detach().item(), "lr": last_lr}
                progress_bar.set_postfix(**logs)

                loss = self.accelerator.reduce(loss, reduction="mean")
                if (
                    self.accelerator.is_local_main_process
                    and not self.config.disable_wandb
                ):
                    log = {
                        "total": loss.detach().item(),
                        "rgb": rgb_loss.detach().item(),
                        "depth": depth_loss.detach().item(),
                        "lr": last_lr,
                    }
                    wandb.log(log, step=self.global_step)

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            out_dir = os.path.join(self.config.save_dir, "final")
            logger.info(f"Saving model to {out_dir}")
            self.accelerator.save_model(
                self.transformer,
                out_dir,
            )

        if self.accelerator.is_local_main_process:
            logger.info("Finish training and wait for cleanup...")
        self.accelerator.end_training()

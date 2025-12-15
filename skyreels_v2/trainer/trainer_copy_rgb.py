import gc
import math
import os
import shutil

import torch
import torch.distributed as dist
import wandb

from diffusers.optimization import get_scheduler
from loguru import logger
from omegaconf import OmegaConf
from torch.amp.autocast_mode import autocast
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import torch.nn.functional as Func

from skyreels_v2.modules import (
    get_image_encoder,
    get_text_encoder,
    get_vae,
    get_transformer,
)
from skyreels_v2.scheduler.fm_solvers_unipc import FlowUniPCMultistepScheduler
from skyreels_v2.trainer.distributed import (
    cleanup_distributed,
    fsdp_state_dict,
    fsdp_wrap,
    launch_distributed_job,
)
from skyreels_v2.trainer.robodataset import RoboDepth


class Trainer:
    def __init__(self, model_path: str, config):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        launch_distributed_job()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.config = config
        self.is_main_process = dist.get_rank() == 0
        if self.is_main_process:
            logger.info(f"[Config] dtype: {self.dtype}")

        # load model and wrap with FSDP
        vae_model_path = os.path.join(model_path, "Wan2.1_VAE.pth")
        self.vae = get_vae(vae_model_path, self.device, weight_dtype=self.dtype)
        self.text_encoder = get_text_encoder(model_path, self.device)
        self.clip = get_image_encoder(model_path, self.device, weight_dtype=self.dtype)
        self.transformer = get_transformer(
            model_path, self.device, weight_dtype=self.dtype
        ).requires_grad_(True)
        if config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.transformer = fsdp_wrap(
            self.transformer,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fsdp_wrap_strategy,
        )

        if self.is_main_process:
            logger.info("[Model] Loaded vae | text_encoder | clip | transformer")

        self.scheduler = self.get_scheduler()

        dataset = RoboDepth(data_root=config.data_root, max_num_frames=49)
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        self.dataloader = DataLoader(
            dataset, batch_size=1, sampler=sampler, num_workers=8
        )
        logger.info(f"[Dataloader] Size: {len(self.dataloader)}")

        if len(self.dataloader) == 1:
            logger.info(f"[Dataloader] {self.dataloader.dataset[0]['path']}")

        self.optimizer = AdamW(
            self.transformer.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        self.lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=config.lr_warmup_steps * dist.get_world_size(),
            num_training_steps=config.max_training_steps * dist.get_world_size(),
            num_cycles=config.lr_num_cycles,
        )

        # real training epoch
        self.num_train_epoch = math.ceil(
            config.max_training_steps / len(self.dataloader)
        )

        self.global_step = 0

        if self.is_main_process and not config.disable_wandb:
            wandb.login(key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                mode="online",
                name="wan_depth",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
            )

    def get_scheduler(self):
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.config.num_train_steps,
            shift=self.config.time_shift
        )
        scheduler.timesteps = scheduler.timesteps.to(self.device)
        return scheduler

    def train(self):
        self.transformer.train()
        if self.is_main_process:
            logger.info("Start training!")
            logger.info(
                f"[Train] max_training_steps: {self.config.max_training_steps}, num_train_epoch: {self.num_train_epoch}"
            )

        progress_bar = tqdm(
            range(self.config.max_training_steps),
            desc="Steps",
            disable=not self.is_main_process,
        )

        with autocast(
            device_type=self.transformer.device.type, dtype=self.transformer.dtype
        ):
            for _ in range(0, self.num_train_epoch):
                for batch in self.dataloader:
                    prompts = batch["prompts"]
                    video = batch["frames"].to(device=self.device, dtype=self.dtype)
                    first_img = batch["first_image"].to(
                        device=self.device, dtype=self.dtype
                    )

                    context = self.text_encoder.encode(prompts).to(device=self.device)

                    clip_context = self.clip.encode_video(first_img).to(
                        device=self.device, dtype=self.dtype
                    )

                    B, C, F, H, W = video.shape

                    # img condition
                    padding_img = torch.zeros(B, C, F - 1, H, W, device=self.device)
                    img_cond = torch.cat([first_img, padding_img], dim=2)
                    img_cond = self.vae.encode(img_cond)
                    mask = torch.ones_like(img_cond, device=self.device)
                    mask[:, :, 1:] = 0
                    img_y = torch.cat([mask[:, :4], img_cond], dim=1)
                    y = img_y.to(device=self.device, dtype=self.dtype)

                    # encode video and depth to get latent
                    latent_video = self.vae.encode(video)
                    latent = latent_video.to(device=self.device, dtype=self.dtype)

                    # noise ready
                    noise = torch.randn_like(
                        latent, device=self.device, dtype=self.dtype
                    )
                    timestep_idx = torch.randint(
                        0,
                        self.config.num_train_steps,
                        (B,),
                        dtype=torch.int64,
                    )
                    timestep = self.scheduler.timesteps[timestep_idx].to(
                        device=self.device
                    )

                    noisy_latent = self.scheduler.add_noise(latent, noise, timestep)

                    transformer_input_kwargs = {
                        "context": context,
                        "clip_fea": clip_context,
                        "y": y,
                    }

                    gc.collect()
                    torch.cuda.empty_cache()

                    flow_pred = self.transformer(
                        noisy_latent, t=timestep, **transformer_input_kwargs
                    )

                    model_pred = flow_pred
                    target = noise - latent

                    def loss_fn(pred, target):
                        return Func.mse_loss(pred, target)

                    loss = loss_fn(model_pred, target)

                    loss.backward()

                    self.transformer.clip_grad_norm_(self.config.max_grad_norm)

                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()

                    progress_bar.update(1)
                    self.global_step += 1

                    if self.global_step % self.config.save_steps == 0:
                        dist.barrier()
                        state_dict = fsdp_state_dict(self.transformer)

                        if self.is_main_process:
                            # guarantee only `config.max_save_ckpt` files, delete the oldest ckpt
                            folders = os.listdir(self.config.save_dir)
                            if len(folders) >= self.config.max_save_ckpt:
                                folders = sorted(folders)
                                logger.info(
                                    f"Deleting the oldest checkpoint: {folders[0]}"
                                )
                                shutil.rmtree(
                                    os.path.join(self.config.save_dir, folders[0])
                                )
                            out_dir = os.path.join(
                                self.config.save_dir, f"step_{self.global_step:08d}"
                            )
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, "model.pt")
                            logger.info(f"Saving model to {out_path}")
                            torch.save(state_dict, out_path)

                    last_lr = self.lr_scheduler.get_last_lr()[0]

                    logs = {"loss": loss.detach().item(), "lr": last_lr}
                    progress_bar.set_postfix(**logs)

                    if self.is_main_process and not self.config.disable_wandb:
                        log = {
                            "total": loss.detach().item(),
                            "lr": last_lr,
                        }
                        wandb.log(log, step=self.global_step)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if self.global_step >= self.config.max_training_steps:
                        break

        dist.barrier()
        state_dict = fsdp_state_dict(self.transformer)
        if self.is_main_process:
            out_dir = os.path.join(self.config.save_dir, "final")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "model.pt")
            logger.info(f"Saving model to {out_path}")
            torch.save(state_dict, out_path)

        if self.is_main_process:
            logger.info("Finish training and wait for cleanup...")
        cleanup_distributed()

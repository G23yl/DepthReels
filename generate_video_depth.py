import argparse
import gc
import json
import os
import random
import time
import numpy as np
import cv2

import torch
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.loading_utils import load_image
from loguru import logger
from torch.amp.autocast_mode import autocast
from torchvision.transforms import transforms

from skyreels_v2.modules import download_model
from skyreels_v2.pipelines import PromptEnhancer, Image2VideoDepthPipeline, Image2VideoDepth2Pipeline

from skyreels_v2.trainer.robodataset import RoboDepth

transform_list = transforms.Compose(
    [
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
)


def load_depth(path):
    try:
        depth_array = np.load(path)["arr_0"].astype(np.float32)
    except Exception:
        depth_array = None

    if depth_array is None:
        try:
            depth_array = np.load(path).astype(np.float32)
        except Exception:
            depth_array = None

    assert depth_array is not None, f"Failed to load depth from {path}."

    depth_img = (depth_array - depth_array.min()) / (
        depth_array.max() - depth_array.min() + 1e-8
    )

    depth_img *= 255.0
    # channel = 1  hw
    if len(depth_img.shape) == 2:
        depth_img = np.stack([depth_img] * 3, axis=-1)  # [H, W, 3]
    else:
        assert depth_img.shape[-1] == 3, "depth has 3 dims but last dim is not 3"

    return depth_img


def crop_and_resize_frames(frames, target_size, interpolation="bilinear"):
    # [F, H, W, C]
    target_height, target_width = target_size
    original_height, original_width = frames[0].shape[:2]
    if original_height == target_height and original_width == target_width:
        return [frame for frame in frames]

    # ==== interpolation method ====
    if interpolation == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif interpolation == "nearest":
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR
        logger.warning(
            f"Unsupported interpolation: {interpolation}. Using bilinear instead."
        )

    processed_frames = []
    for frame in frames:
        original_height, original_width = frame.shape[:2]
        aspect_ratio_target = target_width / target_height
        aspect_ratio_original = original_width / original_height

        if aspect_ratio_original > aspect_ratio_target:
            new_width = int(aspect_ratio_target * original_height)
            start_x = (original_width - new_width) // 2
            cropped_frame = frame[:, start_x : start_x + new_width]
        else:
            new_height = int(original_width / aspect_ratio_target)
            start_y = (original_height - new_height) // 2
            cropped_frame = frame[start_y : start_y + new_height, :]
        resized_frame = cv2.resize(
            cropped_frame, (target_width, target_height), interpolation=interpolation
        )
        processed_frames.append(resized_frame)

    return processed_frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="video_out")
    parser.add_argument(
        "--model_id",
        type=str,
        default="Skywork/SkyReels-V2-T2V-14B-540P",
        help="If using distilled model, just set guidance_scale=1.0 and don't set teacache",
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--resolution", type=str, choices=["480P", "540P", "720P"])
    parser.add_argument("--num_frames", type=int, default=98)
    parser.add_argument("--image", type=str)
    parser.add_argument("--depth", type=str)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--use_usp", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        default="A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.",
    )
    parser.add_argument("--prompt_enhancer", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument(
        "--teacache_thresh",
        type=float,
        default=0.2,
        help="Higher speedup will cause to worse quality -- 0.1 for 2.0x speedup -- 0.2 for 3.0x speedup",
    )
    parser.add_argument(
        "--use_ret_steps",
        action="store_true",
        help="Using Retention Steps will result in faster generation speed and better generation quality.",
    )
    args = parser.parse_args()

    if args.model_path != "":
        args.model_id = args.model_path
    else:
        args.model_id = download_model(args.model_id)
    logger.info(f"Model path: {args.model_id}")

    assert (args.use_usp and args.seed is not None) or (not args.use_usp), (
        "usp mode need seed"
    )
    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    if args.resolution == "480P":
        height = 480
        width = 832
    elif args.resolution == "540P":
        height = 544
        width = 960
    elif args.resolution == "720P":
        height = 720
        width = 1280
    else:
        raise ValueError(f"Invalid resolution: {args.resolution}")

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    local_rank = 0
    if args.use_usp:
        assert not args.prompt_enhancer, (
            "`--prompt_enhancer` is not allowed if using `--use_usp`. We recommend running the skyreels_v2_infer/pipelines/prompt_enhancer.py script first to generate enhanced prompt before enabling the `--use_usp` parameter."
        )
        import torch.distributed as dist
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())
        device = "cuda"

        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )

    prompt_input = args.prompt
    if args.prompt_enhancer and args.image is None:
        logger.info("init prompt enhancer")
        prompt_enhancer = PromptEnhancer()
        prompt_input = prompt_enhancer(prompt_input)
        logger.info(f"enhanced prompt: {prompt_input}")
        del prompt_enhancer
        gc.collect()
        torch.cuda.empty_cache()

    # assert "I2V" in args.model_id, f"check model_id:{args.model_id}"
    logger.info("Initializing img2video pipeline")
    pipe = Image2VideoDepthPipeline(
        model_path=args.model_id,
        use_usp=args.use_usp,
        offload=args.offload,
    )
    # first_image = load_image(args.image).convert("RGB")  # hwc
    # depth_img = load_depth(args.depth)  # [-1, 1] hwc
    # image_width, image_height = first_image.size
    # # height and width exchange
    # if image_height > image_width:
    #     height, width = width, height
    # depth_img = [depth_img]
    # depth_img = crop_and_resize_frames(depth_img, (height, width))[0]  # [H, W, 3]

    # depth_img = transform_list(torch.from_numpy(depth_img).permute(2, 0, 1).float())
    # args.depth = depth_img

    # first_image = crop_and_resize_frames(
    #     [np.array(first_image).astype(np.float32)], (height, width)
    # )[0]
    # first_image = transform_list(torch.from_numpy(first_image).permute(2, 0, 1).float())
    # args.image = first_image

    sample = RoboDepth(data_root="/mnt/zhouxin-mnt", max_num_frames=49)[20]
    args.image = sample["first_image"].squeeze(1)
    args.depth = sample["first_depth"].squeeze(1)
    prompt_input = sample["prompts"]

    # logger.debug(f"depth: {args.depth.shape}, min/max: {args.depth.min()}/{args.depth.max()}\nimage: {args.image.shape}, min/max: {args.image.min()}/{args.image.max()}")

    if args.teacache:
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=args.inference_steps,
            teacache_thresh=args.teacache_thresh,
            use_ret_steps=args.use_ret_steps,
            ckpt_dir=args.model_id,
        )

    kwargs = {
        "prompt": prompt_input,
        "negative_prompt": negative_prompt,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "generator": torch.Generator(device="cuda").manual_seed(args.seed),
        "width": width,
        "height": height,
    }

    kwargs["image"] = args.image
    kwargs["depth"] = args.depth

    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)

    with autocast(device_type="cuda", dtype=pipe.transformer.dtype), torch.no_grad():
        out_config = {}
        exclude_keys = ["generator", "image", "depth"]
        for k, v in kwargs.items():
            if k not in exclude_keys:
                out_config[k] = v
        out_config = json.dumps(out_config, indent=4)
        logger.info(f"Config: {out_config}")

        video_frames = pipe(**kwargs)

    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = (
            f"{prompt_input[:100].replace('/', '')}_{args.seed}_{current_time}.mp4"
        )
        video_output_path = os.path.join(save_dir, video_out_file)
        logger.info(
            f"Saving video to {video_output_path}"
        )
        export_to_video(video_frames, video_output_path, fps=args.fps)

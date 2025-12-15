import argparse
import gc
import json
import os
import random
import time

import torch
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.loading_utils import load_image
from loguru import logger
from torch.amp.autocast_mode import autocast

from skyreels_v2.modules import download_model
from skyreels_v2.pipelines import (
    Image2VideoPipeline,
    PromptEnhancer,
    Text2VideoPipeline,
    resizecrop,
)

MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
}


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
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
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

    image = load_image(args.image).convert("RGB") if args.image else None
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

    if image is None:
        assert "T2V" in args.model_id, f"check model_id:{args.model_id}"
        logger.info("Initializing text2video pipeline")
        pipe = Text2VideoPipeline(
            model_path=args.model_id,
            dit_path=args.model_id,
            use_usp=args.use_usp,
            offload=args.offload,
        )
    else:
        assert "I2V" in args.model_id, f"check model_id:{args.model_id}"
        logger.info("Initializing img2video pipeline")
        pipe = Image2VideoPipeline(
            model_path=args.model_id,
            dit_path=args.model_id,
            use_usp=args.use_usp,
            offload=args.offload,
        )
        args.image = load_image(args.image)
        image_width, image_height = args.image.size
        # height and width exchange
        if image_height > image_width:
            height, width = width, height
        args.image = resizecrop(args.image, height, width)

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

    if image is not None:
        kwargs["image"] = args.image.convert("RGB")

    save_dir = os.path.join("result", args.outdir)
    os.makedirs(save_dir, exist_ok=True)

    with autocast(device_type="cuda", dtype=pipe.transformer.dtype), torch.no_grad():
        out_config = {}
        exclude_keys = ["generator", "image"]
        for k, v in kwargs.items():
            if k not in exclude_keys:
                out_config[k] = v
        out_config = json.dumps(out_config, indent=4)
        logger.info(f"Config: {out_config}")

        video_frames = pipe(**kwargs)

    if local_rank == 0:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = (
            f"{args.prompt[:20].replace('/', '')}_{args.seed}_{current_time}.mp4"
        )
        output_path = os.path.join(save_dir, video_out_file)
        logger.info(f"Saving video to {output_path}")
        export_to_video(video_frames, output_path, fps=args.fps)

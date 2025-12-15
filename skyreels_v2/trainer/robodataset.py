import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import jsonlines
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import logging

import decord
import torch
from diffusers.utils.export_utils import export_to_video
from torch.utils.data import Dataset
from torchvision import transforms

decord.bridge.set_bridge("torch")

logging.basicConfig(level=logging.INFO)

DATASET2ROBOT = {
    "fractal20220817_data": "google robot",
    "bridge": "Trossen WidowX 250 robot arm",
    "ssv2": "human hand",
    "rlbench": "Franka Emika Panda",
}
DATASET2RES = {
    "fractal20220817_data": (256, 320),
    # "fractal20220817_data_superres": (512, 640),
    "bridge": (544, 960),
    "rlbench": (512, 512),
    # common resolutions
    # "480p": (480, 854),
    # "720p": (720, 1280),
}
HEIGHT_BUCKETS = [240, 256, 480, 720]
WIDTH_BUCKETS = [320, 426, 640, 854, 1280]
FRAME_BUCKETS = [9, 49, 100]


def crop_and_resize_frames(frames, target_size, interpolation="bilinear"):
    # frames: [F, H, W, C]
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
        logging.warning(
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


class RoboDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 81,
        id_token: Optional[str] = None,
        height_buckets: List[int] | None = None,
        width_buckets: List[int] | None = None,
        frame_buckets: List[int] | None = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w)
            for h in self.height_buckets
            for w in self.width_buckets
            for f in self.frame_buckets
        ]

        self._init_transforms()
        self._load_samples()

    def _init_transforms(self):
        """Initialize video transforms based on class requirements"""
        transform_list = [
            transforms.Lambda(self.identity_transform),
            transforms.Lambda(self.scale_transform),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]

        if self.random_flip:
            transform_list.insert(0, transforms.RandomHorizontalFlip(self.random_flip))

        self.video_transforms = transforms.Compose(transform_list)

    def _load_samples(self):
        """Load samples from dataset file or local paths"""
        if self.dataset_file is None or not Path(self.dataset_file).exists():
            self.samples, test_samples = self._load_openx_dataset_from_local_path(
                "bridge"
            )
            logging.info(
                f"Loaded {len(self.samples)} train and {len(test_samples)} test samples from Bridge dataset."
            )

            # Save samples to dataset file
            random.shuffle(self.samples)
            with open(self.dataset_file, "w") as f:
                json.dump(self.samples, f)
            with open(self.dataset_file.replace(".json", "_test.json"), "w") as f:
                json.dump(test_samples, f)
        else:
            with open(self.dataset_file, "r") as f:
                self.samples = json.load(f)
            self._get_rlbench_instructions()

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for i in range(5):
            try:
                return self.getitem(index)
            except Exception as e:
                logging.error(f"Error loading sample {self.samples[index][1]}: {e}")
                index = random.randint(0, len(self.samples) - 1)
        return self.getitem(index)

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Special logic for bucket sampler
            return index

        if self.load_tensors:
            raise NotImplementedError("Loading tensors is not supported.")

        sample = self.samples[index]
        image, video = self._preprocess_video(Path(sample[1]))
        instruction = self.get_instruction(index)

        return {
            "prompt": self.id_token + instruction,
            "image": image,
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
        }

    def _train_test_split(
        self, samples: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        if len(samples) > 4000:
            test_size = 200
        else:
            test_size = max(1, int(len(samples) * 0.05))

        indices = list(range(len(samples)))
        random.shuffle(indices)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        test_samples = [samples[i] for i in test_indices]
        train_samples = [samples[i] for i in train_indices]

        return train_samples, test_samples

    def _load_openx_dataset_from_local_path(self, dataname: str) -> List[List[str]]:
        samples = []
        for subdir in (self.data_root / dataname).iterdir():
            if subdir.is_dir():
                # skip when no depth data
                depth_dir = subdir.joinpath("depth", "npz", "depth.npz")
                if not depth_dir.exists():
                    continue

                rgb_dir = subdir.joinpath("video", "rgb.mp4")
                rgb_valid = rgb_dir.exists()
                if not rgb_dir.exists():
                    rgb_dir = subdir.joinpath("image", "rgb")
                    rgb_valid = rgb_dir.exists() and any(rgb_dir.glob("*.png"))
                if rgb_valid:
                    # Load prompt from instruction.txt if available
                    instruction_file = subdir.joinpath("instruction.txt")
                    if instruction_file.is_file():
                        instruction = instruction_file.read_text().strip()
                    else:
                        instruction = "null"
                    # path str
                    samples.append([instruction, str(rgb_dir)])

        # train_samples, test_samples = self._train_test_split(samples)
        return samples

    def _load_ssv2_dataset_from_local_path(self) -> Tuple[List[str], List[Path]]:
        labels_file = Path("data/ssv2/labels/train.json")
        video_root = Path("data/ssv2/20bn-something-something-v2")
        with labels_file.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        samples = []
        for entry in labels:
            video_id = entry.get("id")
            label = entry.get("label", "null")
            video_path = video_root / f"{video_id}.webm"
            samples.append([label, str(video_path)])

        train_samples, test_samples = self._train_test_split(samples)
        return train_samples, test_samples

    def _get_rlbench_instructions(self) -> List[str]:
        self.rlbench_instructions = {}
        taskvar_json = Path("data/rlbench/taskvar_instructions.jsonl")
        if not taskvar_json.exists():
            logging.warning(f"Taskvar json {taskvar_json} does not exist.")
            return
        with jsonlines.open(taskvar_json, "r") as reader:
            for obj in reader:
                task = obj["task"]
                self.rlbench_instructions.setdefault(task, obj["variations"]["0"])

    def _load_rlbench_dataset_from_local_path(self) -> List[List[str]]:
        rlbench_path = Path("data/rlbench/train_dataset/microsteps/seed100")

        self._get_rlbench_instructions()

        samples = [
            [task_dir.name, str(rgb_path)]
            for task_dir in rlbench_path.iterdir()
            for episode_dir in task_dir.glob("variation0/episodes/*")
            for rgb_path in episode_dir.glob("video/*rgb.mp4")
        ]
        # find which path don't have video
        for task_dir in rlbench_path.iterdir():
            for episode_dir in task_dir.glob("variation0/episodes/*"):
                rgb_path = episode_dir.glob("video/*rgb.mp4")
                if not rgb_path:
                    print(f"Missing video: {episode_dir}")
        train_samples, test_samples = self._train_test_split(samples)
        return train_samples, test_samples

    def _adjust_num_frames(self, frames, target_num_frames=None):
        if target_num_frames is None:
            target_num_frames = self.max_num_frames
        frame_count = len(frames)
        if frame_count < target_num_frames:
            extra = target_num_frames - frame_count
            if isinstance(frames, list):
                frames.extend([frames[-1]] * extra)
            elif isinstance(frames, torch.Tensor):
                frame_to_add = [frames[-1]] * extra
                frames = [f for f in frames] + frame_to_add
        elif frame_count > target_num_frames:
            indices = np.linspace(0, frame_count - 1, target_num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        return frames

    def get_instruction(self, index: int) -> str:
        # if random.random() < 0.05:
        #     instruction = ""
        # else:
        sample = self.samples[index]
        instruction = sample[0].lower()
        path = sample[1]

        if "rlbench" in str(path):
            task_name = path.split("/")[5]
            instruction = (
                random.choice(self.rlbench_instructions[task_name])
                + f" {DATASET2ROBOT['rlbench']}"
            )
        elif "fractal20220817_data" in str(path):
            instruction += f" {DATASET2ROBOT['fractal20220817_data']}"
        elif "bridge" in str(path):
            instruction += f" {DATASET2ROBOT['bridge']}"
        elif "ssv2" in str(path):
            instruction += f" {DATASET2ROBOT['ssv2']}"
        else:
            raise ValueError(f"Unknown dataset for path: {path}")

        return instruction

    def _read_rgb_data(self, path: Path) -> torch.Tensor:
        if path.is_dir():
            frames = self._read_video_from_dir(path, adjust_num_frames=False)
        elif path.suffix == ".webm" or path.suffix == ".mp4":
            frames = self._read_video_from_webm(path, adjust_num_frames=False)
        else:
            raise ValueError(f"Unsupported video format: {path}")
        return frames

    def _preprocess_video(
        self, path: Path
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Loads a single video from either:
        - A directory of RGB frames, or
        - A single .webm video file.

        Returns:
            image: the first frame as an image if image_to_video=True, else None
            video: a tensor [F, C, H, W] of frames
            None for embeddings if load_tensors=False
        """
        if path.is_dir():
            frames = self._read_video_from_dir(path)
        elif path.suffix == ".webm" or path.suffix == ".mp4":
            frames = self._read_video_from_webm(path)
            if "ssv2" in str(path):
                frames = crop_and_resize_frames(frames, (256, 320))
        else:
            raise ValueError(f"Unsupported video format: {path}")
        # randome resize to other resolutions
        if random.random() < 0.2:
            target_size = random.choice(list(DATASET2RES.values()))
            frames = crop_and_resize_frames(frames, target_size)

        # transform frames to tensor
        frames = [
            self.video_transforms(torch.from_numpy(img).permute(2, 0, 1).float())
            for img in frames
        ]
        video = torch.stack(frames, dim=0)  # [F, C, H, W]
        image = video[:1].clone() if self.image_to_video else None

        return image, video

    def _read_video_from_dir(
        self, path: Path, adjust_num_frames: bool = True
    ) -> List[np.ndarray]:
        assert path.is_dir(), f"Path {path} is not a directory."
        frame_paths = sorted(list(path.glob("*.png")), key=lambda x: int(x.stem))
        if adjust_num_frames:
            frame_paths = self._adjust_num_frames(frame_paths)
        frames = []
        for fp in frame_paths:
            img = np.array(Image.open(fp).convert("RGB"))
            frames.append(img)
        return frames

    def _read_video_from_webm(
        self, path: Path, adjust_num_frames: bool = True
    ) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if adjust_num_frames:
            frames = self._adjust_num_frames(frames)
        return frames


class RoboDepth(RoboDataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 81,
        id_token: Optional[str] = None,
        height_buckets: List[int] | None = None,
        width_buckets: List[int] | None = None,
        frame_buckets: List[int] | None = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
        )

    def _load_samples(self):
        """Override to load additional datasets"""
        if self.dataset_file is None or not Path(self.dataset_file).exists():
            bridge_train = self._load_openx_dataset_from_local_path("bridge")
            self.samples = bridge_train[:1000]
        else:
            pass

    def _read_depth_data(self, path: Path) -> torch.Tensor:
        """
        Reads a depth data file in .npz format and returns it as a [T, H, W] torch tensor.
        """
        assert path.is_file(), f"Depth file {path} does not exist."
        depth_array = np.load(path)["arr_0"].astype(np.float32)
        return depth_array

    def get_depth_data(
        self, rgb_dir, rgb_video, target_size
    ) -> Tuple[torch.Tensor, bool]:
        depth_path = Path(
            str(rgb_dir).replace("video", "depth/npz").replace("rgb.mp4", "depth.npz")
        )

        if depth_path.exists():
            depth_video = self._read_depth_data(depth_path)  # [T, H, W]
            depth_video = (depth_video - depth_video.min()) / (
                depth_video.max() - depth_video.min() + 1e-8
            )
            if "rlbench" in str(rgb_dir):
                depth_video = 1 - depth_video
            depth_video *= 255.0
            depth_video = np.stack([depth_video] * 3, axis=-1)  # [T, H, W, 3]
            depth_video = crop_and_resize_frames(depth_video, target_size)
            depth_video = [
                self.video_transforms(
                    torch.from_numpy(img).permute(2, 0, 1).float()
                ).unsqueeze(0)
                for img in depth_video
            ]
            depth_video = torch.cat(depth_video, dim=0)  # [T, 3, H, W]
            if len(rgb_video) != len(depth_video):
                logging.warning(
                    f"{depth_path} RGB {len(rgb_video)} != DEPTH {len(depth_video)}"
                )
                depth_video = self._adjust_num_frames(depth_video, len(rgb_video))
                depth_video = torch.stack(depth_video, dim=0)  # [T, 3, H, W]
            have_depth = True
        else:
            depth_video = torch.zeros_like(rgb_video)
            have_depth = False
        return depth_video, have_depth

    def _preprocess_video(self, path: Path):
        """
        Overrides the parent class method to load both RGB and depth data and return a concatenated video.

        Returns:
            image: [C, 1, H, W]
            rgb_video: [C, F, H, W]
            depth_video: [C, F, H, W]
            have_depth: bool
        """
        target_size = DATASET2RES["bridge"]

        # ==== Load RGB frames =====
        rgb_dir = path
        frames = self._read_rgb_data(rgb_dir)
        frames = crop_and_resize_frames(frames, target_size)
        rgb_video = [
            self.video_transforms(
                torch.from_numpy(img).permute(2, 0, 1).float()
            ).unsqueeze(0)
            for img in frames
        ]
        rgb_video = torch.cat(rgb_video, dim=0)  # [T, 3, H, W]

        # ==== Load depth data ====
        depth_video, have_depth = self.get_depth_data(rgb_dir, rgb_video, target_size)
        adjusted_frames, adjusted_depth = (
            self._adjust_num_frames(list(rgb_video)),
            self._adjust_num_frames(list(depth_video)),
        )
        rgb_video = torch.stack(adjusted_frames, dim=0)
        image = rgb_video[:1].clone()
        depth_video = torch.stack(adjusted_depth, dim=0)
        first_depth = depth_video[:1].clone()
        # fchw -> cfhw
        rgb_video, depth_video, image, first_depth = (
            rgb_video.permute(1, 0, 2, 3),
            depth_video.permute(1, 0, 2, 3),
            image.permute(1, 0, 2, 3),
            first_depth.permute(1, 0, 2, 3),
        )

        return image, first_depth, rgb_video, depth_video, have_depth

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Special logic for bucket sampler
            return {"index": index}

        if self.load_tensors:
            raise NotImplementedError("Loading tensors is not supported.")

        sample = self.samples[index]
        image, first_depth, video, depth, have_depth = self._preprocess_video(
            Path(sample[1])
        )
        instruction = self.get_instruction(index)

        ## already between [-1, 1], [c,f,h,w]
        return {
            "prompts": self.id_token + instruction,
            "first_image": image,
            "first_depth": first_depth,
            "frames": video,
            "depth": depth,
            "have_depth": have_depth,
            "path": sample[1],
        }


if __name__ == "__main__":
    # path = Path("/mnt/zhouxin-mnt/bridge/0")

    # depth_path = path / "depth/npz/depth.npz"
    # video_path = path / "video/rgb.mp4"

    # depth_array = np.load(depth_path)["arr_0"].astype(np.float32)
    # first_depth = depth_array[0] # [H, W]
    # np.save("assets/input/first_depth.npy", first_depth)

    # cap = cv2.VideoCapture(str(video_path))
    # frames = []
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frames.append(frame)
    # cap.release()

    # first_frame = frames[0] # [H, W, C]
    # first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("assets/input/first_image.png", first_frame)

    from diffusers.video_processor import VideoProcessor
    from diffusers.utils.loading_utils import load_image
    from skyreels_v2.pipelines import resizecrop

    video_preprocessor = VideoProcessor()
    img = load_image("assets/input/i2v_input.JPG").convert("RGB")
    img = resizecrop(img, 960, 544)
    img_1 = video_preprocessor.preprocess(img, 960, 544)[0]

    path = "assets/input/i2v_input.JPG"
    img = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
    img = np.array(img)
    img = [img]
    img = crop_and_resize_frames(img, (960, 544))

    transform_list = [
        transforms.Lambda(lambda x: x),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
    trans = transforms.Compose(transform_list)
    img_2 = [
        trans(torch.from_numpy(f).permute(2, 0, 1).float()) for f in img
    ][0]

    # print(torch.allclose(img_1, img_2))
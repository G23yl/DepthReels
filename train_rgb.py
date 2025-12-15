from skyreels_v2.trainer.trainer_copy_rgb import Trainer
from omegaconf import OmegaConf
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--config_path", type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    config = OmegaConf.load(args.config_path)

    trainer = Trainer(model_path=args.model_path, config=config)

    trainer.train()

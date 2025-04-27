"""Script for training the model."""

from ctnerf.training.training import train
from ctnerf.utils import get_config_dir

if __name__ == "__main__":
    train(get_config_dir() / "train_config.yaml")

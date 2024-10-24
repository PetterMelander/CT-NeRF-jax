"""Script for training the model."""

import torch

from ctnerf.training.training import train
from ctnerf.utils import get_config_dir

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.allow_tf32 = True
torch.backends.cuda.preferred_blas_library("cublaslt")

if __name__ == "__main__":
    train(get_config_dir() / "train_config.yaml")

"""Script for training the model."""

import os

from ctnerf.training.training import train
from ctnerf.utils import get_config_dir

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
    )
    train(get_config_dir() / "train_config.yaml")

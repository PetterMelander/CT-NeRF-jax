"""Script for generating a CT image from a trained model."""

from ctnerf.image_creation.ct_creation import generate_ct
from ctnerf.setup.config import get_inference_config
from ctnerf.utils import get_config_dir


def main() -> None:
    """Generate a CT image from a trained model."""
    inference_config = get_inference_config(get_config_dir() / "inference_config.yaml")
    generate_ct(inference_config)


if __name__ == "__main__":
    main()

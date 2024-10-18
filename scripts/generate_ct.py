import torch

from ctnerf.ct_creation import generate_ct
from ctnerf.utils import get_data_dir, get_dataset_metadata, get_model_dir


def main():
    output_name = "test.nii.gz"
    model_path = get_model_dir() / "coarse-only" / "20241017-174113" / "71.pt"
    ct_path = get_data_dir() / "ct_images" / "nrrd" / output_name
    img_size = [512, 512, 536]
    spacing = [1.5234375, 1.5234375, 3.0]
    origin = [-389.23828125, -538.2382812499998, -1437.9999999999995]
    direction = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0]
    dataset_metadata = get_dataset_metadata(get_data_dir() / "xrays" / "test")
    chunk_size = 4096 * 64
    device = torch.device("cuda:0")

    generate_ct(
        model_path,
        ct_path,
        dataset_metadata,
        img_size,
        spacing,
        origin,
        direction,
        chunk_size,
        device,
    )


if __name__ == "__main__":
    main()

import torch

from ctnerf.ct_creation import generate_ct
from ctnerf.utils import get_data_dir, get_dataset_metadata, get_model_dir


def main():
    device = torch.device("cuda:0")
    output_name = "test.nii.gz"
    img_size = [512, 512, 536]
    model_path = get_model_dir() / "dev-testing" / "20241013-125042" / "1.pt"
    ct_path = get_data_dir() / "ct_images" / "nrrd" / output_name
    dataset_metadata = get_dataset_metadata(get_data_dir() / "xrays" / "test")
    chunk_size = 4096 * 1024

    generate_ct(
        model_path,
        ct_path,
        img_size,
        chunk_size,
        dataset_metadata,
        device,
    )


if __name__ == "__main__":
    main()

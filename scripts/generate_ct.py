import torch

from ctnerf.ct_creation import generate_ct
from ctnerf.utils import get_data_dir, get_model_dir


def main():
    device = torch.device("cuda:0")
    output_name = "test.nii.gz"
    img_size = [512, 512, 536]
    model_path = get_model_dir() / "dev-testing" / "20241011-194445" / "1.pt"
    ct_path = get_data_dir() / "ct_images" / "nrrd" / output_name
    n_layers = 8
    layer_size = 256
    pos_embed_dim = 20
    chunk_size = 4096 * 128

    generate_ct(
        model_path, n_layers, layer_size, pos_embed_dim, ct_path, img_size, chunk_size, device
    )


if __name__ == "__main__":
    main()

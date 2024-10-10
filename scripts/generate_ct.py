from ctnerf.ct_creation import generate_ct
from ctnerf.utils import get_data_dir, get_model_dir
import torch
import sys



def main():

    device = torch.device("cuda:0")
    output_name = sys.argv[1]
    img_size = [512, 512, 536]
    model_path = get_model_dir() / sys.argv[2]
    ct_path = get_data_dir() / "ct_images" / "nrrd" / output_name
    n_layers = 8 if len(sys.argv) < 4 else int(sys.argv[3])
    layer_size = 256 if len(sys.argv) < 5 else int(sys.argv[4])
    pos_embed_dim = 10 if len(sys.argv) < 6 else int(sys.argv[5])
    chunk_size = 4096 * 128

    generate_ct(
        model_path,
        n_layers,
        layer_size,
        pos_embed_dim,
        ct_path,
        img_size,
        chunk_size,
        device
    )


if __name__ == "__main__":
    main()

import torch
from monai.data import NibabelWriter
from tqdm import tqdm
from ctnerf.models import XRayModel
from pathlib import Path



@torch.no_grad()
def generate_ct(
    model_path: Path,
    n_layers: int,
    layer_size: int,
    pos_embed_dim: int,
    ct_path: Path,
    img_size: tuple[int, int, int],
    chunk_size: int,
    device: torch.device
    ) -> None:

    model = XRayModel(n_layers, layer_size, pos_embed_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    x = torch.arange(-1, 1, 2 / img_size[0])
    y = torch.arange(-1, 1, 2 / img_size[1])
    z = torch.arange(-1, 1, 2 / img_size[2])
    coords = torch.stack(torch.meshgrid((x, y, z), indexing="ij"), dim=-1)
    coords = coords.view(-1, 3)

    coords = coords.to(device)
    
    # To avoid oom, inference is done in batches and result stored on cpu
    output = torch.tensor([], device="cpu")
    coords = coords.split(chunk_size, dim=0)
    for chunk in tqdm(coords, desc="Generating", total=len(coords)):
        chunk = chunk.to(device)
        output_chunk = model(chunk)
        output_chunk = output_chunk.view(-1)
        output = torch.cat((output, output_chunk.cpu()))

    output = output.reshape(img_size[0], img_size[1], img_size[2])

    writer = NibabelWriter() # TODO: use nibabel itself instead of monai?
    writer.set_data_array(output, channel_dim=None)
    writer.set_metadata({"affine": torch.eye(4), "original_affine": torch.eye(4)}) # TODO: handle voxel sizes
    writer.write(ct_path, verbose=False)

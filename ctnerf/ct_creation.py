import torch
from monai.data import NibabelWriter



@torch.no_grad()
def generate_ct(
    model: torch.nn.Module,
    img_size: tuple[int, int, int],
    output_name: str,
    device: torch.device = torch.device("cpu")
    ) -> None:

    training_mode = model.training
    model.eval()
    model.to(device)

    x = torch.arange(-1, 1, 2 / img_size[0])
    y = torch.arange(-1, 1, 2 / img_size[1])
    z = torch.arange(-1, 1, 2 / img_size[2])
    coords = torch.stack(torch.meshgrid((x, y, z), indexing="ij"), dim=-1)
    coords = coords.view(-1, 3)

    coords = coords.to(device)
    
    # For memory reasons, inference needs to be done in batches
    output = torch.tensor([], device=device)
    coords = coords.split(4096 * 128, dim=0)
    for chunk in coords:
        chunk = chunk.to(device)
        output_chunk = model(chunk)
        output_chunk = output_chunk.view(-1)
        output = torch.cat((output, output_chunk))

    output = output.reshape(img_size[0], img_size[1], img_size[2])

    model.train(training_mode)

    writer = NibabelWriter()
    writer.set_data_array(output, channel_dim=None)
    writer.set_metadata({"affine": torch.eye(4), "original_affine": torch.eye(4)}) # TODO: handle voxel sizes
    writer.write(output_name, verbose=False)

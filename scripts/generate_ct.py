from ctnerf.ct_creation import generate_ct
from ctnerf.utils import get_data_dir
from pathlib import Path
from ctnerf.models import XRayModel
import torch



def main():

    device = torch.device("cuda:0")
    output_name = "test_constant_range.nii.gz"
    img_size = [512, 512, 536]
    model_path = Path(__file__).parents[1] / "models" / "dev-testing" / "20241005-213935" / "5.pt"

    model = XRayModel(10, 256, 10)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    datapath = get_data_dir()
    ct_path = datapath / "ct_images" / "nrrd" / output_name
    generate_ct(model, img_size, ct_path, device)


if __name__ == "__main__":
    main()

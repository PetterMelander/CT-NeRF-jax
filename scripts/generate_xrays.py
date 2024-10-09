from ctnerf.xray_creation import generate_xrays
from ctnerf.utils import get_data_dir



datapath = get_data_dir()

ct_path = datapath / "ct_images" / "nrrd" / "2 AC_CT_TBody.nrrd"
output_dir = datapath / "xrays" / ct_path.stem
angle_interval_size = 3
max_angle = 180
device = "cuda:0"

generate_xrays(ct_path, output_dir, angle_interval_size, max_angle, device)

import sys

from ctnerf.utils import get_data_dir
from ctnerf.xray_creation import generate_xrays

max_angle = int(sys.argv[1])
angle_interval_size = int(sys.argv[2])

datapath = get_data_dir()

ct_path = datapath / "ct_images" / "nrrd" / "2 AC_CT_TBody.nrrd"
output_dir = datapath / "xrays" / ct_path.stem
device = "cuda:0"

generate_xrays(ct_path, output_dir, angle_interval_size, max_angle, device)

"""Script for generating X-ray images from a given CT image."""

from ctnerf.utils import get_data_dir
from ctnerf.xray_creation import generate_xrays

datapath = get_data_dir()

max_angle = 180
angle_interval_size = 1
ct_path = datapath / "ct_images" / "nrrd" / "2 AC_CT_TBody.nrrd"
device = "cuda:0"

# output_dir = datapath / "xrays" / ct_path.stem  # noqa: ERA001
output_dir = datapath / "xrays" / "test"

generate_xrays(ct_path, output_dir, angle_interval_size, max_angle)

"""Script for generating X-ray images from a given CT image."""

from ctnerf.utils import get_ct_dir, get_xray_dir
from ctnerf.xray_creation import generate_xrays

max_angle = 180
angle_interval_size = 1
ct_path = get_ct_dir() / "nrrd" / "2 AC_CT_TBody.nrrd"
device = "cuda:0"

output_dir = get_xray_dir() / "xrays" / "test"

generate_xrays(ct_path, output_dir, angle_interval_size, max_angle)

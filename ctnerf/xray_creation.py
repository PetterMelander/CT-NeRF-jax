"""Functions for generating X-ray images from CT images."""

import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from ctnerf.constants import MU_AIR, MU_WATER
from ctnerf.utils import convert_arrays_to_lists


def generate_xrays(
    ct_path: Path,
    output_dir: Path,
    angle_interval_size: int,
    max_angle: int,
) -> None:
    """Generate X-ray images from a given CT image by rotating it and applying Beer-Lambert law.

    Args:
        ct_path (Path): The path to the CT image to be used for X-ray generation.
        output_dir (Path): The directory where generated X-ray images will be saved.
        angle_interval_size (int): The angle interval in degrees between each X-ray image.
        max_angle (int): The maximum angle in degrees for X-ray generation.

    Returns:
        None

    """
    metadata = {}

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    ct = sitk.ReadImage(ct_path)

    file_angle_dict = {}
    for angle in tqdm(range(0, max_angle, angle_interval_size), "Generating X-rays"):
        output_file = f"{angle}.nii.gz"
        rotated_ct = _rotate_ct(ct, np.radians(angle))
        xray = _ct_to_xray(rotated_ct)
        sitk.WriteImage(xray, output_dir / output_file)
        file_angle_dict[output_file] = angle

    metadata["file_angle_map"] = file_angle_dict
    metadata["extra_metadata"] = {key: ct.GetMetaData(key) for key in ct.GetMetaDataKeys()}
    metadata["spacing"] = ct.GetSpacing()[1:]
    metadata["size"] = ct.GetSize()[1:]
    metadata["origin"] = ct.GetOrigin()[1:]
    metadata["direction"] = ct.GetDirection()[4:6] + ct.GetDirection()[7:]
    metadata["dtype"] = {
        "id": ct.GetPixelID(),
        "value": ct.GetPixelIDValue(),
        "string": ct.GetPixelIDTypeAsString(),
    }

    metadata = convert_arrays_to_lists(metadata)
    with (output_dir / "meta.json").open("w") as f:
        json.dump(metadata, f)


def _rotate_ct(img: sitk.Image, angle: float) -> sitk.Image:
    """Rotate image by {angle} degrees about the z axis.

    Uses linear interpolation and pads with -1024.

    Args:
        img (sitk.Image): Input CT image to be rotated.
        angle (float): Angle, in radians, to rotate image counter clockwise.

    Returns:
        sitk.Image: Rotated CT image.

    """
    image_center = img.TransformContinuousIndexToPhysicalPoint(
        [(sz - 1) / 2 for sz in img.GetSize()],
    )

    # Create the Euler3DTransform object
    transform = sitk.Euler3DTransform()
    transform.SetCenter(image_center)
    transform.SetRotation(0, 0, angle)

    # Apply the transform to the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024)
    resampler.SetTransform(transform)

    # Get the rotated image
    return resampler.Execute(img)


def _ct_to_xray(ct_image: sitk.Image) -> sitk.Image:
    """Create an X-ray image from a given CT image.

    Turns a CT image into an X-ray image by using a discretized version of Beer-Lambert's law
    depth-wise, using CT values as attenuation coefficient.

    Args:
        ct_image (sitk.Image): SimpleITK Image object of a CT image.

    Returns:
        sitk.Image: X-ray image with dtype float64

    """
    # Simulate X-ray image from CT
    pixel_spacing = ct_image.GetSpacing()[0]
    img_array = sitk.GetArrayFromImage(ct_image).astype(np.float64)
    img_array = _hounsfield_to_attenuation(img_array)
    img_array = np.exp(-img_array.sum(axis=2) * pixel_spacing / 10)  # Beer-Lambert law along x-axis
    img_array = img_array.clip(0, 1)  # transmittance must be between 0 and 1

    # Convert to sitk.Image, with appropriate metadata
    xray_image = sitk.GetImageFromArray(img_array)
    xray_image = sitk.Cast(xray_image, sitk.sitkFloat64)
    ct_direction = list(ct_image.GetDirection())
    xray_image.SetDirection(ct_direction[4:6] + ct_direction[7:])
    xray_image.SetSpacing(ct_image.GetSpacing()[1:])
    xray_image.SetOrigin(ct_image.GetOrigin()[1:])

    for key in ct_image.GetMetaDataKeys():
        xray_image.SetMetaData(key, ct_image.GetMetaData(key))

    return xray_image


def _hounsfield_to_attenuation(img: np.ndarray) -> np.ndarray:
    """Reverse the formula for Hounsfield units to turn the CT image into attenuation coefficients.

    Args:
        img (np.ndarray): CT image represented as 3D numpy array.

    Returns:
        np.ndarray: CT image rescaled to linear attenuation coefficients in mm.

    """
    img = img * (MU_WATER - MU_AIR) / 1000 + MU_WATER
    # remove negative attenuation coefficients caused by padding with -1024
    return np.clip(img, a_min=0, a_max=None)

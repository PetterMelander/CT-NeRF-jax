"""Script for supplementing a dataset's metadata using a sample X-ray image."""

from ctnerf.utils import add_xray_metadata, get_data_dir


def main() -> None:
    """Add metadata to a dataset's metadata using a sample X-ray image."""
    dataset_path = get_data_dir() / "xrays" / "test"
    xray_path = dataset_path / "0.nii.gz"
    add_xray_metadata(dataset_path, xray_path)


if __name__ == "__main__":
    main()

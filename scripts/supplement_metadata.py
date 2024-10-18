from ctnerf.utils import get_data_dir, add_xray_metadata


def main():
    dataset_path = get_data_dir() / "xrays" / "test"
    xray_path = dataset_path / "0.nii.gz"
    add_xray_metadata(dataset_path, xray_path)

if __name__ == "__main__":
    main()

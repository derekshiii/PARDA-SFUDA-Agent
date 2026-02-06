import os
from pathlib import Path


def clean_nnunet_labels(nnunet_dir):
    """
    Clean up extra files in labelsTr of nnUNet dataset

    Args:
        nnunet_dir: Root directory path of nnUNet dataset
    """
    # Build paths for imagesTr and labelsTr
    images_dir = Path(nnunet_dir) / "imagesTr"
    labels_dir = Path(nnunet_dir) / "labelsTr"

    # Check if directories exist
    if not images_dir.exists():
        print(f"Error: imagesTr directory does not exist: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"Error: labelsTr directory does not exist: {labels_dir}")
        return

    # Get all image files (remove _0000 suffix)
    image_files = list(images_dir.glob("*.nii.gz")) + list(images_dir.glob("*.nii"))

    # Extract base file names (remove _0000.nii.gz)
    base_names = set()
    for img_file in image_files:
        filename = img_file.stem  # Remove .nii.gz or .nii
        if filename.endswith(".nii"):  # Handle .nii.gz case
            filename = filename[:-4]

        # Remove _0000 suffix
        if filename.endswith("_0000"):
            base_name = filename[:-5]
            base_names.add(base_name)
        else:
            print(
                f"Warning: Image filename format does not match expected pattern: {img_file.name}"
            )

    print(f"Found {len(base_names)} valid base filenames in imagesTr")

    # Get all label files
    label_files = list(labels_dir.glob("*.nii.gz")) + list(labels_dir.glob("*.nii"))

    # Check and delete extra label files
    deleted_count = 0
    for label_file in label_files:
        filename = label_file.stem
        if filename.endswith(".nii"):  # Handle .nii.gz case
            filename = filename[:-4]

        # Label file should be the base_name directly (without _0000)
        if filename not in base_names:
            print(f"Deleting extra file: {label_file.name}")
            label_file.unlink()
            deleted_count += 1

    print(f"\nCleanup complete!")
    print(f"- Retained label files: {len(label_files) - deleted_count}")
    print(f"- Deleted extra files: {deleted_count}")


# Usage example
if __name__ == "__main__":
    nnunet_dataset_path = "./nnUNetFrame/nnUNet_raw/"

    # Or get path from command line
    import sys

    if len(sys.argv) > 1:
        nnunet_dataset_path = sys.argv[1]

    print(f"Starting dataset cleanup: {nnunet_dataset_path}")
    clean_nnunet_labels(nnunet_dataset_path)

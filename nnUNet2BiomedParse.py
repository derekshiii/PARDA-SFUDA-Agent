import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools
from PIL import Image


def load_volume(file_path):
    """Load spatial volumes, or a grayscale PNG as a single-slice volume."""
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".png":
        # shortcut: the 3D BiomedParse backend takes grayscale, use an RGB backend for color-dependent tasks.
        with Image.open(file_path) as image:
            data = np.asarray(image.convert("L"))[..., None]
        return nib.Nifti1Image(data, np.eye(4))
    nii = nib.load(file_path)
    if len(nii.shape) == 2:
        return nib.Nifti1Image(nii.get_fdata()[..., None], nii.affine)
    if len(nii.shape) != 3:
        raise ValueError(f"Expected a 2D/3D image: {file_path}")
    return nii


def save_volume(data, reference, output_path):
    """Save in the reference image's geometry and requested file format."""
    data = np.asarray(data)
    if str(output_path).endswith(".png"):
        if data.ndim == 3 and data.shape[-1] == 1:
            data = data[..., 0]
        if data.ndim != 2 or data.min() < 0 or data.max() > 255:
            raise ValueError("PNG labels must be a 2D array with values 0-255")
        Image.fromarray(data.astype(np.uint8)).save(output_path)
    else:
        header = reference.header.copy()
        header.set_data_dtype(data.dtype)
        nib.save(nib.Nifti1Image(data, reference.affine, header), output_path)


def load_nii_gz(file_path):
    """
    Load .nii.gz file and return numpy array
    """
    try:
        nii_img = nib.load(file_path)
        # Note: nibabel loaded data dimension order is typically (W, H, D)
        return nii_img.get_fdata()
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_png(file_path):
    """
    Load .png file and return numpy array
    For 2D images returns (H, W), consistent with single slice
    """
    try:
        img = Image.open(file_path)
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_image_file(file_path):
    """
    Automatically select loading method based on file extension
    """
    suffix = file_path.suffix.lower()
    if suffix == ".gz" and file_path.name.endswith(".nii.gz"):
        return load_nii_gz(file_path)
    elif suffix == ".png":
        data = load_png(file_path)
        # If 2D image, add third dimension to make it (H, W, 1)
        if data is not None and data.ndim == 2:
            data = np.expand_dims(data, axis=-1)
        return data
    else:
        print(f"Unsupported file format: {file_path}")
        return None


def find_matching_label_file(img_file, labels_dir):
    """
    Find label file corresponding to image file
    Supports .nii.gz and .png formats
    """
    # Handle .nii.gz format
    if img_file.name.endswith("_0000.nii.gz"):
        base_name = img_file.name.replace("_0000.nii.gz", "")
        # First try .nii.gz
        label_file = labels_dir / f"{base_name}.nii.gz"
        if label_file.exists():
            return label_file
        # Then try .png
        label_file = labels_dir / f"{base_name}.png"
        if label_file.exists():
            return label_file

    # Handle .png format
    elif img_file.name.endswith("_0000.png"):
        base_name = img_file.name.replace("_0000.png", "")
        # First try .png
        label_file = labels_dir / f"{base_name}.png"
        if label_file.exists():
            return label_file
        # Then try .nii.gz
        label_file = labels_dir / f"{base_name}.nii.gz"
        if label_file.exists():
            return label_file

    return None


def get_base_name(img_file):
    """
    Get base name from image filename
    """
    if img_file.name.endswith("_0000.nii.gz"):
        return img_file.name.replace("_0000.nii.gz", "")
    elif img_file.name.endswith("_0000.png"):
        return img_file.name.replace("_0000.png", "")
    else:
        return img_file.stem


def process_single_file(args):
    """
    Function to process a single file, used for multiprocessing
    """
    img_file, labels_dir, output_path, text_prompts = args

    # Get base filename
    base_name = get_base_name(img_file)

    try:
        nii = load_volume(img_file)
        img_data = nii.get_fdata()
        if not np.isfinite(img_data).all():
            raise ValueError("Image contains non-finite intensities")
        if img_data.min() < 0 or img_data.max() > 255 or 0 < img_data.max() <= 1:
            low, high = np.percentile(img_data, [0.5, 99.5])
            img_data = np.clip((img_data - low) / max(high - low, 1e-6), 0, 1) * 255
        img_data = img_data.astype(np.uint8)

        output_data = {
            "imgs": img_data,
            "text_prompts": text_prompts,
            "affine": nii.affine,
            "header": np.frombuffer(nii.header.binaryblock, dtype=np.uint8),
        }

        # Save as npz file
        output_file = output_path / f"{base_name}.npz"
        np.savez_compressed(output_file, **output_data)

        return f"Successfully processed: {base_name}", True

    except Exception as e:
        return f"Error processing {base_name}: {str(e)}", False


def convert_nnunet_to_biomedparse(
    input_dir, output_dir, text_prompts, num_processes=None
):
    """
    将nnUNet格式数据转换为biomedparse格式（多进程版本）
    支持 .nii.gz 和 .png 格式
    """
    if num_processes is None:
        num_processes = cpu_count()

    images_dir = Path(input_dir) / "imagesTr"
    labels_dir = Path(input_dir) / "labelsTr"
    output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")

    # Find all supported image file formats
    image_files = []
    image_files.extend(images_dir.glob("*_0000.nii.gz"))
    image_files.extend(images_dir.glob("*_0000.png"))

    if not image_files:
        raise ValueError("No supported image files found in imagesTr")

    num_processes = min(num_processes, len(image_files))

    print(f"Found {len(image_files)} image files")
    print(f"Using {num_processes} processes for conversion")

    # Count file formats
    nii_count = sum(1 for f in image_files if f.name.endswith(".nii.gz"))
    png_count = sum(1 for f in image_files if f.name.endswith(".png"))
    print(f"File format distribution: {nii_count} .nii.gz, {png_count} .png")

    process_args = [
        (img_file, labels_dir, output_path, text_prompts) for img_file in image_files
    ]

    processed_count = 0
    if num_processes == 1:
        results = [process_single_file(args) for args in process_args]
    else:
        with Pool(num_processes) as pool:
            results = list(tqdm(pool.imap(process_single_file, process_args),
                                total=len(image_files), desc="Converting files"))

    for message, success in results:
        if success:
            processed_count += 1
        # Print error messages for debugging
        if not success:
            print(message)

    if processed_count != len(image_files):
        raise RuntimeError(f"Conversion failed for {len(image_files) - processed_count} images")

    print(
        f"\nConversion completed! Processed {processed_count}/{len(image_files)} files."
    )
    print(f"Output directory: {output_path}")


def verify_conversion(npz_file_path):
    """
    Verify converted npz file
    """
    try:
        # allow_pickle=True is for loading text_prompts dictionary object
        data = np.load(npz_file_path, allow_pickle=True)

        print(f"\nVerification for: {npz_file_path}")
        print(f"Keys in file: {list(data.keys())}")

        if "imgs" in data:
            print(f"Images shape: {data['imgs'].shape}")
            print(f"Images dtype: {data['imgs'].dtype}")
            print(f"Images range: [{data['imgs'].min():.2f}, {data['imgs'].max():.2f}]")

        if "gts" in data:
            print(f"Labels shape: {data['gts'].shape}")
            print(f"Labels dtype: {data['gts'].dtype}")
            print(f"Unique label values: {np.unique(data['gts'])}")

        if "text_prompts" in data:
            # Use .item() to correctly extract Python dict from numpy array
            print(f"Text prompts: {data['text_prompts'].item()}")

        data.close()

    except Exception as e:
        print(f"Error verifying {npz_file_path}: {e}")

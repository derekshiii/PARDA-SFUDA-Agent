import os
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm


def keep_only_specified_label(
    input_dir, suffix="_label01.nii.gz", keep_label=1, output_dir=None
):
    """
    Keep only the specified label value, set all others to 0

    Parameters
    ----------
    input_dir : str or Path
        Input nii.gz label directory
    suffix : str
        File suffix, e.g., "_label01.nii.gz"
    keep_label : int
        Label value to keep
    output_dir : str or Path or None
        Output directory, None means overwrite in place
    """

    input_dir = Path(input_dir)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(f"*{suffix}"))
    assert len(files) > 0, f"No files found with suffix {suffix}"

    for nii_path in tqdm(files, desc="Processing NIfTI labels"):
        nii = nib.load(nii_path)
        data = nii.get_fdata()

        # Keep specified label, set all others to 0
        new_data = np.zeros_like(data, dtype=np.uint8)
        new_data[data == keep_label] = keep_label

        # Preserve affine and header
        new_nii = nib.Nifti1Image(new_data, nii.affine, nii.header)

        if output_dir is None:
            save_path = nii_path
        else:
            save_path = output_dir / nii_path.name

        nib.save(new_nii, save_path)


keep_only_specified_label(
    input_dir="/8TB_HDD_2/SFUDA_Agent/ZOOM_ROI_Output/refined_labels",
    suffix="_label04.nii.gz",
    keep_label=4,
)

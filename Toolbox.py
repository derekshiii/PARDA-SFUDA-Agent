import os
import sys
import json
import time, cv2
import subprocess, shutil
import numpy as np
import nibabel as nib
from scipy.ndimage import label as nd_label
from scipy import ndimage
import torch
import torch.nn.functional as F
import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from multiprocessing import Process, Queue
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Union, Literal, Tuple


def _save_as_nii(mask_array, output_path: str, affine=None) -> None:
    """Save mask array as NIfTI format."""
    if torch.is_tensor(mask_array):
        mask_array = mask_array.cpu().numpy()

    if affine is None:
        affine = np.eye(4)

    mask_array = mask_array.astype(np.uint8)
    nii_img = nib.Nifti1Image(mask_array, affine)
    nib.save(nii_img, output_path)


def _bp_worker_process(
    gpu_id,
    file_queue,
    result_queue,
    progress_queue,
    output_dir,
    checkpoint_path,
    slice_batch_size,
    interpolate_size,
    nms_threshold,
    score_threshold,
):
    """
    BiomedParse worker process function
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    # --- FIX: Ensure utils can be imported ---
    BIOMEDPARSE_ROOT = "/8TB_HDD_2/BiomedParsev2"
    if BIOMEDPARSE_ROOT not in sys.path:
        sys.path.insert(0, BIOMEDPARSE_ROOT)

    try:
        from utils import process_input, process_output, slice_nms

        # --- FIX: Use absolute config dir ---
        GlobalHydra.instance().clear()
        config_abs_path = os.path.join(BIOMEDPARSE_ROOT, "configs/model")

        # Use initialize_config_dir to support absolute paths
        hydra.initialize_config_dir(
            config_dir=config_abs_path, version_base=None, job_name=f"worker_{gpu_id}"
        )
        cfg = compose(config_name="biomedparse_3D")

        model = hydra.utils.instantiate(cfg, _convert_="object")
        model.load_pretrained(checkpoint_path)
        model.to(device)
        model.eval()

        while True:
            item = file_queue.get()
            if item is None:
                break

            input_path, filename = item

            try:
                # Load Data
                data = np.load(input_path, allow_pickle=True)
                imgs = data["imgs"]
                text_prompts = data["text_prompts"].item()

                # Prepare text
                ids = [int(k) for k in text_prompts.keys() if k != "instance_label"]
                ids.sort()
                text = "[SEP]".join([text_prompts[str(i)] for i in ids])

                # Preprocess
                imgs_tensor, pad_width, padded_size, valid_axis = process_input(
                    imgs, interpolate_size
                )
                imgs_tensor = imgs_tensor.to(device).int()

                # Inference
                input_tensor = {"image": imgs_tensor.unsqueeze(0), "text": [text]}

                with torch.no_grad():
                    output = model(
                        input_tensor, mode="eval", slice_batch_size=slice_batch_size
                    )

                mask_logits = output["predictions"]["pred_gmasks"]
                object_existence = output["predictions"]["object_existence"]

                # Resize back
                mask_logits_resized = F.interpolate(
                    mask_logits,
                    size=(interpolate_size, interpolate_size),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )

                # NMS and Post-process
                if mask_logits_resized.shape[0] > 1:
                    mask_preds = slice_nms(
                        mask_logits_resized.sigmoid(),
                        object_existence.sigmoid(),
                        iou_threshold=nms_threshold,
                        score_threshold=score_threshold,
                    )
                else:
                    mask_preds = (mask_logits_resized.sigmoid()) * (
                        object_existence.sigmoid() > score_threshold
                    ).int().unsqueeze(-1).unsqueeze(-1)

                # Merge multi-class
                bg_mask = 0.5 * torch.ones_like(mask_preds[0:1])
                keep_masks = torch.cat([bg_mask, mask_preds], dim=0)
                class_mask = keep_masks.argmax(dim=0)

                # Map back to original IDs
                id_map = {j + 1: int(ids[j]) for j in range(len(ids))}
                final_mask = class_mask.clone()
                for j_idx, real_id in id_map.items():
                    final_mask[class_mask == j_idx] = real_id

                # Crop padding
                final_mask = process_output(
                    final_mask, pad_width, padded_size, valid_axis
                )

                # Save NIfTI directly
                base_name = os.path.splitext(filename)[0]
                save_path = os.path.join(output_dir, f"{base_name}.nii.gz")
                _save_as_nii(final_mask, save_path)

                progress_queue.put({"status": "completed", "file": filename})

            except Exception as e:
                progress_queue.put(
                    {"status": "error", "file": filename, "error": str(e)}
                )

    except Exception as e:
        import traceback

        print(f"Worker {gpu_id} fatal error: {e}")
        print(traceback.format_exc())
        progress_queue.put({"status": "fatal", "error": str(e)})


class BiomedParsePredictor:
    def __init__(self, checkpoint_path: str, cuda_devices: List[str] = ["0", "1"]):
        self.checkpoint_path = checkpoint_path
        self.cuda_devices = cuda_devices
        self.num_workers = len(cuda_devices)

    def predict(
        self,
        data_path: str,
        output_dir: str,
        slice_batch_size: int = 4,
        interpolate_size: int = 512,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.5,
    ):
        os.makedirs(output_dir, exist_ok=True)

        all_files = [f for f in os.listdir(data_path) if f.endswith(".npz")]
        if not all_files:
            print("No .npz files found.")
            return

        file_queue = Queue()
        result_queue = Queue()
        progress_queue = Queue()

        for f in all_files:
            file_queue.put((os.path.join(data_path, f), f))

        for _ in range(self.num_workers):
            file_queue.put(None)

        processes = []
        print(
            f"Starting BiomedParse inference on {len(all_files)} files with {self.num_workers} GPUs..."
        )

        for i, gpu_id in enumerate(self.cuda_devices):
            p = Process(
                target=_bp_worker_process,
                args=(
                    gpu_id,
                    file_queue,
                    result_queue,
                    progress_queue,
                    output_dir,
                    self.checkpoint_path,
                    slice_batch_size,
                    interpolate_size,
                    nms_threshold,
                    score_threshold,
                ),
            )
            p.start()
            processes.append(p)

        with tqdm(total=len(all_files)) as pbar:
            finished = 0
            while finished < len(all_files):
                if not progress_queue.empty():
                    msg = progress_queue.get()
                    if msg["status"] == "completed":
                        finished += 1
                        pbar.update(1)
                    elif msg["status"] == "error":
                        finished += 1
                        pbar.update(1)
                        print(f"Error processing {msg.get('file')}: {msg.get('error')}")
                    elif msg["status"] == "fatal":
                        print(f"Fatal worker error: {msg.get('error')}")
                time.sleep(0.1)

        for p in processes:
            p.join()

        print(f"BiomedParse Inference Completed. Results in {output_dir}")
        return output_dir


# =============================================================================
#  Part 2: SAM3 Wrapper (Cross-Environment Call)
# =============================================================================


class SAM3Wrapper:
    """
    SAM3 Wrapper (Enhanced version, supports Crop mode config generation)
    """

    def __init__(self, python_path: str, script_path: str, logger=None):
        self.python_path = python_path
        self.script_path = os.path.abspath(script_path)
        self.logger = logger

    def generate_config_for_crops(
        self, metadata_path: str, crop_dir: str, target_map: Dict[int, str]
    ) -> Dict:
        """
        Generate SAM3 config file for Crops created by ZoomROI
        Strategy:
        1. Image: NIfTI path after cropping
        2. Prompt:
           - Text: Corresponding organ name
           - Box: Box covering entire Crop [0, 0, W, H] (since Crop itself is the ROI)
           - Key Slice: Middle slice of the Crop
        """
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        config = {}

        for label_info in metadata["labels"]:
            label_val = label_info["label_value"]
            if label_val not in target_map:
                continue

            organ_name = target_map[label_val]
            roi_filename = label_info["roi_file"]
            img_path = os.path.join(crop_dir, roi_filename)

            # Determine Box and Slice
            # ROI is already a stacked Volume.
            # Box: Use maximum slice dimensions
            max_h, max_w = label_info["max_slice_shape"]
            # Box format [x, y, w, h] -> [0, 0, max_w, max_h]
            box = [0, 0, max_w, max_h]

            # Key Slice: Take the middle layer
            num_slices = len(label_info["slices_info"])
            key_slice_idx = num_slices // 2

            # Use filename (without extension) as Key to ensure SAM3 output matches ZoomROI expectations
            case_key = roi_filename.replace(".nii.gz", "")

            config[case_key] = {
                "image": img_path,
                "prompts": [
                    {
                        "obj_id": int(label_val),
                        "prompt_type": "text_visual",
                        "text": organ_name,
                        "box": box,
                        "key_slice": key_slice_idx,
                    }
                ],
            }

        return config

    def predict(
        self,
        config_json_path: str,
        output_dir: str,
        gpus: List[int] = [0],
        checkpoint: str = "",
    ):
        cmd = [
            self.python_path,
            self.script_path,
            "--json_config",
            config_json_path,
            "--output_dir",
            output_dir,
            "--checkpoint",
            checkpoint,
            "--gpus",
            ",".join(map(str, gpus)),
        ]
        if self.logger:
            self.logger.info(f"Running SAM3: {' '.join(cmd)}")
        try:
            # Use subprocess to call external environment
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.logger.info("SAM3 Inference Output:\n" + result.stdout)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SAM3 Failed:\n{e.stderr}")
            raise e


class ZoomSAM3Refiner:
    """
    Pipeline scheduler combining ZoomROI and SAM3
    """

    def __init__(
        self,
        sam3_python_path: str,
        sam3_script_path: str,
        sam3_checkpoint: str,
        temp_root: str = "./temp_zoom_sam3",
        logger=None,
    ):
        self.logger = logger or logging.getLogger("ZoomSAM3Refiner")
        self.temp_root = temp_root
        self.cropper = ZoomROITool(self.logger)
        self.sam3 = SAM3Wrapper(sam3_python_path, sam3_script_path, logger=self.logger)
        self.sam3_checkpoint = sam3_checkpoint

    def run(
        self,
        image_path: str,
        coarse_label_path: str,
        output_path: str,
        target_map: Dict[int, str],
        margin: int = 10,
        gpus: List[int] = [0],
        cleanup: bool = True,
    ):
        """
        Execute full pipeline: Crop -> SAM3 Refine -> Merge
        """
        # 1. Prepare temp directory
        case_id = os.path.basename(image_path).split(".")[0]
        session_dir = os.path.join(self.temp_root, case_id)
        crop_dir = os.path.join(session_dir, "crops")
        sam3_out_dir = os.path.join(session_dir, "sam3_out")
        meta_path = os.path.join(session_dir, "metadata.json")
        sam3_config_path = os.path.join(session_dir, "sam3_config.json")

        for d in [crop_dir, sam3_out_dir]:
            os.makedirs(d, exist_ok=True)

        try:
            # 2. ZoomROI Crop
            self.logger.info(">>> Step 1: Cropping ROIs...")
            self.cropper.run_cropping(
                image_path,
                coarse_label_path,
                crop_dir,
                meta_path,
                params={"margin": margin},
            )

            # 3. Generate SAM3 Config
            self.logger.info(">>> Step 2: Generating SAM3 Config...")
            sam3_config = self.sam3.generate_config_for_crops(
                meta_path, crop_dir, target_map
            )

            if not sam3_config:
                self.logger.warning("No targets found to refine. Skipping.")
                # Optionally copy coarse label to output, or raise error
                shutil.copy(coarse_label_path, output_path)
                return output_path

            with open(sam3_config_path, "w") as f:
                json.dump(sam3_config, f, indent=2)

            # 4. SAM3 Inference
            self.logger.info(">>> Step 3: Running SAM3 Inference...")
            self.sam3.predict(
                sam3_config_path,
                sam3_out_dir,
                gpus=gpus,
                checkpoint=self.sam3_checkpoint,
            )

            # 5. ZoomROI Merge
            self.logger.info(">>> Step 4: Merging Results...")
            self.cropper.run_merging(meta_path, sam3_out_dir, output_path)
            self.logger.info(f"Refinement Complete. Saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Pipeline Failed: {e}")
            raise e

        finally:
            # 6. Cleanup
            if cleanup:
                if os.path.exists(session_dir):
                    self.logger.info(f"Cleaning up temp dir: {session_dir}")
                    shutil.rmtree(session_dir)

        return output_path


class CLAHE:
    def __init__(self, logger=None):
        self.logger = logger

    def _apply_clahe_to_slice(
        self, slice_2d, clip_limit=2.0, tile_grid_size=(8, 8), bg_threshold=0.0
    ):
        """Apply foreground CLAHE to a single slice"""
        slice_2d = slice_2d.astype(np.float32)
        fg_mask = slice_2d > bg_threshold

        # Skip if almost all background
        if fg_mask.sum() < 50:
            return slice_2d

        fg_pixels = slice_2d[fg_mask]
        fg_min, fg_max = fg_pixels.min(), fg_pixels.max()

        if fg_max - fg_min < 1e-6:
            return slice_2d

        # Normalize foreground
        normalized = np.zeros_like(slice_2d, dtype=np.uint8)
        normalized[fg_mask] = (
            (slice_2d[fg_mask] - fg_min) / (fg_max - fg_min) * 255
        ).astype(np.uint8)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(normalized)

        # Restore value range
        restored = slice_2d.copy()
        restored[fg_mask] = (enhanced[fg_mask].astype(np.float32) / 255.0) * (
            fg_max - fg_min
        ) + fg_min
        return restored

    def run_clahe(self, input_path: str, output_path: str, params: Dict = None) -> str:
        """
        Execute 3D CLAHE processing
        Args:
            params: {
                "clip_limit": 2.0,
                "tile_grid_size": [8, 8],
                "bg_threshold": 0.0
            }
        """
        if params is None:
            params = {}
        clip_limit = params.get("clip_limit", 2.0)
        grid_size = tuple(params.get("tile_grid_size", [8, 8]))
        bg_thresh = params.get("bg_threshold", 0.0)

        if self.logger:
            self.logger.info(
                f"Running CLAHE on {os.path.basename(input_path)} | Params: {params}"
            )

        # Load NIfTI
        nii = nib.load(input_path)
        data = nii.get_fdata()

        # Auto-detect slice axis (axis with fewest slices)
        shape = data.shape
        min_axis = np.argmin(shape)
        num_slices = shape[min_axis]

        enhanced_volume = np.zeros_like(data, dtype=data.dtype)

        # Process slice by slice
        for i in range(num_slices):
            if min_axis == 0:
                slice_data = data[i, :, :]
                enhanced_slice = self._apply_clahe_to_slice(
                    slice_data, clip_limit, grid_size, bg_thresh
                )
                enhanced_volume[i, :, :] = enhanced_slice
            elif min_axis == 1:
                slice_data = data[:, i, :]
                enhanced_slice = self._apply_clahe_to_slice(
                    slice_data, clip_limit, grid_size, bg_thresh
                )
                enhanced_volume[:, i, :] = enhanced_slice
            else:
                slice_data = data[:, :, i]
                enhanced_slice = self._apply_clahe_to_slice(
                    slice_data, clip_limit, grid_size, bg_thresh
                )
                enhanced_volume[:, :, i] = enhanced_slice

        # Save
        new_nii = nib.Nifti1Image(enhanced_volume, nii.affine, nii.header)
        nib.save(new_nii, output_path)

        if self.logger:
            self.logger.info(f"CLAHE completed → {os.path.basename(output_path)}")

        return output_path


class TopologyCleaner:
    """
    Post-processing tool: Topology cleaning (remove small connected components, keep Top-K)
    """

    def __init__(self, logger=None):
        self.logger = logger

    def _get_structure(self, connectivity: int):
        """Return structure element based on connectivity"""
        if connectivity == 1:
            # 6-connectivity (face-connected)
            return np.array(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                ]
            )
        elif connectivity == 2:
            # 18-connectivity (face+edge)
            return np.array(
                [
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                ]
            )
        else:
            # 26-connectivity (fully-connected)
            return np.ones((3, 3, 3))

    def _clean_label_3d(
        self,
        binary_mask: np.ndarray,
        label_id: int,
        structure,
        min_size: int,
        keep_top_k: int,
    ):
        """3D connected component cleaning"""
        labeled_array, num_features = nd_label(binary_mask, structure=structure)

        if num_features == 0:
            return binary_mask

        # Calculate size of each connected component
        sizes = np.bincount(labeled_array.ravel())[1:]  # Skip background 0

        # Filter out components smaller than min_size
        valid_indices = np.where(sizes >= min_size)[0] + 1

        if len(valid_indices) == 0:
            return np.zeros_like(binary_mask)

        # Keep the largest K components
        valid_sizes = sizes[valid_indices - 1]
        top_indices = valid_indices[np.argsort(valid_sizes)[::-1][:keep_top_k]]

        # Generate cleaned mask
        cleaned_mask = np.isin(labeled_array, top_indices).astype(np.uint8)
        return cleaned_mask

    def _clean_label_2d(
        self,
        binary_mask: np.ndarray,
        label_id: int,
        slice_axis: int,
        min_size: int,
        keep_top_k: int,
    ):
        """2D slice-by-slice cleaning (fixed version)"""
        cleaned_mask = np.zeros_like(binary_mask)
        num_slices = binary_mask.shape[slice_axis]

        structure_2d = np.ones((3, 3))  # 2D 8-connectivity

        for i in range(num_slices):
            # 提取 2D slice
            if slice_axis == 0:
                slice_2d = binary_mask[i, :, :]
            elif slice_axis == 1:
                slice_2d = binary_mask[:, i, :]
            else:
                slice_2d = binary_mask[:, :, i]

            if slice_2d.sum() == 0:
                continue

            # 2D connected component analysis
            labeled_2d, num_comp = nd_label(slice_2d, structure=structure_2d)

            if num_comp == 0:
                continue

            # Calculate size and filter
            comp_sizes = np.bincount(labeled_2d.ravel())[1:]
            valid_comp = np.where(comp_sizes >= min_size)[0] + 1

            if len(valid_comp) > 0:
                valid_sizes = comp_sizes[valid_comp - 1]
                top_k_idx = np.argsort(valid_sizes)[::-1][:keep_top_k]
                keep_comp = valid_comp[top_k_idx]

                cleaned_slice = np.isin(labeled_2d, keep_comp).astype(np.uint8)

                # Put back into 3D volume
                if slice_axis == 0:
                    cleaned_mask[i, :, :] = cleaned_slice
                elif slice_axis == 1:
                    cleaned_mask[:, i, :] = cleaned_slice
                else:
                    cleaned_mask[:, :, i] = cleaned_slice

        return cleaned_mask

    def _clean_label_hybrid(
        self,
        binary_mask: np.ndarray,
        label_id: int,
        structure,
        slice_axis: int,
        min_size: int,
        keep_top_k: int,
    ):
        """Hybrid mode: 3D first, then 2D"""
        # Step 1: 3D cleaning
        mask_3d = self._clean_label_3d(
            binary_mask, label_id, structure, min_size, keep_top_k
        )

        # Step 2: 2D refinement
        if mask_3d.sum() > 0:
            mask_final = self._clean_label_2d(
                mask_3d, label_id, slice_axis, min_size, keep_top_k
            )
            return mask_final

        return mask_3d

    def run_cleaning(
        self, input_path: str, output_path: str, params: Dict = None
    ) -> str:
        """
        Execute connected component cleaning
        Args:
            params: {
                "mode": "3d",  # '2d', '3d', 'hybrid'
                "min_size": 100,
                "keep_top_k": 1,
                "connectivity": 2,
                "multi_component_labels": {1: 2}  # Optional
            }
        """
        if params is None:
            params = {}

        mode = params.get("mode", "3d")
        min_size = params.get("min_size", 100)
        keep_top_k = params.get("keep_top_k", 1)
        connectivity = params.get("connectivity", 2)
        multi_comp_map = params.get("multi_component_labels", {})

        if self.logger:
            self.logger.info(
                f"Running Topology Cleaning on {os.path.basename(input_path)} | Mode: {mode}"
            )

        # Load NIfTI
        nii = nib.load(input_path)
        data = nii.get_fdata().astype(np.int16)
        cleaned_data = np.zeros_like(data)

        unique_labels = [l for l in np.unique(data) if l != 0]
        structure = self._get_structure(connectivity)
        slice_axis = np.argmin(data.shape)  # Used for 2D/Hybrid mode

        # Process each label
        for label_id in unique_labels:
            binary_mask = (data == label_id).astype(np.uint8)

            # Get keep_top_k for this label (prioritize from multi_component_labels)
            k = multi_comp_map.get(
                str(label_id), multi_comp_map.get(label_id, keep_top_k)
            )

            # Select cleaning method based on mode
            if mode == "3d":
                cleaned_mask = self._clean_label_3d(
                    binary_mask, label_id, structure, min_size, k
                )
            elif mode == "2d":
                cleaned_mask = self._clean_label_2d(
                    binary_mask, label_id, slice_axis, min_size, k
                )
            elif mode == "hybrid":
                cleaned_mask = self._clean_label_hybrid(
                    binary_mask, label_id, structure, slice_axis, min_size, k
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Write cleaned mask to output
            cleaned_data[cleaned_mask > 0] = label_id

            if self.logger:
                original_voxels = binary_mask.sum()
                cleaned_voxels = cleaned_mask.sum()
                self.logger.info(
                    f"  Label {label_id}: {original_voxels} → {cleaned_voxels} voxels"
                )

        # Save
        new_nii = nib.Nifti1Image(cleaned_data, nii.affine, nii.header)
        nib.save(new_nii, output_path)

        if self.logger:
            self.logger.info(f"Cleaning completed → {os.path.basename(output_path)}")

        return output_path


class SymmetryChecker:
    """
    Post-processing tool: Symmetry detection
    Calculate volume ratio and flip overlap (Dice) for paired left-right organs to detect segmentation anomalies.
    """

    def __init__(self, logger=None):
        self.logger = logger

    def _detect_axes(self, shape: Tuple[int, ...]) -> Dict[str, int]:
        """Auto-detect axes: Z-axis (fewest slices), Flip-axis (usually the first remaining)"""
        z_axis = int(np.argmin(shape))
        remaining = [i for i in range(3) if i != z_axis]
        flip_axis = remaining[0]
        return {"z_axis": z_axis, "flip_axis": flip_axis}

    def _calculate_metrics(
        self, mask_l: np.ndarray, mask_r: np.ndarray, axes: Dict[str, int]
    ) -> Dict:
        """Core calculation: Flip right side, align centroids, calculate Dice and volume ratio"""
        vol_l = np.sum(mask_l)
        vol_r = np.sum(mask_r)

        if vol_l == 0 or vol_r == 0:
            return {"valid_pair": False, "error": "One side is empty"}

        flip_ax = axes["flip_axis"]
        # 1. Flip right organ
        mask_r_flipped = np.flip(mask_r, axis=flip_ax)

        # 2. Centroid alignment
        com_l = np.array(ndimage.center_of_mass(mask_l))
        com_r_flipped = np.array(ndimage.center_of_mass(mask_r_flipped))
        shift_vector = com_l - com_r_flipped

        # 3. Translation alignment (Order=0 preserves binary)
        mask_r_aligned = ndimage.shift(mask_r_flipped, shift_vector, order=0)

        # 4. Calculate metrics
        intersection = np.logical_and(mask_l, mask_r_aligned).sum()
        union = mask_l.sum() + mask_r_aligned.sum()
        dice_3d = (2.0 * intersection) / union if union > 0 else 0.0
        vol_ratio = float(min(vol_l, vol_r) / max(vol_l, vol_r))

        return {
            "valid_pair": True,
            "volume_ratio": vol_ratio,
            "symmetry_dice_3d": float(dice_3d),
        }

    def run_check(self, input_path: str, params: Dict = None) -> Dict:
        """
        Execute symmetry check
        Args:
            input_path: Path to predicted Label NIfTI file
            params: {
                "organ_pairs": [{"Left Kidney": 1, "Right Kidney": 2}],
                "vol_threshold": 0.2,
                "dice_threshold": 0.1
            }
        Returns:
            Dict: Contains abnormality flag and detailed metrics
        """
        if params is None:
            params = {}
        organ_pairs = params.get("organ_pairs", [])
        vol_thresh = params.get("vol_threshold", 0.2)
        dice_thresh = params.get("dice_threshold", 0.1)

        if self.logger:
            self.logger.info(
                f"Running Symmetry Check on {os.path.basename(input_path)}"
            )

        nii = nib.load(input_path)
        data = np.round(nii.get_fdata()).astype(np.int16)
        axes = self._detect_axes(data.shape)

        report = {"is_abnormal": False, "abnormal_pairs": [], "details": {}}

        for pair in organ_pairs:
            # Parse pair info {'NameL': 1, 'NameR': 2}
            keys = list(pair.keys())
            name_l, id_l = keys[0], pair[keys[0]]
            name_r, id_r = keys[1], pair[keys[1]]
            pair_name = f"{name_l}_vs_{name_r}"

            mask_l = data == id_l
            mask_r = data == id_r

            if not np.any(mask_l) or not np.any(mask_r):
                # Missing either one is a potential anomaly, or not detected
                continue

            metrics = self._calculate_metrics(mask_l, mask_r, axes)

            if metrics.get("valid_pair"):
                is_abnormal = (metrics["volume_ratio"] < vol_thresh) or (
                    metrics["symmetry_dice_3d"] < dice_thresh
                )

                metrics["abnormal_flag"] = is_abnormal
                report["details"][pair_name] = metrics

                if is_abnormal:
                    report["is_abnormal"] = True
                    report["abnormal_pairs"].append(pair_name)
                    if self.logger:
                        self.logger.warning(
                            f"Symmetry Alert [{pair_name}]: VolRatio={metrics['volume_ratio']:.2f}, Dice={metrics['symmetry_dice_3d']:.2f}"
                        )

        return report


class ZoomROITool:
    """
    ROI cropping and restoration tool (maintains original logic)
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger("ZoomROI")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    def _find_axial_axis(self, shape: Tuple[int, ...]) -> int:
        return int(np.argmin(shape))

    def _crop_single_label(self, img_data, label_data, label_value, axial_axis, margin):
        mask_3d = label_data == label_value
        coords = np.where(mask_3d)
        slice_indices = sorted(np.unique(coords[axial_axis]))

        if not slice_indices:
            return None

        other_axes = [i for i in range(3) if i != axial_axis]
        slices_data = []
        slices_info = []

        for slice_idx in slice_indices:
            # Extract slice
            if axial_axis == 0:
                slice_mask = mask_3d[slice_idx, :, :]
                slice_img = img_data[slice_idx, :, :]
            elif axial_axis == 1:
                slice_mask = mask_3d[:, slice_idx, :]
                slice_img = img_data[:, slice_idx, :]
            else:
                slice_mask = mask_3d[:, :, slice_idx]
                slice_img = img_data[:, :, slice_idx]

            # Calculate BBox
            slice_coords = np.where(slice_mask)
            if len(slice_coords[0]) == 0:
                continue

            bbox_slice = {}
            for i, axis in enumerate(other_axes):
                min_val = max(0, int(np.min(slice_coords[i])) - margin)
                max_val = min(
                    img_data.shape[axis], int(np.max(slice_coords[i])) + margin + 1
                )
                bbox_slice[f"axis_{axis}"] = {"min": min_val, "max": max_val}

            # Crop
            cropped = slice_img[
                bbox_slice[f"axis_{other_axes[0]}"]["min"] : bbox_slice[
                    f"axis_{other_axes[0]}"
                ]["max"],
                bbox_slice[f"axis_{other_axes[1]}"]["min"] : bbox_slice[
                    f"axis_{other_axes[1]}"
                ]["max"],
            ]

            slices_data.append(cropped)
            slices_info.append(
                {
                    "slice_idx": int(slice_idx),
                    "bbox": bbox_slice,
                    "shape": list(cropped.shape),
                }
            )

        # Padding
        max_h = max((s.shape[0] for s in slices_data), default=0)
        max_w = max((s.shape[1] for s in slices_data), default=0)

        padded_slices = []
        for s in slices_data:
            pad_h = max_h - s.shape[0]
            pad_w = max_w - s.shape[1]
            padded_slices.append(
                np.pad(s, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            )

        # Stack (Follow original stacking logic: always create a volume)
        if axial_axis == 0:
            roi_volume = np.stack(padded_slices, axis=0)
        elif axial_axis == 1:
            roi_volume = np.stack(padded_slices, axis=1)
        else:
            roi_volume = np.stack(padded_slices, axis=2)

        return {
            "label_value": int(label_value),
            "roi_volume": roi_volume,
            "max_slice_shape": [max_h, max_w],
            "slices_info": slices_info,
        }

    def run_cropping(
        self, image_path, label_path, output_roi_dir, output_meta_path, params=None
    ):
        if params is None:
            params = {}
        margin = params.get("margin", 10)

        case_id = os.path.basename(image_path).split(".")[0]  # Simple split
        if self.logger:
            self.logger.info(f"Cropping Case: {case_id}")

        img_nii = nib.load(image_path)
        lbl_nii = nib.load(label_path)
        img_data = img_nii.get_fdata()
        lbl_data = lbl_nii.get_fdata().astype(np.int32)
        axial_axis = self._find_axial_axis(img_data.shape)

        metadata = {
            "case_id": case_id,
            "original_shape": list(img_data.shape),
            "original_affine": img_nii.affine.tolist(),
            "axial_axis": int(axial_axis),
            "labels": [],
        }

        generated_files = []
        unique_labels = [int(l) for l in np.unique(lbl_data) if l > 0]

        for lbl in unique_labels:
            res = self._crop_single_label(img_data, lbl_data, lbl, axial_axis, margin)
            if res:
                roi_name = f"{case_id}_label{lbl:02d}.nii.gz"
                roi_path = os.path.join(output_roi_dir, roi_name)
                nib.save(nib.Nifti1Image(res["roi_volume"], np.eye(4)), roi_path)
                generated_files.append(roi_path)

                res["roi_file"] = roi_name
                del res["roi_volume"]
                metadata["labels"].append(res)

        os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)
        with open(output_meta_path, "w") as f:
            json.dump(metadata, f, indent=2, cls=self.NumpyEncoder)

        return generated_files

    def run_merging(self, meta_path, refined_roi_dir, output_path):
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        final_label = np.zeros(metadata["original_shape"], dtype=np.int32)
        axial_axis = metadata["axial_axis"]

        sorted_labels = sorted(metadata["labels"], key=lambda x: x["label_value"])

        for info in sorted_labels:
            label_val = info["label_value"]
            roi_path = os.path.join(refined_roi_dir, info["roi_file"])

            if not os.path.exists(roi_path):
                if self.logger:
                    self.logger.warning(f"Refined ROI not found: {roi_path}")
                continue

            refined_data = nib.load(roi_path).get_fdata().astype(np.int32)
            other_axes = [i for i in range(3) if i != axial_axis]

            for i, slice_info in enumerate(info["slices_info"]):
                if axial_axis == 0:
                    refined_slice_padded = refined_data[i, :, :]
                elif axial_axis == 1:
                    refined_slice_padded = refined_data[:, i, :]
                else:
                    refined_slice_padded = refined_data[:, :, i]

                h, w = slice_info["shape"]
                refined_slice = refined_slice_padded[:h, :w]
                refined_mask = (
                    refined_slice == label_val
                )  # Or just > 0 if binary per file

                orig_idx = slice_info["slice_idx"]
                bbox_0 = slice_info["bbox"][f"axis_{other_axes[0]}"]
                bbox_1 = slice_info["bbox"][f"axis_{other_axes[1]}"]

                if axial_axis == 0:
                    final_label[
                        orig_idx,
                        bbox_0["min"] : bbox_0["max"],
                        bbox_1["min"] : bbox_1["max"],
                    ][refined_mask] = label_val
                elif axial_axis == 1:
                    final_label[
                        bbox_0["min"] : bbox_0["max"],
                        orig_idx,
                        bbox_1["min"] : bbox_1["max"],
                    ][refined_mask] = label_val
                else:
                    final_label[
                        bbox_0["min"] : bbox_0["max"],
                        bbox_1["min"] : bbox_1["max"],
                        orig_idx,
                    ][refined_mask] = label_val

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(
            nib.Nifti1Image(final_label, np.array(metadata["original_affine"])),
            output_path,
        )
        return output_path

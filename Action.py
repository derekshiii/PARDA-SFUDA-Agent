import os
import json
import shutil
import numpy as np
import nibabel as nib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from nnUNet2BiomedParse import convert_nnunet_to_biomedparse
from Toolbox import (
    BiomedParsePredictor,
    SAM3Wrapper,
    ZoomSAM3Refiner,
    TopologyCleaner,
    CLAHE,
)
from tqdm import tqdm

SAM3_PYTHON_PATH = "/home/syl/miniconda3/envs/sam3/bin/python"
SAM3_SCRIPT_PATH = "run_sam3_standalone.py"


class ActionModule:
    def __init__(
        self,
        result_dir: str,
        logger,
        biomedparse_checkpoint: str,
        sam3_checkpoint: str,
        temp_dir: Optional[str] = None,
        gpus: List[int] = [0, 1],
    ):
        self.result_dir = result_dir
        self.logger = logger
        self.gpus = gpus

        if temp_dir is None:
            temp_dir = os.path.join(result_dir, "temp")
        self.temp_dir = temp_dir

        self.biomed_input_dir = os.path.join(self.temp_dir, "biomedparse_input_npz")
        self.pseudolabel_dir = os.path.join(result_dir, "Pseudolabels_NIfTI")

        os.makedirs(self.biomed_input_dir, exist_ok=True)
        os.makedirs(self.pseudolabel_dir, exist_ok=True)

        self.bp_predictor = BiomedParsePredictor(
            checkpoint_path=biomedparse_checkpoint, cuda_devices=[str(g) for g in gpus]
        )

        self.sam3_wrapper = SAM3Wrapper(
            python_path=SAM3_PYTHON_PATH,
            script_path=SAM3_SCRIPT_PATH,
            logger=self.logger,
        )
        self.sam3_checkpoint = sam3_checkpoint

    def execute_adaptation_plan(
        self, dataset_path: str, adaptation_plan: Dict, primary_tool: str = None
    ) -> str:  # primary_tool is now optional
        """
        Execute actions according to the Adaptation Plan.
        If primary_tool is None, read from adaptation_plan
        """
        target_organs = adaptation_plan.get("target_organs", [])
        modality = adaptation_plan.get("domain_context", {}).get("modality", "CT")
        prompts = adaptation_plan.get("text_prompts", {})

        if not prompts:
            raise ValueError("No text prompts found in adaptation plan!")

        # === Read tool selection from adaptation_plan ===
        strategy_recs = adaptation_plan.get("domain_context", {}).get(
            "strategy_recommendations", {}
        )

        if primary_tool is None:
            primary_tool = strategy_recs.get("primary_tool", "Dual-Expert")

        # === Read CLAHE configuration ===
        use_clahe = strategy_recs.get("use_clahe", False)
        clahe_params = strategy_recs.get(
            "clahe_params",
            {"clip_limit": 2.0, "tile_grid_size": [8, 8], "bg_threshold": 0.0},
        )

        self.logger.info(f"Executing with tool: {primary_tool}")
        self.logger.info(f"CLAHE Enhancement: {use_clahe}")
        if use_clahe:
            self.logger.info(f"CLAHE Params: {clahe_params}")
        self.logger.info(f"Target organs: {target_organs}")

        # === If CLAHE preprocessing is needed, process all images first ===
        if use_clahe:
            dataset_path = self._apply_clahe_to_dataset(dataset_path, clahe_params)

        final_pseudolabel_dir = None

        if primary_tool == "BiomedParse":
            self.logger.info(f"BiomedParse prompts: {prompts}")
            npz_input_dir = self._prepare_data_for_biomedparse(dataset_path, prompts)
            final_pseudolabel_dir = self.bp_predictor.predict(
                data_path=npz_input_dir,
                output_dir=self.pseudolabel_dir,
                slice_batch_size=4,
                nms_threshold=0.5,
            )

        elif primary_tool == "Dual-Expert":
            final_pseudolabel_dir = self._execute_dual_expert(
                dataset_path=dataset_path,
                target_organs=target_organs,
                prompts=prompts,
                dice_threshold=0.5,
            )

        elif primary_tool == "SAM3":
            self.logger.info(f"SAM3 will use organ names: {target_organs}")
            sam3_config_path = os.path.join(self.temp_dir, "sam3_run_config.json")
            sam3_run_config = self._generate_sam3_text_config(
                dataset_path, target_organs
            )

            with open(sam3_config_path, "w") as f:
                json.dump(sam3_run_config, f, indent=2)

            self.sam3_wrapper.predict(
                config_json_path=sam3_config_path,
                output_dir=self.pseudolabel_dir,
                gpus=self.gpus,
                checkpoint=self.sam3_checkpoint,
            )
            final_pseudolabel_dir = self.pseudolabel_dir

        else:
            raise ValueError(f"Unsupported tool: {primary_tool}")

        if final_pseudolabel_dir and os.path.exists(final_pseudolabel_dir):
            constructed_dataset_path = self._construct_nnunet_dataset(
                pseudolabel_dir=final_pseudolabel_dir,
                original_dataset_path=dataset_path,
                target_organs=target_organs,
                modality=modality,
            )
            return constructed_dataset_path
        else:
            raise RuntimeError("Inference failed to produce pseudolabels.")

    def _execute_dual_expert(
        self,
        dataset_path: str,
        target_organs: List[str],
        prompts: Dict[str, str],
        dice_threshold: float = 0.7,
    ) -> str:
        """
        Core implementation of Dual-Expert strategy

        Workflow:
        1. BiomedParse inference
        2. Generate SAM3 Text+Visual Prompts based on BP results
        3. SAM3 inference
        4. Calculate 3D Dice, fuse results

        Args:
            dataset_path: Original dataset path
            target_organs: List of target organs
            prompts: Text prompts for BiomedParse
            dice_threshold: Dice threshold, fuse if above this value
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DUAL-EXPERT MODE ACTIVATED")
        self.logger.info("=" * 60)

        # === Step 1: BiomedParse Inference ===
        self.logger.info("\n>>> Step 1/4: Running BiomedParse Inference...")
        bp_output_dir = os.path.join(self.temp_dir, "dual_expert", "biomedparse_output")
        os.makedirs(bp_output_dir, exist_ok=True)

        npz_input_dir = self._prepare_data_for_biomedparse(dataset_path, prompts)
        self.bp_predictor.predict(
            data_path=npz_input_dir,
            output_dir=bp_output_dir,
            slice_batch_size=4,
            nms_threshold=0.5,
        )
        self.logger.info(f"BiomedParse results saved to: {bp_output_dir}")

        # === Step 2: Generate SAM3 Text+Visual Prompts ===
        self.logger.info(
            "\n>>> Step 2/4: Generating SAM3 Text+Visual Prompts from BP results..."
        )
        sam3_config = self._generate_sam3_prompts_from_bp(
            dataset_path=dataset_path,
            bp_label_dir=bp_output_dir,
            target_organs=target_organs,
        )

        sam3_config_path = os.path.join(
            self.temp_dir, "dual_expert", "sam3_config.json"
        )
        os.makedirs(os.path.dirname(sam3_config_path), exist_ok=True)
        with open(sam3_config_path, "w") as f:
            json.dump(sam3_config, f, indent=2)

        self.logger.info(f"SAM3 config generated: {sam3_config_path}")
        self.logger.info(
            f"Total prompts: {sum(len(v['prompts']) for v in sam3_config.values())}"
        )

        # === Step 3: SAM3 Inference ===
        self.logger.info(
            "\n>>> Step 3/4: Running SAM3 Inference with Text+Visual Prompts..."
        )
        sam3_output_dir = os.path.join(self.temp_dir, "dual_expert", "sam3_output")
        os.makedirs(sam3_output_dir, exist_ok=True)

        self.sam3_wrapper.predict(
            config_json_path=sam3_config_path,
            output_dir=sam3_output_dir,
            gpus=self.gpus,
            checkpoint=self.sam3_checkpoint,
        )
        self.logger.info(f"SAM3 results saved to: {sam3_output_dir}")

        # === Step 4: Dice Calculation and Fusion ===
        self.logger.info("\n>>> Step 4/4: Computing Dice and Fusing Results...")
        final_output_dir = self.pseudolabel_dir
        os.makedirs(final_output_dir, exist_ok=True)

        fusion_stats = self._fuse_dual_expert_results(
            bp_label_dir=bp_output_dir,
            sam3_label_dir=sam3_output_dir,
            output_dir=final_output_dir,
            target_organs=target_organs,
            dice_threshold=dice_threshold,
        )

        # Print fusion statistics
        self.logger.info("\n" + "=" * 60)
        self.logger.info("DUAL-EXPERT FUSION STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total cases processed: {fusion_stats['total_cases']}")
        self.logger.info(f"BP-only cases: {fusion_stats['bp_only_count']}")
        self.logger.info(f"Fused cases: {fusion_stats['fused_count']}")
        self.logger.info(f"Average Dice (fused): {fusion_stats['avg_dice_fused']:.4f}")
        self.logger.info(
            f"Average Dice (BP-only): {fusion_stats['avg_dice_bp_only']:.4f}"
        )
        self.logger.info("=" * 60 + "\n")

        return final_output_dir

    def _generate_sam3_prompts_from_bp(
        self, dataset_path: str, bp_label_dir: str, target_organs: List[str]
    ) -> Dict:
        """
        Extract BBox and Key Slice from BiomedParse pseudo-labels to generate SAM3 config

        Args:
            dataset_path: Original dataset path
            bp_label_dir: BiomedParse output label directory
            target_organs: List of target organs

        Returns:
            SAM3 configuration dictionary
        """
        images_dir = os.path.join(dataset_path, "imagesTr")
        if not os.path.exists(images_dir):
            images_dir = os.path.join(dataset_path, "imagesTs")

        label_files = sorted(
            [f for f in os.listdir(bp_label_dir) if f.endswith(".nii.gz")]
        )
        sam3_config = {}

        for label_file in label_files:
            case_name = label_file.replace(".nii.gz", "")
            label_path = os.path.join(bp_label_dir, label_file)
            img_path = os.path.join(images_dir, f"{case_name}_0000.nii.gz")

            if not os.path.exists(img_path):
                self.logger.warning(f"Image not found for {case_name}, skipping")
                continue

            # Load BP labels
            label_nii = nib.load(label_path)
            label_data = np.round(label_nii.get_fdata()).astype(np.int16)

            # Generate Prompt for each organ
            case_prompts = []
            for label_id, organ_name in enumerate(target_organs, start=1):
                organ_mask = label_data == label_id

                if not np.any(organ_mask):
                    self.logger.debug(f"  {case_name}: No {organ_name} found, skipping")
                    continue

                # Extract BBox and Key Slice
                bbox_info = self._extract_bbox_and_key_slice(
                    organ_mask, label_data.shape
                )

                if bbox_info is None:
                    continue

                case_prompts.append(
                    {
                        "obj_id": label_id,
                        "prompt_type": "text_visual",
                        "text": organ_name,
                        "box": bbox_info["box"],  # [x, y, w, h]
                        "key_slice": bbox_info["key_slice"],
                    }
                )

                self.logger.debug(
                    f"  {case_name}/{organ_name}: BBox={bbox_info['box']}, KeySlice={bbox_info['key_slice']}"
                )

            if case_prompts:
                sam3_config[case_name] = {"image": img_path, "prompts": case_prompts}

        return sam3_config

    def _extract_bbox_and_key_slice(
        self, organ_mask: np.ndarray, volume_shape: Tuple
    ) -> Optional[Dict]:
        """
        Extract BBox and Key Slice from 3D Mask

        Strategy:
        - Key Slice: Select the slice with the largest organ area
        - BBox: Calculate 2D BBox on the Key Slice

        Args:
            organ_mask: 3D binary Mask
            volume_shape: Volume shape (D, H, W) or other arrangement

        Returns:
            {"box": [x, y, w, h], "key_slice": int} or None
        """
        # Determine slice axis (minimum dimension)
        slice_axis = int(np.argmin(volume_shape))
        num_slices = volume_shape[slice_axis]

        # Calculate organ area for each slice
        slice_areas = []
        for i in range(num_slices):
            if slice_axis == 0:
                slice_2d = organ_mask[i, :, :]
            elif slice_axis == 1:
                slice_2d = organ_mask[:, i, :]
            else:
                slice_2d = organ_mask[:, :, i]

            slice_areas.append(np.sum(slice_2d))

        # Select the slice with maximum area
        if max(slice_areas) == 0:
            return None

        key_slice = int(np.argmax(slice_areas))

        # Extract the Mask from that slice
        if slice_axis == 0:
            key_slice_mask = organ_mask[key_slice, :, :]
        elif slice_axis == 1:
            key_slice_mask = organ_mask[:, key_slice, :]
        else:
            key_slice_mask = organ_mask[:, :, key_slice]

        # Calculate 2D BBox
        coords = np.where(key_slice_mask)
        if len(coords[0]) == 0:
            return None

        y_min, y_max = int(coords[0].min()), int(coords[0].max())
        x_min, x_max = int(coords[1].min()), int(coords[1].max())

        # BBox format: [x, y, width, height]
        box = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]

        return {"box": box, "key_slice": key_slice}

    def _fuse_dual_expert_results(
        self,
        bp_label_dir: str,
        sam3_label_dir: str,
        output_dir: str,
        target_organs: List[str],
        dice_threshold: float,
    ) -> Dict:
        """
        Fuse BiomedParse and SAM3 results

        Calculate Dice per case:
        - Dice < threshold: Use BP only
        - Dice >= threshold: Fuse BP + SAM3 (union)

        Returns:
            Fusion statistics
        """
        bp_files = sorted(
            [f for f in os.listdir(bp_label_dir) if f.endswith(".nii.gz")]
        )

        stats = {
            "total_cases": 0,
            "bp_only_count": 0,
            "fused_count": 0,
            "avg_dice_fused": 0.0,
            "avg_dice_bp_only": 0.0,
            "dice_scores": [],
        }

        for bp_file in bp_files:
            case_name = bp_file.replace(".nii.gz", "")
            bp_path = os.path.join(bp_label_dir, bp_file)
            sam3_path = os.path.join(sam3_label_dir, bp_file)
            output_path = os.path.join(output_dir, bp_file)

            # Load BP results
            bp_nii = nib.load(bp_path)
            bp_data = np.round(bp_nii.get_fdata()).astype(np.int16)

            # Check if SAM3 results exist
            if not os.path.exists(sam3_path):
                self.logger.warning(
                    f"  {case_name}: SAM3 result missing, using BP only"
                )
                shutil.copy(bp_path, output_path)
                stats["bp_only_count"] += 1
                stats["total_cases"] += 1
                continue

            # Load SAM3 results
            sam3_nii = nib.load(sam3_path)
            sam3_data = np.round(sam3_nii.get_fdata()).astype(np.int16)

            # Calculate 3D Dice
            dice_score = self._calculate_3d_dice(bp_data, sam3_data, target_organs)
            stats["dice_scores"].append(dice_score)

            # Decide whether to fuse based on Dice
            if dice_score < dice_threshold:
                self.logger.info(
                    f"  {case_name}: Dice={dice_score:.4f} < {dice_threshold}, using BP only"
                )
                final_data = bp_data
                stats["bp_only_count"] += 1
            else:
                self.logger.info(
                    f"  {case_name}: Dice={dice_score:.4f} >= {dice_threshold}, fusing BP+SAM3"
                )
                final_data = self._fuse_masks(bp_data, sam3_data, len(target_organs))
                stats["fused_count"] += 1

            # Save fused results
            final_nii = nib.Nifti1Image(final_data, bp_nii.affine, bp_nii.header)
            nib.save(final_nii, output_path)

            stats["total_cases"] += 1

        # Calculate average Dice
        if stats["dice_scores"]:
            stats["avg_dice_fused"] = np.mean([d for d in stats["dice_scores"]])
            bp_only_dices = stats["dice_scores"][: stats["bp_only_count"]]
            stats["avg_dice_bp_only"] = np.mean(bp_only_dices) if bp_only_dices else 0.0

        return stats

    def _calculate_3d_dice(
        self, pred1: np.ndarray, pred2: np.ndarray, target_organs: List[str]
    ) -> float:
        """
        Calculate average Dice between two multi-label 3D Volumes

        Args:
            pred1: First prediction (BiomedParse)
            pred2: Second prediction (SAM3)
            target_organs: List of organs (used to determine number of labels)

        Returns:
            Average Dice Score
        """
        dice_scores = []

        for label_id in range(1, len(target_organs) + 1):
            mask1 = pred1 == label_id
            mask2 = pred2 == label_id

            # Skip if both are empty
            if not np.any(mask1) and not np.any(mask2):
                continue

            # Calculate Dice
            intersection = np.sum(mask1 & mask2)
            union = np.sum(mask1) + np.sum(mask2)

            if union > 0:
                dice = (2.0 * intersection) / union
                dice_scores.append(dice)

        return np.mean(dice_scores) if dice_scores else 0.0

    def _fuse_masks(
        self, bp_mask: np.ndarray, sam3_mask: np.ndarray, num_labels: int
    ) -> np.ndarray:
        """
        Fuse two Masks (union)

        Conflict resolution strategy: BP priority (fill BP first, then fill non-overlapping regions from SAM3)

        Args:
            bp_mask: BiomedParse result
            sam3_mask: SAM3 result
            num_labels: Number of labels

        Returns:
            Fused Mask
        """
        fused = bp_mask.copy()

        for label_id in range(1, num_labels + 1):
            # Additional regions from SAM3 for this label (non-overlapping with BP)
            sam3_region = sam3_mask == label_id
            bp_occupied = fused > 0  # All regions already occupied by BP

            # Add non-overlapping parts from SAM3
            additional_region = sam3_region & (~bp_occupied)
            fused[additional_region] = label_id

        return fused

    # === Keep original methods ===
    def _prepare_data_for_biomedparse(
        self, dataset_path: str, prompts: Dict[str, str]
    ) -> str:
        """Internal: Call conversion script to convert nnUNet data to NPZ format for BiomedParse"""
        text_prompts_map = {}
        for idx, (organ, prompt) in enumerate(prompts.items(), start=1):
            text_prompts_map[str(idx)] = prompt
        text_prompts_map["instance_label"] = 0

        if os.path.exists(self.biomed_input_dir):
            shutil.rmtree(self.biomed_input_dir)
        os.makedirs(self.biomed_input_dir, exist_ok=True)

        self.logger.info(
            f"Converting data for BiomedParse: {dataset_path} -> {self.biomed_input_dir}"
        )
        convert_nnunet_to_biomedparse(
            input_dir=dataset_path,
            output_dir=self.biomed_input_dir,
            text_prompts=text_prompts_map,
        )
        return self.biomed_input_dir

    def _generate_sam3_text_config(
        self, dataset_path: str, target_organs: List[str]
    ) -> Dict:
        """Internal: Generate text-only mode Config for SAM3"""
        config = {}
        images_dir = os.path.join(dataset_path, "imagesTr")
        img_files = sorted(
            [f for f in os.listdir(images_dir) if f.endswith("_0000.nii.gz")]
        )

        for img_file in img_files:
            case_name = img_file.replace("_0000.nii.gz", "")
            img_path = os.path.join(images_dir, img_file)

            case_prompts = []
            for idx, organ_name in enumerate(target_organs, start=1):
                case_prompts.append(
                    {"obj_id": idx, "prompt_type": "text", "text": organ_name}
                )

            config[case_name] = {"image": img_path, "prompts": case_prompts}

        return config

    def _construct_nnunet_dataset(
        self,
        pseudolabel_dir: str,
        original_dataset_path: str,
        target_organs: List[str],
        modality: str,
    ) -> str:
        """Package generated pseudo-labels into nnUNet format"""
        self.logger.info("Constructing nnUNet Dataset from pseudolabels...")

        nnunet_raw_root = os.path.dirname(os.path.abspath(original_dataset_path))
        original_dirname = os.path.basename(os.path.abspath(original_dataset_path))

        import re

        match = re.match(r"Dataset(\d+)_", original_dirname)
        original_suffix = original_dirname.split("_", 1)[1] if match else "SFUDA_Data"

        existing_ids = set()
        if os.path.exists(nnunet_raw_root):
            for d in os.listdir(nnunet_raw_root):
                if d.startswith("Dataset"):
                    try:
                        existing_ids.add(int(d[7:10]))
                    except:
                        pass

        next_id = 1
        while next_id in existing_ids:
            next_id += 1

        new_dataset_name = f"Dataset{next_id:03d}_{original_suffix}_QWen3-235B"
        output_dataset_dir = os.path.join(nnunet_raw_root, new_dataset_name)

        imagesTr = os.path.join(output_dataset_dir, "imagesTr")
        labelsTr = os.path.join(output_dataset_dir, "labelsTr")
        os.makedirs(imagesTr, exist_ok=True)
        os.makedirs(labelsTr, exist_ok=True)

        label_files = [f for f in os.listdir(pseudolabel_dir) if f.endswith(".nii.gz")]

        valid_source_dirs = [
            os.path.join(original_dataset_path, "imagesTr"),
            os.path.join(original_dataset_path, "imagesTs"),
        ]
        source_dir = next((d for d in valid_source_dirs if os.path.exists(d)), None)

        if not source_dir:
            raise FileNotFoundError("Original imagesTr not found.")

        success_count = 0
        for lbl in label_files:
            shutil.copy2(
                os.path.join(pseudolabel_dir, lbl), os.path.join(labelsTr, lbl)
            )

            img_name = lbl.replace(".nii.gz", "_0000.nii.gz")
            src_img = os.path.join(source_dir, img_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(imagesTr, img_name))
                success_count += 1

        labels_json = {"background": 0}
        for i, organ in enumerate(target_organs, 1):
            labels_json[organ] = i

        dataset_json = {
            "channel_names": {"0": modality},
            "labels": labels_json,
            "numTraining": success_count,
            "file_ending": ".nii.gz",
            "name": new_dataset_name,
        }

        with open(os.path.join(output_dataset_dir, "dataset.json"), "w") as f:
            json.dump(dataset_json, f, indent=4)

        self.logger.info(
            f"Dataset constructed at: {output_dataset_dir} (Size: {success_count})"
        )
        return output_dataset_dir

    # Retain original refinement methods...
    def execute_refinement_actions(
        self,
        constructed_dataset_path: str,
        refinement_strategy: Dict,
        adaptation_plan: Dict,
    ) -> None:
        """Execute Refinement Actions (keep original implementation)"""
        pass  # Retain original code

    def _apply_clahe_to_dataset(self, dataset_path: str, clahe_params: Dict) -> str:
        """
        Apply CLAHE preprocessing to the entire dataset

        Args:
            dataset_path: Path to the original dataset
            clahe_params: CLAHE parameter dictionary

        Returns:
            Path to the processed dataset
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CLAHE PREPROCESSING STAGE")
        self.logger.info("=" * 60)

        # Create temporary directory for CLAHE-processed data
        enhanced_dataset_dir = os.path.join(self.temp_dir, "clahe_enhanced_dataset")
        os.makedirs(enhanced_dataset_dir, exist_ok=True)

        # Copy dataset structure
        for subdir in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
            src_dir = os.path.join(dataset_path, subdir)
            if os.path.exists(src_dir):
                dst_dir = os.path.join(enhanced_dataset_dir, subdir)
                os.makedirs(dst_dir, exist_ok=True)

        # Copy dataset.json (if exists)
        dataset_json_src = os.path.join(dataset_path, "dataset.json")
        if os.path.exists(dataset_json_src):
            shutil.copy2(
                dataset_json_src, os.path.join(enhanced_dataset_dir, "dataset.json")
            )

        # Initialize CLAHE tool
        clahe_tool = CLAHE(logger=self.logger)

        # Process all image files
        for subdir in ["imagesTr", "imagesTs"]:
            src_dir = os.path.join(dataset_path, subdir)
            if not os.path.exists(src_dir):
                continue

            dst_dir = os.path.join(enhanced_dataset_dir, subdir)
            image_files = [f for f in os.listdir(src_dir) if f.endswith(".nii.gz")]

            self.logger.info(f"Processing {len(image_files)} images in {subdir}...")

            for img_file in tqdm(image_files, desc=f"CLAHE-{subdir}"):
                src_path = os.path.join(src_dir, img_file)
                dst_path = os.path.join(dst_dir, img_file)

                try:
                    clahe_tool.run_clahe(src_path, dst_path, params=clahe_params)
                except Exception as e:
                    self.logger.error(f"CLAHE failed on {img_file}: {e}")
                    # If processing fails, copy original file
                    shutil.copy2(src_path, dst_path)

        # Copy label files (labels are not processed)
        for subdir in ["labelsTr", "labelsTs"]:
            src_dir = os.path.join(dataset_path, subdir)
            if not os.path.exists(src_dir):
                continue

            dst_dir = os.path.join(enhanced_dataset_dir, subdir)
            label_files = [f for f in os.listdir(src_dir) if f.endswith(".nii.gz")]

            for lbl_file in label_files:
                shutil.copy2(
                    os.path.join(src_dir, lbl_file), os.path.join(dst_dir, lbl_file)
                )

        self.logger.info(f"CLAHE preprocessing completed → {enhanced_dataset_dir}")
        self.logger.info("=" * 60 + "\n")

        return enhanced_dataset_dir

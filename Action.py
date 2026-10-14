import os
import sys
import json
import shutil
import tempfile
from PIL import Image
import numpy as np
import nibabel as nib
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from nnUNet2BiomedParse import convert_nnunet_to_biomedparse, load_volume, save_volume
from Toolbox import (
    BiomedParsePredictor,
    BiomedParse2DPredictor,
    SAM3Wrapper,
    ZoomSAM3Refiner,
    TopologyCleaner,
    CLAHE,
    SymmetryChecker,
)
from tqdm import tqdm

SAM3_PYTHON_PATH = os.environ.get("SAM3_PYTHON", sys.executable)
SAM3_SCRIPT_PATH = "run_sam3_standalone.py"


class ActionModule:
    def __init__(
        self,
        result_dir: str,
        logger,
        biomedparse_checkpoint: str,
        sam3_checkpoint: str,
        temp_dir: Optional[str] = None,
        gpus: List[int] = None,
    ):
        self.result_dir = result_dir
        self.logger = logger
        self.gpus = [0] if gpus is None else list(gpus)
        if not self.gpus:
            raise ValueError("At least one GPU must be selected")

        if temp_dir is None:
            temp_dir = os.path.join(result_dir, "temp")
        self.temp_dir = temp_dir
        self.temp_root = temp_dir

        self.biomed_input_dir = os.path.join(self.temp_dir, "biomedparse_input_npz")
        self.pseudolabel_dir = os.path.join(result_dir, "Pseudolabels_NIfTI")

        os.makedirs(self.biomed_input_dir, exist_ok=True)
        os.makedirs(self.pseudolabel_dir, exist_ok=True)

        self.bp_predictor = BiomedParsePredictor(
            checkpoint_path=biomedparse_checkpoint, cuda_devices=[str(g) for g in self.gpus]
        )
        self.bp_2d_predictor = BiomedParse2DPredictor()

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
        original_dataset_path = dataset_path
        self.temp_dir = tempfile.mkdtemp(prefix="execution_", dir=self.temp_root)
        self.biomed_input_dir = os.path.join(self.temp_dir, "biomedparse_input_npz")
        target_organs = adaptation_plan.get("target_organs", [])
        modality = adaptation_plan.get("domain_context", {}).get("modality", "CT")
        prompts = adaptation_plan.get("text_prompts", {})

        if not prompts:
            raise ValueError("No text prompts found in adaptation plan!")

        # === Read tool selection from adaptation_plan ===
        strategy_recs = adaptation_plan.get("strategy_recommendations", {})

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

        adaptation_plan["working_dataset_path"] = os.path.abspath(dataset_path)
        self.pseudolabel_dir = tempfile.mkdtemp(prefix="Pseudolabels_", dir=self.result_dir)
        adaptation_plan["pseudolabel_dir"] = self.pseudolabel_dir
        final_pseudolabel_dir = None

        if primary_tool == "BiomedParse":
            self.logger.info(f"BiomedParse prompts: {prompts}")
            final_pseudolabel_dir = self._predict_biomedparse(dataset_path, prompts, self.pseudolabel_dir)

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
                original_dataset_path=original_dataset_path,
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
        dual_dir = os.path.join(self.temp_dir, "dual_expert")
        if os.path.exists(dual_dir):
            shutil.rmtree(dual_dir)
        bp_output_dir = os.path.join(dual_dir, "biomedparse_output")
        os.makedirs(bp_output_dir, exist_ok=True)

        self._predict_biomedparse(dataset_path, prompts, bp_output_dir)
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
            candidates = [Path(images_dir) / f"{case_name}_0000{ext}" for ext in (".nii.gz", ".png")]
            img_path = str(next((p for p in candidates if p.is_file()), candidates[0]))

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
    def _predict_biomedparse(self, dataset_path, prompts, output_dir):
        info = json.loads((Path(dataset_path) / "dataset.json").read_text())
        if info["file_ending"] == ".png":
            return self.bp_2d_predictor.predict(dataset_path, output_dir, prompts, self.gpus)
        npz_input_dir = self._prepare_data_for_biomedparse(dataset_path, prompts)
        return self.bp_predictor.predict(data_path=npz_input_dir, output_dir=output_dir,
                                         slice_batch_size=4, nms_threshold=0.5)

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

    def _generate_sam3_text_config(self, dataset_path, target_organs):
        images = Path(dataset_path) / "imagesTr"
        return {p.name.removesuffix(".nii.gz").removesuffix(".png").removesuffix("_0000"):
                {"image": str(p), "prompts": [{"obj_id": i, "text": organ, "prompt_type": "text"}
                                              for i, organ in enumerate(target_organs, 1)]}
                for p in sorted(images.iterdir()) if p.name.endswith(("_0000.nii.gz", "_0000.png"))}

    def _construct_nnunet_dataset(self, pseudolabel_dir, original_dataset_path, target_organs, modality):
        original = Path(original_dataset_path)
        source_dir = original / "imagesTr"
        with (original / "dataset.json").open() as stream:
            original_info = json.load(stream)
        ending = original_info["file_ending"]
        if ending not in {".nii.gz", ".png"}:
            raise ValueError(f"Unsupported dataset file ending: {ending}")
        if ending == ".nii.gz" and len(original_info.get("channel_names", {"0": modality})) != 1:
            raise ValueError("Supply the single target modality as channel _0000")
        images = sorted(source_dir.glob("*_0000" + ending))
        if not images:
            raise ValueError("No target training images")
        root = Path(os.environ.get("nnUNet_raw", str(original.parent)))
        root.mkdir(parents=True, exist_ok=True)
        ids = {int(p.name[7:10]) for p in root.glob("Dataset[0-9][0-9][0-9]_*")}
        dataset_id = next((i for i in range(1, 1000) if i not in ids), None)
        if dataset_id is None:
            raise ValueError("No free nnUNet dataset ID")
        output = root / f"Dataset{dataset_id:03d}_{original.name.split('_', 1)[-1]}_PARDA"
        (output / "imagesTr").mkdir(parents=True)
        (output / "labelsTr").mkdir()
        for image in images:
            case = image.name[:-len("_0000" + ending)]
            reference = load_volume(image)
            mask = load_volume(Path(pseudolabel_dir) / f"{case}.nii.gz")
            data = mask.get_fdata()
            self._validate_mask(data, reference, len(target_organs))
            if not np.allclose(mask.affine, reference.affine):
                raise ValueError(f"Pseudo-label geometry differs for {case}")
            save_volume(data.astype(np.uint16), reference, output / "labelsTr" / f"{case}{ending}")
            shutil.copy2(image, output / "imagesTr" / image.name)
        info = {"channel_names": original_info.get("channel_names", {"0": modality}),
                "labels": {"background": 0, **{organ: i for i, organ in enumerate(target_organs, 1)}},
                "numTraining": len(images), "file_ending": ending, "name": output.name}
        (output / "dataset.json").write_text(json.dumps(info, indent=2))
        return str(output)

    @staticmethod
    def _validate_mask(data, reference, num_labels):
        if data.shape != reference.shape or not np.isfinite(data).all():
            raise ValueError("Invalid pseudo-label shape or non-finite values")
        if not np.equal(data, np.rint(data)).all() or data.min() < 0 or data.max() > num_labels:
            raise ValueError("Pseudo-label contains invalid class IDs")

    def _resegment_case(self, image_path, label_path, output_path, target_map, proposal=None,
                        label_ids=None, margin=10):
        reference = load_volume(label_path)
        masks = reference.get_fdata().astype(np.uint16) if proposal is None else proposal
        selected = set(target_map) if label_ids is None else set(label_ids)
        prompts = []
        for label_id in sorted(selected):
            prompt = {"obj_id": label_id, "text": target_map[label_id], "prompt_type": "text"}
            box = self._extract_bbox_and_key_slice(masks == label_id, masks.shape)
            if box:
                axis = int(np.argmin(masks.shape))
                h, w = [masks.shape[i] for i in range(3) if i != axis]
                x, y, bw, bh = box["box"]
                x0, y0 = max(0, x - margin), max(0, y - margin)
                prompt.update(box=[x0, y0, min(w, x + bw + margin) - x0,
                                   min(h, y + bh + margin) - y0], key_slice=box["key_slice"])
            prompts.append(prompt)
        with tempfile.TemporaryDirectory(dir=self.temp_dir, prefix="resegment_") as work:
            config = Path(work) / "config.json"
            config.write_text(json.dumps({"case": {"image": str(image_path), "prompts": prompts}}))
            self.sam3_wrapper.predict(str(config), work, gpus=self.gpus, checkpoint=self.sam3_checkpoint)
            result = load_volume(Path(work) / "case.nii.gz")
            data = result.get_fdata()
            self._validate_mask(data, reference, max(target_map))
            if not np.allclose(result.affine, load_volume(image_path).affine):
                raise ValueError("SAM3 changed image geometry")
            original = reference.get_fdata().astype(np.uint16)
            writable = (original == 0) | np.isin(original, list(selected))
            original[np.isin(original, list(selected))] = 0
            replace = writable & np.isin(data, list(selected))
            original[replace] = data[replace].astype(np.uint16)
            save_volume(original, reference, output_path)

    def _refine_symmetry(self, image_path, label_path, output_path, target_map, pairs, margin):
        reference = load_volume(label_path)
        image = load_volume(image_path)
        labels = reference.get_fdata().astype(np.uint16)
        checker = SymmetryChecker(self.logger)
        axes = checker._detect_axes(labels.shape, reference.affine)
        flip_axis = axes["flip_axis"]
        # shortcut: foreground extent estimates the body midline, use anatomical registration for off-center/oblique scans.
        body = image.get_fdata() > np.percentile(image.get_fdata(), 10)
        coordinates = np.where(body)[flip_axis]
        if not coordinates.size:
            raise ValueError("Cannot estimate body midline")
        midline = (coordinates.min() + coordinates.max()) / 2
        proposal = labels.copy()
        from scipy.ndimage import affine_transform
        matrix, offset = np.eye(3), np.zeros(3)
        matrix[flip_axis, flip_axis] = -1
        offset[flip_axis] = 2 * midline
        candidates = []
        for pair in pairs:
            ids = list(pair.values())
            if len(ids) != 2 or any(i not in target_map for i in ids):
                raise ValueError("Invalid anatomical pair")
            sizes = [np.count_nonzero(labels == i) for i in ids]
            if max(sizes) == 0:
                continue
            metrics = checker._calculate_metrics(labels == ids[0], labels == ids[1], axes)
            agreement = metrics["volume_ratio"] * metrics["symmetry_dice_3d"] if metrics["valid_pair"] else 0
            candidates.append((agreement, ids, sizes))
        if not candidates:
            raise ValueError("No non-empty paired organ available for symmetry refinement")
        _, ids, sizes = min(candidates, key=lambda item: item[0])
        # Correct the least-consistent pair so other paired structures remain intact.
        source, target = (ids[0], ids[1]) if sizes[0] >= sizes[1] else (ids[1], ids[0])
        mirrored = affine_transform((labels == source).astype(np.uint8), matrix,
                                    offset=offset, order=0, mode="constant") > 0
        proposal[(mirrored & (labels == 0)) | (labels == target)] = target
        self._resegment_case(image_path, label_path, output_path, target_map,
                             proposal=proposal, label_ids=[target], margin=margin)

    def execute_refinement_actions(self, constructed_dataset_path, refinement_strategy, adaptation_plan):
        dataset = Path(constructed_dataset_path)
        info = json.loads((dataset / "dataset.json").read_text())
        ending = info["file_ending"]
        target_map = {i: organ for organ, i in info["labels"].items() if organ != "background"}
        strategy = adaptation_plan["strategy_recommendations"]
        working = Path(adaptation_plan.get("working_dataset_path", constructed_dataset_path)) / "imagesTr"
        outcomes = {}
        seen = set()
        for tool, action in refinement_strategy.get("actions", {}).items():
            for case in action["cases"]:
                if case in seen or Path(case).name != case:
                    raise ValueError("Each valid case must have exactly one refinement action")
                seen.add(case)
                label = dataset / "labelsTr" / f"{case}{ending}"
                image = working / f"{case}_0000{ending}"
                if not image.is_file():
                    image = dataset / "imagesTr" / f"{case}_0000{ending}"
                params = action.get("params", {})
                try:
                    with tempfile.TemporaryDirectory(dir=label.parent, prefix=".refine_") as work:
                        result = Path(work) / ("refined" + ending)
                        if tool == "ZoomSAM3":
                            coarse = load_volume(label)
                            if not np.any(coarse.get_fdata()):
                                self._resegment_case(image, label, result, target_map, margin=params.get("margin", 10))
                            else:
                                coarse_path = Path(work) / "coarse.nii.gz"
                                nib.save(coarse, coarse_path)
                                refined_path = Path(work) / "refined.nii.gz"
                                refiner = ZoomSAM3Refiner(SAM3_PYTHON_PATH, SAM3_SCRIPT_PATH,
                                    self.sam3_checkpoint, temp_root=work, logger=self.logger)
                                refiner.run(str(image), str(coarse_path), str(refined_path),
                                    target_map, margin=params.get("margin", 10), gpus=self.gpus)
                                save_volume(load_volume(refined_path).get_fdata().astype(np.uint16), coarse, result)
                        elif tool == "CLAHE":
                            enhanced = Path(work) / ("enhanced" + ending)
                            CLAHE(self.logger).run_clahe(str(image), str(enhanced), params)
                            self._resegment_case(enhanced, label, result, target_map, margin=10)
                        elif tool == "TopologyCleaner":
                            params = {**params, "multi_component_labels": strategy.get("multi_component_labels", {})}
                            TopologyCleaner(self.logger).run_cleaning(str(label), str(result), params)
                        elif tool == "SymmetryChecker":
                            self._refine_symmetry(image, label, result, target_map, strategy.get("organ_pairs", []), params.get("margin", 10))
                        else:
                            raise ValueError(f"Unsupported refinement tool: {tool}")
                        reference, refined = load_volume(label), load_volume(result)
                        data = refined.get_fdata()
                        self._validate_mask(data, reference, max(target_map))
                        if not np.allclose(reference.affine, refined.affine):
                            raise ValueError("Refinement changed label geometry")
                        os.replace(result, label)
                    outcomes[case] = {"tool": tool, "params": params, "status": "success"}
                except Exception as exc:
                    self.logger.exception(f"Refinement failed for {case} with {tool}")
                    outcomes[case] = {"tool": tool, "params": params, "status": "failed", "error": str(exc)}
        return outcomes

    def _apply_clahe_to_dataset(self, dataset_path, clahe_params):
        source = Path(dataset_path)
        enhanced = Path(self.temp_dir) / "clahe_enhanced_dataset"
        if enhanced.exists():
            shutil.rmtree(enhanced)
        (enhanced / "imagesTr").mkdir(parents=True)
        shutil.copy2(source / "dataset.json", enhanced / "dataset.json")
        for image in sorted((source / "imagesTr").iterdir()):
            if image.name.endswith(("_0000.nii.gz", "_0000.png")):
                CLAHE(self.logger).run_clahe(str(image), str(enhanced / "imagesTr" / image.name), clahe_params)
        return str(enhanced)

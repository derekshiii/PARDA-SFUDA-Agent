import os
import re
import json
import subprocess
import glob
import sys
from pathlib import Path
from typing import Optional, List


class FinetuningModule:
    def __init__(self, logger, result_dir: str, target_env_path: Optional[str] = None, gpus=None):
        """
        Args:
            logger: Logger instance
            result_dir: Result output directory
            target_env_path: External nnUNet environment path (optional)
        """
        self.logger = logger
        self.result_dir = result_dir
        self.target_env_path = target_env_path
        self.gpus = [0] if gpus is None else list(gpus)
        if not self.gpus:
            raise ValueError("At least one GPU must be selected")

        self.PLANNER_NAME = "ResEncUNetPlanner"  # Corresponds to -pl in script
        self.FIXED_PLANS_IDENTIFIER = (
            "nnUNetResEncUNetPlans"  # Corresponds to -p in script
        )
        self.FOLD = "all"  # Corresponds to fold_all in script

        self._check_environment()

        if self.target_env_path:
            self.logger.info(
                f"Using external nnUNet environment: {self.target_env_path}"
            )

    def _check_environment(self):
        """Check required nnUNet environment variables"""
        required_vars = ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]
        missing = [v for v in required_vars if v not in os.environ]
        if missing:
            self.logger.warning(
                f"Missing nnUNet environment variables: {missing}. "
                "Ensure these are exported in the shell running the Agent."
            )
        else:
            self.logger.info(f"nnUNet environment variables checked: OK")

    def _get_cmd_path(self, command_name: str) -> str:
        """Get command absolute path, intelligently handling bin directory"""
        if self.target_env_path:
            base_path = self.target_env_path.rstrip(os.path.sep)
            if base_path.endswith("bin"):
                cmd_path = os.path.join(base_path, command_name)
            else:
                cmd_path = os.path.join(base_path, "bin", command_name)

            if not os.path.exists(cmd_path):
                self.logger.warning(f"Executable NOT found at: {cmd_path}")
                self.logger.warning("Attempting to fall back to system PATH...")
                return command_name
            return cmd_path
        return command_name

    def _determine_config_from_dataset(self, dataset_id: int) -> str:
        """
        [Retained Logic] Read dataset.json and determine configuration based on file extension
        .nii.gz -> 3d_fullres
        .png/.jpg etc. -> 2d
        """
        nnunet_raw = os.environ.get("nnUNet_raw")
        if not nnunet_raw:
            self.logger.warning(
                "nnUNet_raw not set, cannot auto-detect config. Defaulting to 3d_fullres."
            )
            return "3d_fullres"

        # Find Dataset folder
        search_pattern = os.path.join(nnunet_raw, f"Dataset{dataset_id:03d}_*")
        candidates = glob.glob(search_pattern)

        if not candidates:
            self.logger.warning(
                f"Dataset folder for ID {dataset_id} not found. Defaulting to 3d_fullres."
            )
            return "3d_fullres"

        dataset_dir = candidates[0]
        json_path = os.path.join(dataset_dir, "dataset.json")

        if not os.path.exists(json_path):
            self.logger.warning(f"dataset.json not found. Defaulting to 3d_fullres.")
            return "3d_fullres"

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            file_ending = data.get("file_ending", "")
            self.logger.info(f"Detected file ending in dataset.json: '{file_ending}'")

            if ".nii.gz" in file_ending:
                return "3d_fullres"
            else:
                return "2d"

        except Exception as e:
            self.logger.error(
                f"Error reading dataset.json: {e}. Defaulting to 3d_fullres."
            )
            return "3d_fullres"

    def extract_dataset_id(self, dataset_path: str) -> int:
        dirname = os.path.basename(os.path.normpath(dataset_path))
        match = re.match(r"Dataset(\d{3})_", dirname)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Could not extract Dataset ID from path: {dataset_path}")

    def run_preprocessing(self, dataset_id: int, checkpoint_path=None) -> str:
        """
        Corresponds to Shell script: Step 1: Plan and preprocess fine-tuning dataset
        Command: nnUNetv2_plan_and_preprocess -d ID -c CONFIG -pl ResEncUNetPlanner
        """
        # 1. Auto-determine configuration (2d or 3d_fullres)
        auto_config = self._determine_config_from_dataset(dataset_id)
        source_config = None
        if checkpoint_path:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            initialization = checkpoint["init_args"]
            auto_config = initialization["configuration"]
            if auto_config not in {"2d", "3d_fullres"}:
                raise ValueError("Paper adaptation supports a 2d or 3d_fullres source model")
            source_config = initialization["plans"]["configurations"][auto_config]
            raw_dir = next(Path(os.environ["nnUNet_raw"]).glob(f"Dataset{dataset_id:03d}_*"))
            target_info = json.loads((raw_dir / "dataset.json").read_text())
            if len(initialization["dataset_json"]["channel_names"]) != len(target_info["channel_names"]):
                raise ValueError("Source and target input channel counts must match")
            source_ids = sorted(int(v) for v in initialization["dataset_json"]["labels"].values())
            target_ids = sorted(int(v) for v in target_info["labels"].values())
            if source_ids != target_ids:
                raise ValueError("Full source initialization requires identical source/target class IDs; preserve source label order")
            del checkpoint

        self.logger.info("-" * 50)
        self.logger.info(f"Starting Preprocessing (Shell Logic Step 1)")
        self.logger.info(f"Dataset ID: {dataset_id}")
        self.logger.info(f"Auto-detected Configuration: {auto_config}")
        self.logger.info(f"Fixed Planner: {self.PLANNER_NAME}")
        self.logger.info("-" * 50)

        python = self._get_cmd_path("python") if self.target_env_path else sys.executable

        cmd = [
            python,
            str(Path(__file__).with_name("nnunet_runtime.py")),
            "-d",
            str(dataset_id),
            "-c",
            auto_config,  # Auto-determined
            "-pl",
            self.PLANNER_NAME,  # Fixed: ResEncUNetPlanner
            "--verify_dataset_integrity",
        ]

        self._execute_command(cmd, "Preprocessing")

        if source_config is not None:
            directory = next(Path(os.environ["nnUNet_preprocessed"]).glob(f"Dataset{dataset_id:03d}_*"))
            path = directory / (self.FIXED_PLANS_IDENTIFIER + ".json")
            plans = json.loads(path.read_text())
            configuration = plans["configurations"][auto_config]
            # Keep target normalization/spacing, but build exactly the source network and compatible patches.
            for key in ("architecture", "patch_size", "batch_size"):
                configuration[key] = source_config[key]
            path.write_text(json.dumps(plans, indent=2))

        # Return configuration for training phase to use
        return auto_config

    def run_training(self, dataset_id: int, configuration: str, checkpoint_path: str):
        """Distill ACCEPT pseudo-labels for 300 epochs from source-domain weights."""
        self.logger.info("-" * 50)
        self.logger.info(f"Starting Finetuning (Shell Logic: run_finetuning)")
        self.logger.info(f"Dataset ID: {dataset_id}")
        self.logger.info(f"Configuration: {configuration}")
        self.logger.info(f"Fixed Fold: {self.FOLD}")
        self.logger.info(f"Fixed Plans: {self.FIXED_PLANS_IDENTIFIER}")
        self.logger.info(f"Pretrained Weights: {checkpoint_path}")
        self.logger.info("-" * 50)

        if not checkpoint_path or not Path(checkpoint_path).is_file():
            raise ValueError("An existing source checkpoint is required")
        python = self._get_cmd_path("python") if self.target_env_path else sys.executable

        cmd = [
            python,
            str(Path(__file__).with_name("train_student.py")),
            str(dataset_id),
            configuration,  # Passed from preprocessing or auto-determined
            self.FOLD,  # Fixed: all
            "-pretrained_weights",
            checkpoint_path,
            "-p",
            self.FIXED_PLANS_IDENTIFIER,  # Fixed: nnUNetResEncUNetPlans
            "-num_gpus",
            str(len(self.gpus)),
        ]

        self._execute_command(cmd, "Finetuning")

    def _execute_command(self, cmd: List[str], stage_name: str):
        self.logger.info(f"Executing: {' '.join(cmd)}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpus))
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=env,
            )
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.logger.info(f"[{stage_name}] {line}")
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            self.logger.info(f"{stage_name} completed successfully.")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"{stage_name} failed with return code {e.returncode}")
            raise
        except Exception as e:
            self.logger.error(f"{stage_name} execution error: {e}")
            raise

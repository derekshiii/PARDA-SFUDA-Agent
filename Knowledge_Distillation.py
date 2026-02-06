import os
import re
import json
import subprocess
import glob
from typing import Tuple, Optional, List
import torch


class FinetuningModule:
    def __init__(self, logger, result_dir: str, target_env_path: Optional[str] = None):
        """
        Args:
            logger: Logger instance
            result_dir: Result output directory
            target_env_path: External nnUNet environment path (optional)
        """
        self.logger = logger
        self.result_dir = result_dir
        self.target_env_path = target_env_path

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

    def run_preprocessing(self, dataset_id: int) -> str:
        """
        Corresponds to Shell script: Step 1: Plan and preprocess fine-tuning dataset
        Command: nnUNetv2_plan_and_preprocess -d ID -c CONFIG -pl ResEncUNetPlanner
        """
        # 1. Auto-determine configuration (2d or 3d_fullres)
        auto_config = self._determine_config_from_dataset(dataset_id)

        self.logger.info("-" * 50)
        self.logger.info(f"Starting Preprocessing (Shell Logic Step 1)")
        self.logger.info(f"Dataset ID: {dataset_id}")
        self.logger.info(f"Auto-detected Configuration: {auto_config}")
        self.logger.info(f"Fixed Planner: {self.PLANNER_NAME}")
        self.logger.info("-" * 50)

        cmd_executable = self._get_cmd_path("nnUNetv2_plan_and_preprocess")

        cmd = [
            cmd_executable,
            "-d",
            str(dataset_id),
            "-c",
            auto_config,  # Auto-determined
            "-pl",
            self.PLANNER_NAME,  # Fixed: ResEncUNetPlanner
            "--verify_dataset_integrity",
        ]

        self._execute_command(cmd, "Preprocessing")

        # Return configuration for training phase to use
        return auto_config

    def run_training(self, dataset_id: int, configuration: str, checkpoint_path: str):
        """
        Corresponds to Shell script: run_finetuning function
        Command: nnUNetv2_train ID CONFIG all -pretrained_weights CKPT -p nnUNetResEncUNetPlans -num_gpus 2
        """
        self.logger.info("-" * 50)
        self.logger.info(f"Starting Finetuning (Shell Logic: run_finetuning)")
        self.logger.info(f"Dataset ID: {dataset_id}")
        self.logger.info(f"Configuration: {configuration}")
        self.logger.info(f"Fixed Fold: {self.FOLD}")
        self.logger.info(f"Fixed Plans: {self.FIXED_PLANS_IDENTIFIER}")
        self.logger.info(f"Pretrained Weights: {checkpoint_path}")
        self.logger.info("-" * 50)

        cmd_executable = self._get_cmd_path("nnUNetv2_train")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
        else:
            self.logger.warning(
                "No CUDA devices found, defaulting to 1 GPU (or CPU if forced)."
            )
            gpu_count = 1

        cmd = [
            cmd_executable,
            str(dataset_id),
            configuration,  # Passed from preprocessing or auto-determined
            self.FOLD,  # Fixed: all
            "-pretrained_weights",
            checkpoint_path,
            "-p",
            self.FIXED_PLANS_IDENTIFIER,  # Fixed: nnUNetResEncUNetPlans
            "-num_gpus",
            str(
                gpu_count
            ),  # Keep original Python logic, use dual GPUs for acceleration
        ]

        self._execute_command(cmd, "Finetuning")

    def _execute_command(self, cmd: List[str], stage_name: str):
        self.logger.info(f"Executing: {' '.join(cmd)}")
        env = os.environ.copy()
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

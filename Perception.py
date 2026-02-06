import os
import random
import base64
import json
import numpy as np
import nibabel as nib
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Union, Tuple
from openai import OpenAI
from Prompts import PROMPT_REGISTRY
from datetime import datetime


class PerceptionModule:
    def __init__(self, client: OpenAI, result_dir: str, model: str, logger):
        self.client = client
        self.model = model
        self.logger = logger
        self.result_dir = result_dir
        self.data_info = {}  # Store data statistics

    def _normalize_image_data(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize image data to 0-255 uint8."""
        image_data = image_data.astype(np.float32)
        # Percentile clipping to remove outliers
        p_low, p_high = np.percentile(image_data, [1, 99])
        image_data = np.clip(image_data, p_low, p_high)

        if image_data.max() > image_data.min():
            image_data = (image_data - image_data.min()) / (
                image_data.max() - image_data.min()
            )
            image_data = (image_data * 255).astype(np.uint8)
        else:
            image_data = np.zeros_like(image_data, dtype=np.uint8)
        return image_data

    def _process_image_to_base64(self, img: Image.Image) -> str:
        """Resize and convert PIL Image to base64 string."""
        if img.size[0] < 512 or img.size[1] < 512:
            scale = max(512 / img.size[0], 512 / img.size[1])
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)

        buffered = BytesIO()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _create_grid_image(self, images: List[Image.Image]) -> Image.Image:
        """Create a 2x2 grid from up to 4 images."""
        if not images:
            raise ValueError("No images to create grid")

        # Use first image size as target
        target_size = images[0].size
        resized_images = []

        for img in images:
            if img.size != target_size:
                img = img.resize(target_size, Image.LANCZOS)
            resized_images.append(img)

        # Pad with blank images if less than 4
        while len(resized_images) < 4:
            blank = Image.new("RGB", target_size, (0, 0, 0))
            resized_images.append(blank)

        w, h = target_size
        grid = Image.new("RGB", (w * 2, h * 2))
        grid.paste(resized_images[0], (0, 0))
        grid.paste(resized_images[1], (w, 0))
        grid.paste(resized_images[2], (0, h))
        grid.paste(resized_images[3], (w, h))
        return grid

    def _extract_from_nifti(
        self, file_path: str, num_slices: int = 4
    ) -> List[Image.Image]:
        """
        Extract multiple axial slices from NIfTI file and return as PIL Images.

        Args:
            file_path: Path to NIfTI file
            num_slices: Number of slices to extract (default 4 for 2x2 grid)

        Returns:
            List of PIL Image objects
        """
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()

            shape = data.shape
            axial_axis = np.argmin(shape)
            total_slices = shape[axial_axis]

            # Sample from middle 80% of volume
            valid_indices = range(int(total_slices * 0.1), int(total_slices * 0.9))
            if not valid_indices:
                valid_indices = range(total_slices)

            # Sample evenly distributed slices
            sampled_indices = random.sample(
                list(valid_indices), min(num_slices, len(valid_indices))
            )
            sampled_indices.sort()  # Sort to maintain anatomical order

            images = []
            for idx in sampled_indices:
                if axial_axis == 0:
                    slice_data = data[idx, :, :]
                elif axial_axis == 1:
                    slice_data = data[:, idx, :]
                else:
                    slice_data = data[:, :, idx]

                norm_data = self._normalize_image_data(slice_data)
                img = Image.fromarray(norm_data)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                images.append(img)

            return images

        except Exception as e:
            self.logger.error(f"Error reading NIfTI {file_path}: {e}")
            return []

    def _extract_from_image(self, file_path: str) -> List[Image.Image]:
        """Load standard image file (PNG/JPG) and return as list."""
        try:
            img = Image.open(file_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return [img]
        except Exception as e:
            self.logger.error(f"Failed to load image {file_path}: {e}")
            return []

    def scan_and_sample(
        self, data_path: str, num_samples: int = 5, slices_per_case: int = 4
    ) -> Tuple[List[Dict], Dict]:
        """
        Scan directory and sample images. For each case, extract multiple slices and create grid.

        Args:
            data_path: Path to data directory or file
            num_samples: Number of cases to sample
            slices_per_case: Number of slices to extract per case (for grid creation)

        Returns:
            Tuple of (visual_samples, data_info)
        """
        path_obj = Path(data_path)
        supported_ext = {".nii", ".gz", ".png", ".jpg", ".jpeg"}
        all_files = []

        if path_obj.is_file():
            all_files.append(path_obj)
        else:
            for ext in supported_ext:
                all_files.extend(list(path_obj.rglob(f"*{ext}")))

        valid_files = [
            str(f)
            for f in all_files
            if "".join(f.suffixes) in [".nii.gz", ".nii"]
            or f.suffix in [".png", ".jpg", ".jpeg"]
        ]

        if not valid_files:
            raise FileNotFoundError(f"No valid medical images found in {data_path}")

        file_extensions = {}
        dimension_info = {"2D": 0, "3D": 0}

        for f in valid_files:
            if f.endswith(".nii.gz") or f.endswith(".nii"):
                ext = ".nii.gz" if f.endswith(".gz") else ".nii"
                dimension_info["3D"] += 1
            else:
                ext = Path(f).suffix.lower()
                dimension_info["2D"] += 1
            file_extensions[ext] = file_extensions.get(ext, 0) + 1

        primary_dimension = (
            "3D" if dimension_info["3D"] >= dimension_info["2D"] else "2D"
        )
        primary_extension = (
            max(file_extensions, key=file_extensions.get)
            if file_extensions
            else "Unknown"
        )

        self.data_info = {
            "input_dimension": primary_dimension,
            "primary_file_format": primary_extension,
            "file_format_distribution": file_extensions,
            "dimension_distribution": dimension_info,
            "total_files": len(valid_files),
            "slices_per_case": slices_per_case,
        }

        self.logger.info(f"Data Info: {self.data_info}")

        # Sample files
        sampled_files = random.sample(valid_files, min(num_samples, len(valid_files)))
        visual_samples = []

        for f in sampled_files:
            try:
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    # Extract multiple slices from 3D volume
                    images = self._extract_from_nifti(f, num_slices=slices_per_case)

                    if images:
                        # Create grid image from slices
                        grid_img = self._create_grid_image(images)
                        visual_samples.append(
                            {
                                "file": os.path.basename(f),
                                "type": "nifti_grid",
                                "num_slices": len(images),
                                "image_base64": self._process_image_to_base64(grid_img),
                            }
                        )
                        # self.logger.info(f"Created grid with {len(images)} slices from {os.path.basename(f)}")
                else:
                    # For 2D images, just use as-is
                    images = self._extract_from_image(f)
                    if images:
                        visual_samples.append(
                            {
                                "file": os.path.basename(f),
                                "type": "2d_image",
                                "num_slices": 1,
                                "image_base64": self._process_image_to_base64(
                                    images[0]
                                ),
                            }
                        )
                        self.logger.info(f"Loaded 2D image {os.path.basename(f)}")
            except Exception as e:
                self.logger.error(f"Failed to process {f}: {e}")
                continue

        if not visual_samples:
            raise RuntimeError("Failed to extract any valid visual samples")

        self.logger.info(f"Successfully created {len(visual_samples)} visual samples")
        return visual_samples, self.data_info

    def analyze_domain(
        self, raw_target_organs: List[str], visual_samples: List[Dict]
    ) -> Dict:
        """
        Call VLM to analyze modality, anatomy, AND specific tool strategies.
        Also corrects user input targets.
        """
        self.logger.info("Calling VLM for Domain Analysis & Strategy Extraction...")

        targets_formatted = "\n".join(
            [f"{i + 1}. {t}" for i, t in enumerate(raw_target_organs)]
        )

        user_prompt = f"""Here are the User Input Targets (Index. Name):
{targets_formatted}

I have provided {len(visual_samples)} case samples. Each 3D case is shown as a 2x2 grid of slices from different depths.
Please perform the Data Cleaning and Strategic Image Analysis as requested in the system prompt.
Output strictly in JSON."""

        content = [{"type": "text", "text": user_prompt}]

        for sample in visual_samples:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{sample['image_base64']}"
                    },
                }
            )

        messages = [
            {"role": "system", "content": PROMPT_REGISTRY["PERCEPTION_ANALYSIS"]},
            {"role": "user", "content": content},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.1, max_tokens=4096
            )
            res_text = response.choices[0].message.content

            # Enhanced JSON cleaning logic
            self.logger.debug(f"Raw response: {res_text[:200]}...")

            # Remove various possible markdown markers
            clean_json = res_text.strip()
            if clean_json.startswith("```"):
                lines = clean_json.split("\n")
                clean_json = "\n".join(lines[1:]) if len(lines) > 1 else clean_json
            clean_json = (
                clean_json.replace("```json", "")
                .replace("```JSON", "")
                .replace("```", "")
                .strip()
            )

            # Try to find the start and end of JSON object
            start_idx = clean_json.find("{")
            end_idx = clean_json.rfind("}")
            if start_idx != -1 and end_idx != -1:
                clean_json = clean_json[start_idx : end_idx + 1]

            self.logger.debug(f"Cleaned JSON: {clean_json[:200]}...")

            analysis_result = json.loads(clean_json)

            self.logger.info("VLM Analysis Successful.")
            self.logger.info(
                f"Corrected Targets: {analysis_result.get('corrected_targets')}"
            )
            self.logger.info(
                f"Strategy: {analysis_result.get('strategy_recommendations')}"
            )

            return analysis_result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON Decode Error: {e}")
            self.logger.error(
                f"Problematic text at position {e.pos}: {res_text[max(0, e.pos - 50) : e.pos + 50]}"
            )
            # Save raw response to file for debugging
            error_file = os.path.join(
                self.result_dir,
                f"vlm_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            )
            with open(error_file, "w") as f:
                f.write(res_text)
            self.logger.error(f"Raw response saved to {error_file}")

            # Return fallback
            return {
                "corrected_targets": raw_target_organs,
                "modality": "Unknown",
                "anatomical_site": "Unknown",
                "image_characteristics": {"contrast_rating": 5},
                "strategy_recommendations": {
                    "organ_pairs": [],
                    "multi_component_labels": {},
                    "requires_zoom_roi": False,
                },
            }
        except Exception as e:
            self.logger.error(f"VLM Analysis Failed: {e}")
            return {
                "corrected_targets": raw_target_organs,
                "modality": "Unknown",
                "anatomical_site": "Unknown",
                "image_characteristics": {"contrast_rating": 5},
                "strategy_recommendations": {
                    "organ_pairs": [],
                    "multi_component_labels": {},
                    "requires_zoom_roi": False,
                },
            }

    def construct_prompts(
        self, corrected_target_organs: List[str], domain_context: Dict
    ) -> Dict[str, str]:
        """
        Construct standardized BiomedParse prompts based on VLM analyzed context.
        Strategy: [Target] in [Site] [Modality]
        """
        body_part = domain_context.get("anatomical_site", "Body")
        modality = domain_context.get("modality", "Medical Image")

        prompts = {}
        for organ in corrected_target_organs:
            # VLM has already corrected targets, use them directly
            p = f"{organ} in {body_part} {modality}"
            prompts[organ] = p
            self.logger.info(f"Generated Prompt: {p}")

        self.generated_prompts = prompts
        return prompts

    def save_adaptation_plan(
        self, analysis_result: Dict, data_info: Dict = None
    ) -> str:
        """
        Save a simplified adaptation plan (without hardcoded execution_strategy).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adaptation_plan_{timestamp}.json"
        path = os.path.join(self.result_dir, filename)

        # Merge data_info
        if data_info is None and hasattr(self, "data_info"):
            data_info = self.data_info

        # Extract analysis results
        strategy = analysis_result.get("strategy_recommendations", {})
        img_chars = analysis_result.get("image_characteristics", {})

        # Simplified plan structure - only metadata, no execution strategy
        plan = {
            "timestamp": timestamp,
            "target_organs": analysis_result.get("corrected_targets", []),
            "domain_context": {
                "modality": analysis_result.get("modality"),
                "anatomical_site": analysis_result.get("anatomical_site"),
                "input_dimension": data_info.get("input_dimension"),
                "file_stats": data_info,
            },
            "image_analysis": img_chars,
            # Store prompts for use by Action module
            "text_prompts": self.generated_prompts,
            # Store recommendations that Action module can use
            "strategy_recommendations": {
                "use_clahe": img_chars.get("contrast_rating", 5) < 4,
                "clahe_params": {"clip_limit": 2.0},
                "use_zoom_roi": strategy.get("requires_zoom_roi", False),
                "zoom_roi_params": {"margin": 10},
                "multi_component_labels": strategy.get("multi_component_labels", {}),
                "organ_pairs": strategy.get("organ_pairs", []),
            },
        }

        with open(path, "w") as f:
            json.dump(plan, f, indent=2)
        self.logger.info(f"Adaptation plan saved to {path}")
        return path

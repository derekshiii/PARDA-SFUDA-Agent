import os
import json
import random
import base64
import numpy as np
import nibabel as nib
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback


class ReflectionModule:
    def __init__(self, client, result_dir: str, model: str, logger):
        """
        Initialize reflection module

        Args:
            client: OpenAI client (retained, but each thread creates its own client during concurrent execution)
            result_dir: Result directory
            model: VLM model name
            logger: Logger instance
        """
        self.client = client
        self.model = model
        self.logger = logger
        self.result_dir = result_dir

        self.reflection_images_dir = os.path.join(result_dir, "reflection_check_images")
        os.makedirs(self.reflection_images_dir, exist_ok=True)

        # For creating client in concurrent threads (avoid potential thread safety issues with shared client)
        # Some OpenAI-compatible SDK/gateways are unstable with reused sessions - single thread is fine, but concurrent execution can cause errors
        self._client_kwargs = {}
        try:
            # openai>=1.x OpenAI client typically has api_key/base_url attributes, but not guaranteed
            if hasattr(client, "api_key"):
                self._client_kwargs["api_key"] = getattr(client, "api_key")
            if hasattr(client, "base_url"):
                self._client_kwargs["base_url"] = getattr(client, "base_url")
        except Exception:
            pass

    # -----------------------------
    # Image utilities
    # -----------------------------
    def _normalize_image_data(self, image_data: np.ndarray) -> np.ndarray:
        """Normalize image data to 0-255 uint8"""
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

    def _get_boundary_palette(self, max_label: int) -> Dict[int, np.ndarray]:
        """
        Assign a "dark" boundary color (RGB) to each label to avoid confusion when all boundaries are yellow.

        Notes:
        - Using dark tones with high saturation, as distinct as possible from each other.
        - If label count exceeds preset colors, colors will be cycled (you can extend the palette).
        """
        palette = [
            (180, 30, 30),  # deep red
            (30, 160, 60),  # deep green
            (30, 80, 200),  # deep blue
            (140, 60, 170),  # deep purple
            (200, 120, 20),  # deep orange
            (20, 140, 140),  # deep teal
            (120, 120, 120),  # gray
            (180, 20, 120),  # magenta-ish
            (90, 60, 20),  # brown
            (20, 20, 20),  # near black
        ]
        out = {}
        for lbl in range(1, max_label + 1):
            out[lbl] = np.array(palette[(lbl - 1) % len(palette)], dtype=np.float32)
        return out

    def _extract_boundary(self, mask: np.ndarray, thickness: int = 1) -> np.ndarray:
        """
        Extract mask boundary (contour line)

        Args:
            mask: Binary mask
            thickness: Boundary line thickness (pixels)

        Returns:
            Boundary mask
        """
        from scipy.ndimage import binary_erosion

        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)

        eroded = binary_erosion(mask, iterations=thickness)
        boundary = mask & ~eroded
        return boundary

    def _create_multi_class_overlay(
        self,
        image_slice: np.ndarray,
        label_slice: np.ndarray,
        opacity: float = 0.2,
        boundary_thickness: int = 1,
    ) -> Image.Image:
        """
        Create label overlay visualization:
        - Interior uses unified cyan semi-transparent fill (consistent with prompt description)
        - Boundary lines are assigned dark colors per label (different colors for different classes) to avoid confusion at boundaries

        Args:
            image_slice: Original image slice
            label_slice: Label slice (multi-class, 0 is background)
            opacity: Label interior opacity
            boundary_thickness: Boundary line thickness
        """
        norm_image = self._normalize_image_data(image_slice)
        rgb_image = np.stack([norm_image] * 3, axis=-1)
        overlay = rgb_image.astype(np.float32)

        # Interior fill color: cyan (consistent with prompt)
        fill_color = np.array([0, 255, 255], dtype=np.float32)

        if label_slice is None:
            return Image.fromarray(rgb_image)

        # label_slice may be float (nibabel get_fdata), convert to int
        lbl = label_slice.astype(np.int32)

        max_label = int(lbl.max()) if lbl.size else 0
        if max_label <= 0:
            return Image.fromarray(rgb_image)

        # 1) Fill: all non-background regions with unified cyan semi-transparent
        mask_all = lbl > 0
        overlay[mask_all] = (1 - opacity) * overlay[mask_all] + opacity * fill_color

        # 2) Boundary: draw "completely opaque" boundary with independent color for each label
        boundary_colors = self._get_boundary_palette(max_label)

        # For clearer "inter-class boundaries": extract boundary for each class separately and colorize
        for k in range(1, max_label + 1):
            mk = lbl == k
            if not np.any(mk):
                continue
            bd = self._extract_boundary(mk, thickness=boundary_thickness)
            if np.any(bd):
                overlay[bd] = boundary_colors[k]  # 完全不透明深色边界

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)

    def _process_image_to_base64(self, img: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        if img.size[0] < 512 or img.size[1] < 512:
            scale = max(512 / img.size[0], 512 / img.size[1])
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.LANCZOS)

        buffered = BytesIO()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    # -----------------------------
    # Slice selection / loading
    # -----------------------------
    def _extract_mixed_slices(
        self,
        label_data: np.ndarray,
        axial_axis: int,
        num_top: int = 2,
        num_random: int = 2,
    ) -> List[int]:
        """Mixed selection strategy: first N slices selected by largest label area, last M slices randomly selected"""
        total_slices = label_data.shape[axial_axis]
        slice_areas = []

        for idx in range(total_slices):
            if axial_axis == 0:
                slice_label = label_data[idx, :, :]
            elif axial_axis == 1:
                slice_label = label_data[:, idx, :]
            else:
                slice_label = label_data[:, :, idx]

            area = np.sum(slice_label > 0)
            if area > 0:
                slice_areas.append((idx, area))

        if len(slice_areas) < num_top + num_random:
            self.logger.warning(
                f"Not enough slices with labels. Found {len(slice_areas)}, "
                f"requested {num_top + num_random}"
            )
            all_indices = [idx for idx, _ in slice_areas]
            all_indices.sort()
            return all_indices

        slice_areas.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in slice_areas[:num_top]]

        remaining_slices = slice_areas[num_top:]
        random_candidates = random.sample(
            remaining_slices, min(num_random, len(remaining_slices))
        )
        random_indices = [idx for idx, _ in random_candidates]

        top_indices.sort()
        random_indices.sort()

        return top_indices + random_indices

    def _load_case_data(
        self, image_path: str, label_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        img_nii = nib.load(image_path)
        lbl_nii = nib.load(label_path)

        img_data = img_nii.get_fdata()
        lbl_data = lbl_nii.get_fdata()
        return img_data, lbl_data

    def _get_axial_axis(self, shape: Tuple[int, ...]) -> int:
        """Heuristically determine axial axis (assumes shortest axis is axial)"""
        return int(np.argmin(shape))

    def _extract_case_name(self, image_filename: str) -> str:
        if image_filename.endswith("_0000.nii.gz"):
            return image_filename[: -len("_0000.nii.gz")]
        elif image_filename.endswith("_0000.nii"):
            return image_filename[: -len("_0000.nii")]
        else:
            return Path(image_filename).stem.replace("_0000", "")

    def _get_label_filename(self, case_name: str) -> str:
        return f"{case_name}.nii.gz"

    # -----------------------------
    # Grid image utilities
    # -----------------------------
    def _create_grid_image(self, images: List[Image.Image]) -> Image.Image:
        if not images:
            raise ValueError("No images to create grid")

        target_size = images[0].size
        resized_images = []
        for img in images:
            if img.size != target_size:
                img = img.resize(target_size, Image.LANCZOS)
            resized_images.append(img)

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

    def _save_check_image(
        self, grid_image: Image.Image, case_name: str, iteration: int = 0
    ) -> str:
        """
        Save check image, supports multiple iterations

        Args:
            grid_image: Image to save
            case_name: Case name
            iteration: Refinement iteration count (0=initial reflection, 1+=refinement iterations)
        """
        check_dir = getattr(self, "_current_check_dir", self.reflection_images_dir)

        # Name file based on iteration
        if iteration == 0:
            filename = f"{case_name}_check_initial.png"
        else:
            filename = f"{case_name}_check_iter{iteration}.png"

        save_path = os.path.join(check_dir, filename)
        grid_image.save(save_path, format="PNG", quality=95)
        self.logger.debug(f"Saved check image: {save_path}")
        return save_path

    # -----------------------------
    # VLM calling (with retry)
    # -----------------------------
    def _make_thread_client(self):
        """Create independent OpenAI client for each thread to reduce SDK/connection reuse issues in concurrent execution"""
        from openai import OpenAI

        # If kwargs unavailable, fall back to reusing self.client (not recommended)
        if self._client_kwargs:
            return OpenAI(**self._client_kwargs)
        return self.client

    def _call_vlm_critic(
        self,
        target_organs: List[str],
        slice_indices: List[int],
        grid_base64: str,
        case_name: str,
        retries: int = 2,
        retry_backoff_sec: float = 2.0,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> Dict:
        """
        Call VLM for quality evaluation (using 2x2 grid image), with retry/backoff.
        """
        from Prompts import PROMPT_REGISTRY

        organs_list = ", ".join(target_organs)
        user_prompt = f"""Case ID: {case_name}

Target Segmentation Task: {organs_list}

I am providing you with a 2x2 grid image containing 4 slices from this medical scan:

Grid Layout & Sampling Strategy:
- **Top-Left (Slice {slice_indices[0]})**: Largest labeled region #1
- **Top-Right (Slice {slice_indices[1]})**: Largest labeled region #2
- **Bottom-Left (Slice {slice_indices[2]})**: Randomly sampled slice
- **Bottom-Right (Slice {slice_indices[3]})**: Randomly sampled slice

The top row shows slices with the MOST labeled tissue, while the bottom row provides random samples to check consistency across the volume.

Visual Encoding:
- **Cyan semi-transparent overlay (20% opacity)**: Predicted target regions
- **Boundary lines (solid, 1px)**: Exact prediction boundary (colors may differ by class)

The boundary line represents the EXACT boundary of the AI prediction. Use this to assess boundary precision against visible anatomical edges.

**IMPORTANT**: Do NOT assume which specific organ each colored region represents. Focus on evaluating whether:

1. **Anatomical Plausibility**: Do the highlighted regions correspond to structures that COULD be the target organs ({organs_list})?

2. **Boundary Precision**:
   - Does the boundary line align with visible tissue interfaces?
   - Look for intensity gradients, organ edges, or texture changes
   - Minor misalignment (<5 pixels) is acceptable

3. **Consistency**: Do predictions in the top row (high density) and bottom row (random samples) show similar quality?

4. **False Positives**: Are there highlighted regions in anatomically impossible locations?

Provide a single overall decision: ACCEPT, REJECT, or UNCERTAIN."""

        messages = [
            {"role": "system", "content": PROMPT_REGISTRY["REFLECTION_CRITIC"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{grid_base64}"},
                    },
                ],
            },
        ]

        client = self._make_thread_client()

        last_err = None
        for attempt in range(1, retries + 2):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                res_text = response.choices[0].message.content or ""
                last_response = res_text  # Save raw response for debugging

                # ===== Enhanced JSON cleaning logic =====
                self.logger.debug(
                    f"[{case_name}] Raw response preview: {res_text[:150]}..."
                )

                # 1. Remove markdown code block markers
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

                # 2. Extract JSON object (find first { and last })
                start_idx = clean_json.find("{")
                end_idx = clean_json.rfind("}")

                if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                    raise ValueError(f"No valid JSON object found in response")

                clean_json = clean_json[start_idx : end_idx + 1]

                self.logger.debug(
                    f"[{case_name}] Cleaned JSON preview: {clean_json[:150]}..."
                )

                # 3. Try to parse
                result = json.loads(clean_json)

                # 4. Validate required fields
                if "final_decision" not in result:
                    self.logger.warning(
                        f"[{case_name}] Missing 'final_decision' in response, adding default"
                    )
                    result["final_decision"] = "UNCERTAIN"

                return result

            except json.JSONDecodeError as e:
                last_err = f"JSON Decode Error at position {e.pos}: {str(e)}"
                self.logger.warning(
                    f"[VLM_ERROR] case={case_name} attempt={attempt} {last_err}"
                )

                # Save error response to file
                if last_response:
                    error_file = os.path.join(
                        self.reflection_images_dir,
                        f"vlm_error_{case_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    )
                    try:
                        with open(error_file, "w", encoding="utf-8") as f:
                            f.write(f"Case: {case_name}\n")
                            f.write(f"Attempt: {attempt}\n")
                            f.write(f"Error: {last_err}\n")
                            f.write(f"\n{'=' * 60}\n")
                            f.write(f"Raw Response:\n{last_response}\n")
                            f.write(f"\n{'=' * 60}\n")
                            f.write(f"Cleaned JSON:\n{clean_json}\n")
                        self.logger.debug(f"Error response saved to: {error_file}")
                    except:
                        pass

                if attempt <= retries:
                    sleep_s = retry_backoff_sec * attempt
                    self.logger.debug(f"[{case_name}] Retrying after {sleep_s}s...")
                    time.sleep(sleep_s)
                    continue

            except ValueError as e:
                last_err = f"Value Error: {str(e)}"
                self.logger.warning(
                    f"[VLM_ERROR] case={case_name} attempt={attempt} {last_err}"
                )

                if attempt <= retries:
                    sleep_s = retry_backoff_sec * attempt
                    time.sleep(sleep_s)
                    continue

            except Exception as e:
                last_err = str(e)
                self.logger.warning(
                    f"[VLM_ERROR] case={case_name} attempt={attempt} err={last_err}"
                )
                self.logger.debug(traceback.format_exc())

                if attempt <= retries:
                    sleep_s = retry_backoff_sec * attempt
                    time.sleep(sleep_s)
                    continue

        # Return safe fallback after all retries fail
        self.logger.error(
            f"[VLM_FAILED] case={case_name} after {retries + 1} attempts. Last error: {last_err}"
        )

        return {
            "error": last_err or "Unknown error",
            "final_decision": "UNCERTAIN",
            "confidence": 0.0,
            "reasoning": f"VLM evaluation failed after {retries + 1} attempts",
            "suggested_remedy": {
                "tool": "TopologyCleaner",
                "reason": "Default fallback due to VLM error",
            },
        }

    # -----------------------------
    # Per-case evaluation (unchanged interface, thread-safe)
    # -----------------------------
    def evaluate_case(
        self,
        image_path: str,
        label_path: str,
        target_organs: List[str],
        organ_label_mapping: Dict[str, int],
        num_slices: int = 4,
        iteration: int = 0,
    ) -> Dict:
        """
        Evaluate single case segmentation quality
        (Note: This function will be called concurrently, avoid sharing mutable state)
        """
        case_name = self._extract_case_name(Path(image_path).name)
        self.logger.debug(f"Evaluating case: {case_name}")

        img_data, lbl_data = self._load_case_data(image_path, label_path)
        axial_axis = self._get_axial_axis(img_data.shape)

        slice_indices = self._extract_mixed_slices(
            lbl_data, axial_axis, num_top=2, num_random=2
        )

        if not slice_indices or len(slice_indices) < 2:
            self.logger.warning(f"Insufficient valid slices for {case_name}")
            return {
                "case_name": case_name,
                "error": "Insufficient slices with labels",
                "vlm_evaluation": {"final_decision": "REJECT", "confidence": 1.0},
                "vlm_evaluation_timestamp": datetime.now().isoformat(),
            }

        overlay_images_list = []
        for idx in slice_indices[:4]:
            if axial_axis == 0:
                img_slice = img_data[idx, :, :]
                lbl_slice = lbl_data[idx, :, :]
            elif axial_axis == 1:
                img_slice = img_data[:, idx, :]
                lbl_slice = lbl_data[:, idx, :]
            else:
                img_slice = img_data[:, :, idx]
                lbl_slice = lbl_data[:, :, idx]

            overlay = self._create_multi_class_overlay(
                img_slice, lbl_slice, opacity=0.2, boundary_thickness=1
            )
            overlay_images_list.append(overlay)

        grid_image = self._create_grid_image(overlay_images_list)
        saved_image_path = self._save_check_image(
            grid_image, case_name, iteration=iteration
        )
        grid_base64 = self._process_image_to_base64(grid_image)

        vlm_start_time = datetime.now()
        vlm_result = self._call_vlm_critic(
            target_organs, slice_indices, grid_base64, case_name
        )
        vlm_evaluation_timestamp = datetime.now()
        vlm_duration = (vlm_evaluation_timestamp - vlm_start_time).total_seconds()

        result = {
            "case_name": case_name,
            "target_organs": target_organs,
            "evaluated_slices": len(slice_indices),
            "slice_indices": slice_indices,
            "sampling_strategy": {
                "top_left": f"Largest area (slice {slice_indices[0]})",
                "top_right": f"2nd largest (slice {slice_indices[1]})"
                if len(slice_indices) > 1
                else "N/A",
                "bottom_left": f"Random (slice {slice_indices[2]})"
                if len(slice_indices) > 2
                else "N/A",
                "bottom_right": f"Random (slice {slice_indices[3]})"
                if len(slice_indices) > 3
                else "N/A",
            },
            "check_image_path": saved_image_path,
            "vlm_evaluation": vlm_result,
            "vlm_evaluation_timestamp": vlm_evaluation_timestamp.isoformat(),
            "vlm_evaluation_duration_seconds": round(vlm_duration, 2),
        }
        return result

    # -----------------------------
    # Dataset reflection (CONCURRENT)
    # -----------------------------
    def reflect_on_dataset(
        self,
        constructed_dataset_path: str,
        adaptation_plan_path: str,
        num_slices_per_case: int = 5,
        max_workers: int = 8,
        retries: int = 2,
        retry_backoff: float = 2.0,
        iteration: int = 0,
    ) -> Dict:
        """
        Perform reflection and quality control on constructed nnUNet dataset (concurrent version)

        Optimization notes:
        - reflection: saves current final state (including complete detailed_evaluations)
        - reflection_history: only saves key changes from each iteration (without detailed_evaluations)
        """
        reflection_start_time = datetime.now()

        self.logger.info("=" * 60)
        self.logger.info(
            f"Starting Reflection Phase (CONCURRENT) - Iteration {iteration}"
        )
        self.logger.info(f"Dataset: {constructed_dataset_path}")
        self.logger.info("=" * 60)

        dataset_name = self._extract_dataset_name(constructed_dataset_path)

        # Create independent subdirectory for each iteration
        if iteration == 0:
            dataset_check_dir = os.path.join(
                self.reflection_images_dir, dataset_name, "initial"
            )
        else:
            dataset_check_dir = os.path.join(
                self.reflection_images_dir, dataset_name, f"iteration_{iteration}"
            )

        os.makedirs(dataset_check_dir, exist_ok=True)
        self.logger.info(f"Check images will be saved to: {dataset_check_dir}")

        self._current_check_dir = dataset_check_dir

        with open(adaptation_plan_path, "r") as f:
            adaptation_plan = json.load(f)
        target_organs = adaptation_plan.get("target_organs", [])

        dataset_json_path = os.path.join(constructed_dataset_path, "dataset.json")
        with open(dataset_json_path, "r") as f:
            dataset_info = json.load(f)

        organ_label_mapping = {}
        labels_dict = dataset_info.get("labels", {})
        for organ_name, label_value in labels_dict.items():
            if organ_name != "background":
                organ_label_mapping[organ_name] = label_value

        self.logger.info(f"Organ-Label Mapping: {organ_label_mapping}")

        images_dir = os.path.join(constructed_dataset_path, "imagesTr")
        labels_dir = os.path.join(constructed_dataset_path, "labelsTr")
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            raise FileNotFoundError(
                f"imagesTr or labelsTr not found in {constructed_dataset_path}"
            )

        image_files = sorted(
            [f for f in os.listdir(images_dir) if f.endswith("_0000.nii.gz")]
        )
        if not image_files:
            self.logger.warning("No image files found for reflection")
            return {}

        self.logger.info(f"Found {len(image_files)} cases to evaluate")

        # Build task list
        tasks = []
        for img_file in image_files:
            case_name = self._extract_case_name(img_file)
            label_file = self._get_label_filename(case_name)

            img_path = os.path.join(images_dir, img_file)
            lbl_path = os.path.join(labels_dir, label_file)

            if not os.path.exists(lbl_path):
                self.logger.warning(
                    f"Label not found for {img_file}: {lbl_path}, skipping"
                )
                continue

            tasks.append((case_name, img_path, lbl_path))

        all_evaluations = []
        rejected_cases = []
        accepted_cases = []
        uncertain_cases = []

        # Concurrent execution
        def _run_one(case_name: str, img_path: str, lbl_path: str) -> Dict:
            return self.evaluate_case(
                img_path,
                lbl_path,
                target_organs,
                organ_label_mapping,
                num_slices_per_case,
                iteration=iteration,
            )

        self._vlm_retries = retries
        self._vlm_retry_backoff = retry_backoff

        self.logger.info(
            f"Submitting {len(tasks)} cases with ThreadPoolExecutor(max_workers={max_workers}) ..."
        )
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_run_one, case_name, img_path, lbl_path): case_name
                for case_name, img_path, lbl_path in tasks
            }

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Reflecting (concurrent)",
            ):
                case_name = futures[fut]
                try:
                    evaluation = fut.result()
                except Exception as e:
                    self.logger.error(f"[FUTURE_ERROR] case={case_name} err={e}")
                    self.logger.debug(traceback.format_exc())
                    evaluation = {
                        "case_name": case_name,
                        "error": str(e),
                        "vlm_evaluation": {
                            "final_decision": "UNCERTAIN",
                            "confidence": 0.0,
                        },
                        "vlm_evaluation_timestamp": datetime.now().isoformat(),
                    }

                decision = evaluation.get("vlm_evaluation", {}).get(
                    "final_decision", "UNCERTAIN"
                )

                if decision == "REJECT":
                    rejected_cases.append(case_name)
                    all_evaluations.append(evaluation)
                elif decision == "UNCERTAIN":
                    uncertain_cases.append(case_name)
                    all_evaluations.append(evaluation)
                else:
                    accepted_cases.append(case_name)
                    all_evaluations.append(evaluation)

        reflection_end_time = datetime.now()
        reflection_duration = (
            reflection_end_time - reflection_start_time
        ).total_seconds()

        # Build complete summary for current iteration
        reflection_summary = {
            "dataset_name": dataset_name,
            "check_images_directory": dataset_check_dir,
            "iteration": iteration,
            "total_evaluated": len(all_evaluations),
            "accepted": len(accepted_cases),
            "rejected": len(rejected_cases),
            "uncertain": len(uncertain_cases),
            "acceptance_rate": len(accepted_cases) / len(all_evaluations)
            if all_evaluations
            else 0,
            "rejection_rate": len(rejected_cases) / len(all_evaluations)
            if all_evaluations
            else 0,
            "accepted_cases": accepted_cases,
            "rejected_cases": rejected_cases,
            "uncertain_cases": uncertain_cases,
            "detailed_evaluations": all_evaluations,
            "reflection_start_time": reflection_start_time.isoformat(),
            "reflection_end_time": reflection_end_time.isoformat(),
            "total_reflection_duration_seconds": round(reflection_duration, 2),
            "concurrency": {
                "max_workers": max_workers,
                "retries": retries,
                "retry_backoff": retry_backoff,
            },
        }

        # Generate refinement strategy
        refinement_strategy = self.generate_refinement_plan(
            reflection_summary=reflection_summary, threshold_rejection_rate=0.05
        )

        # === Key modification: separate saving of reflection and reflection_history ===
        try:
            with open(adaptation_plan_path, "r") as f:
                full_plan = json.load(f)

            # 1. Update reflection (complete current state, including detailed evaluations)
            full_plan["reflection"] = reflection_summary

            # 2. Update current_refinement_plan
            full_plan["current_refinement_plan"] = refinement_strategy

            # 3. Add lightweight history record (without detailed_evaluations)
            if "reflection_history" not in full_plan:
                full_plan["reflection_history"] = []

            history_entry = {
                "iteration": iteration,
                "timestamp": reflection_end_time.isoformat(),
                "summary": {
                    "total_evaluated": len(all_evaluations),
                    "accepted": len(accepted_cases),
                    "rejected": len(rejected_cases),
                    "uncertain": len(uncertain_cases),
                    "acceptance_rate": round(reflection_summary["acceptance_rate"], 4),
                    "rejection_rate": round(reflection_summary["rejection_rate"], 4),
                    "duration_seconds": round(reflection_duration, 2),
                    # Only record case name lists, not detailed evaluations
                    "rejected_case_names": rejected_cases,
                    "uncertain_case_names": uncertain_cases,
                },
                "refinement_strategy_summary": {
                    "stop_refinement": refinement_strategy.get(
                        "stop_refinement", False
                    ),
                    "reason": refinement_strategy.get("reason", ""),
                    "actions_count": len(refinement_strategy.get("actions", {})),
                },
            }

            full_plan["reflection_history"].append(history_entry)

            with open(adaptation_plan_path, "w") as f:
                json.dump(full_plan, f, indent=2)

            self.logger.info(
                f"Updated adaptation plan: iteration {iteration} - reflection updated, history appended"
            )

        except Exception as e:
            self.logger.error(f"Failed to update adaptation plan JSON: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

        # Update dataset.json
        self._update_dataset_json(
            constructed_dataset_path,
            len(reflection_summary["accepted_cases"])
            + len(reflection_summary["uncertain_cases"]),
        )

        if hasattr(self, "_current_check_dir"):
            delattr(self, "_current_check_dir")

        reflection_summary["refinement_strategy"] = refinement_strategy

        # Log output
        if refinement_strategy.get("stop_refinement", False):
            self.logger.info(
                "[STOP_REFINEMENT] Quality threshold met. Agent will handle final cleanup."
            )
        else:
            self.logger.info("[CONTINUE_REFINEMENT] Need more refinement iterations.")

        return reflection_summary

    # -----------------------------
    # File operations / plan update
    # -----------------------------
    def _remove_rejected_cases(
        self, rejected_cases: List[str], images_dir: str, labels_dir: str
    ):
        self.logger.info(f"Removing {len(rejected_cases)} rejected cases...")

        for case_name in rejected_cases:
            img_file = f"{case_name}_0000.nii.gz"
            lbl_file = self._get_label_filename(case_name)

            img_path = os.path.join(images_dir, img_file)
            lbl_path = os.path.join(labels_dir, lbl_file)

            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
                    self.logger.debug(f"Removed image: {img_file}")

                if os.path.exists(lbl_path):
                    os.remove(lbl_path)
                    self.logger.debug(f"Removed label: {lbl_file}")

            except Exception as e:
                self.logger.error(f"Failed to remove {case_name}: {e}")

    def _update_adaptation_plan(
        self, adaptation_plan_path: str, reflection_summary: Dict
    ):
        try:
            with open(adaptation_plan_path, "r") as f:
                plan = json.load(f)

            plan["reflection"] = reflection_summary

            with open(adaptation_plan_path, "w") as f:
                json.dump(plan, f, indent=2)

            self.logger.info("Updated adaptation plan with reflection results")

        except Exception as e:
            self.logger.error(f"Failed to update adaptation plan: {e}")

    def _update_dataset_json(self, dataset_path: str, new_num_training: int):
        dataset_json_path = os.path.join(dataset_path, "dataset.json")

        try:
            with open(dataset_json_path, "r") as f:
                dataset_info = json.load(f)

            old_num = dataset_info.get("numTraining", 0)
            dataset_info["numTraining"] = new_num_training

            with open(dataset_json_path, "w") as f:
                json.dump(dataset_info, f, indent=4)

            self.logger.info(
                f"Updated dataset.json: numTraining {old_num} -> {new_num_training}"
            )

        except Exception as e:
            self.logger.error(f"Failed to update dataset.json: {e}")

    def _extract_dataset_name(self, dataset_path: str) -> str:
        return Path(dataset_path).name

    def generate_refinement_plan(
        self, reflection_summary: Dict, threshold_rejection_rate: float = 0.05
    ) -> Dict:
        """
        Generate Refinement strategy dictionary based on Reflection results (does not save file directly).
        """
        self.logger.info("Generating Refinement Strategy...")

        rejection_rate = reflection_summary["rejection_rate"]
        uncertain_rate = (
            reflection_summary["uncertain"] / reflection_summary["total_evaluated"]
            if reflection_summary["total_evaluated"] > 0
            else 0
        )
        total_bad_rate = rejection_rate + uncertain_rate

        # Base structure
        refinement_strategy = {
            "generated_at": datetime.now().isoformat(),
            "stop_refinement": False,
            "reason": "",
            "actions": {},
        }

        # 1. Check stop condition
        if total_bad_rate < threshold_rejection_rate:
            self.logger.info(
                f"Bad case rate ({total_bad_rate:.1%}) is low. Advising stop."
            )
            refinement_strategy["stop_refinement"] = True
            refinement_strategy["reason"] = (
                f"Quality threshold met (Bad rate {total_bad_rate:.1%} < {threshold_rejection_rate:.1%})"
            )
            return refinement_strategy

        # 2. Aggregate tool allocation
        tool_allocation = {
            "ZoomSAM3": [],
            "CLAHE": [],
            "TopologyCleaner": [],
            "SymmetryChecker": [],
        }
        default_tool = "TopologyCleaner"

        detailed_evals = reflection_summary.get("detailed_evaluations", [])
        for ev in detailed_evals:
            decision = ev.get("vlm_evaluation", {}).get(
                "final_decision", "ACCEPT"
            )  # Default to ACCEPT as fallback

            if decision in ["REJECT", "UNCERTAIN"]:
                case_name = ev["case_name"]
                remedy = ev.get("vlm_evaluation", {}).get("suggested_remedy", {})
                tool_name = remedy.get("tool", "None")

                # Match tool
                valid_tools = [
                    "ZoomSAM3",
                    "CLAHE",
                    "TopologyCleaner",
                    "SymmetryChecker",
                ]
                match = next(
                    (t for t in valid_tools if t.lower() in str(tool_name).lower()),
                    None,
                )

                if match:
                    tool_allocation[match].append(case_name)
                else:
                    tool_allocation[default_tool].append(case_name)

        # 3. Generate action parameters
        if tool_allocation["CLAHE"]:
            refinement_strategy["actions"]["CLAHE"] = {
                "cases": tool_allocation["CLAHE"],
                "params": {"clip_limit": 3.0, "tile_grid_size": [8, 8]},
            }

        if tool_allocation["ZoomSAM3"]:
            refinement_strategy["actions"]["ZoomSAM3"] = {
                "cases": tool_allocation["ZoomSAM3"],
                "params": {"margin": 10},
            }

        if tool_allocation["TopologyCleaner"]:
            refinement_strategy["actions"]["TopologyCleaner"] = {
                "cases": tool_allocation["TopologyCleaner"],
                "params": {"min_size": 100, "keep_top_k": 1, "mode": "3d"},
            }

        if tool_allocation["SymmetryChecker"]:
            refinement_strategy["actions"]["SymmetryChecker"] = {
                "cases": tool_allocation["SymmetryChecker"],
                "params": {"vol_threshold": 0.2},
            }

        # If there are bad cases but no actions assigned (edge case), also advise stopping
        if not refinement_strategy["actions"]:
            refinement_strategy["stop_refinement"] = True
            refinement_strategy["reason"] = "No suitable refinement actions generated."

        return refinement_strategy

    def reflect_specific_cases(
        self,
        constructed_dataset_path: str,
        case_names: List[str],
        target_organs: List[str],
        organ_label_mapping: Dict[str, int],
        num_slices: int = 4,
        max_workers: int = 8,
        iteration: int = 1,
    ) -> Dict:
        """
        Only reflect on specified cases (used for re-evaluation after refinement)

        Args:
            case_names: List of case names to evaluate (without suffix)

        Returns:
            Dictionary containing evaluation results for these cases
        """
        self.logger.info(f"Re-reflecting {len(case_names)} specific cases...")

        images_dir = os.path.join(constructed_dataset_path, "imagesTr")
        labels_dir = os.path.join(constructed_dataset_path, "labelsTr")

        dataset_name = self._extract_dataset_name(constructed_dataset_path)
        dataset_check_dir = os.path.join(
            self.reflection_images_dir, dataset_name, f"iteration_{iteration}"
        )
        os.makedirs(dataset_check_dir, exist_ok=True)
        self._current_check_dir = dataset_check_dir

        # Build task list (only process specified cases)
        tasks = []
        for case_name in case_names:
            img_file = f"{case_name}_0000.nii.gz"
            label_file = f"{case_name}.nii.gz"

            img_path = os.path.join(images_dir, img_file)
            lbl_path = os.path.join(labels_dir, label_file)

            if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                self.logger.warning(f"Files not found for {case_name}, skipping")
                continue

            tasks.append((case_name, img_path, lbl_path))

        if not tasks:
            self.logger.warning("No valid cases to re-reflect")
            return {"evaluations": [], "cases_evaluated": 0}

        # Concurrent execution of evaluation (reusing existing logic)
        all_evaluations = []

        def _run_one(case_name: str, img_path: str, lbl_path: str) -> Dict:
            return self.evaluate_case(
                img_path,
                lbl_path,
                target_organs,
                organ_label_mapping,
                num_slices,
                iteration=iteration,
            )

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_run_one, cn, ip, lp): cn for cn, ip, lp in tasks}

            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Re-reflecting"
            ):
                case_name = futures[fut]
                try:
                    evaluation = fut.result()
                    all_evaluations.append(evaluation)
                except Exception as e:
                    self.logger.error(f"Re-reflection failed for {case_name}: {e}")
                    self.logger.debug(traceback.format_exc())

        return {
            "evaluations": all_evaluations,
            "cases_evaluated": len(all_evaluations),
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
        }

    def merge_reflection_results(
        self, adaptation_plan_path: str, new_results: Dict
    ) -> None:
        """
        Merge re-reflection results back into adaptation plan

        Optimization notes:
        - Only update reflection (current final state, including complete detailed_evaluations)
        - Do not add to reflection_history (history records only added during complete reflection)
        """
        self.logger.info("Merging re-reflection results into adaptation plan...")

        with open(adaptation_plan_path, "r") as f:
            plan = json.load(f)

        old_reflection = plan.get("reflection", {})
        old_detailed = old_reflection.get("detailed_evaluations", [])

        # Create case_name -> evaluation mapping (avoid duplicates)
        eval_map = {}

        # First load all old evaluations
        for ev in old_detailed:
            case_name = ev["case_name"]
            eval_map[case_name] = ev

        # Override with new results (only update re-evaluated cases)
        new_evals = new_results.get("evaluations", [])
        updated_cases = []
        for new_ev in new_evals:
            case_name = new_ev["case_name"]
            old_decision = (
                eval_map.get(case_name, {})
                .get("vlm_evaluation", {})
                .get("final_decision", "UNKNOWN")
            )
            new_decision = new_ev.get("vlm_evaluation", {}).get(
                "final_decision", "UNKNOWN"
            )

            eval_map[case_name] = new_ev  # Override old
            updated_cases.append(case_name)

            if old_decision != new_decision:
                self.logger.info(
                    f"Updated evaluation for {case_name}: {old_decision} -> {new_decision}"
                )
            else:
                self.logger.debug(f"Re-evaluated {case_name}: still {new_decision}")

        # Rebuild detailed_evaluations (no duplicates)
        updated_detailed = list(eval_map.values())

        # Re-classify
        accepted = []
        rejected = []
        uncertain = []

        for ev in updated_detailed:
            decision = ev.get("vlm_evaluation", {}).get("final_decision", "UNCERTAIN")
            case_name = ev["case_name"]

            if decision == "ACCEPT":
                accepted.append(case_name)
            elif decision == "REJECT":
                rejected.append(case_name)
            else:
                uncertain.append(case_name)

        total = len(updated_detailed)

        # Update reflection summary (override old state, preserve some metadata)
        updated_reflection = {
            "dataset_name": old_reflection.get("dataset_name", ""),
            "check_images_directory": old_reflection.get("check_images_directory", ""),
            "iteration": new_results.get(
                "iteration", old_reflection.get("iteration", 0)
            ),
            "total_evaluated": total,
            "accepted": len(accepted),
            "rejected": len(rejected),
            "uncertain": len(uncertain),
            "accepted_cases": sorted(accepted),
            "rejected_cases": sorted(rejected),
            "uncertain_cases": sorted(uncertain),
            "acceptance_rate": len(accepted) / total if total > 0 else 0,
            "rejection_rate": len(rejected) / total if total > 0 else 0,
            "detailed_evaluations": updated_detailed,
            "reflection_start_time": old_reflection.get("reflection_start_time", ""),
            "reflection_end_time": datetime.now().isoformat(),
            "total_reflection_duration_seconds": old_reflection.get(
                "total_reflection_duration_seconds", 0
            ),
            "concurrency": old_reflection.get("concurrency", {}),
            "last_merged": datetime.now().isoformat(),
            "merge_info": {
                "updated_cases": updated_cases,
                "updated_count": len(updated_cases),
            },
        }

        plan["reflection"] = updated_reflection

        # Regenerate refinement strategy
        new_refinement_strategy = self.generate_refinement_plan(
            reflection_summary=updated_reflection, threshold_rejection_rate=0.05
        )

        plan["current_refinement_plan"] = new_refinement_strategy

        # Save
        with open(adaptation_plan_path, "w") as f:
            json.dump(plan, f, indent=2)

        self.logger.info("=" * 60)
        self.logger.info(f"Merge Summary:")
        self.logger.info(f"  Total Cases:      {total}")
        self.logger.info(
            f"  Accepted:         {len(accepted)} ({len(accepted) / total * 100:.1f}%)"
        )
        self.logger.info(
            f"  Rejected:         {len(rejected)} ({len(rejected) / total * 100:.1f}%)"
        )
        self.logger.info(
            f"  Uncertain:        {len(uncertain)} ({len(uncertain) / total * 100:.1f}%)"
        )
        self.logger.info(f"  Updated Cases:    {len(updated_cases)}")
        self.logger.info(
            f"  Stop Refinement:  {new_refinement_strategy.get('stop_refinement', False)}"
        )
        self.logger.info("=" * 60)

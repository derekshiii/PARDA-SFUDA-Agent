import os
import json
import random
import base64
import colorsys
import numpy as np
import nibabel as nib
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from nnUNet2BiomedParse import load_volume

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
        - Extra colors depend only on label ID so legends match slices with absent classes.
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
            color = (palette[lbl - 1] if lbl <= len(palette) else
                     tuple(255 * v for v in colorsys.hsv_to_rgb((lbl * 0.61803398875) % 1, 0.75, 0.7)))
            out[lbl] = np.array(color, dtype=np.float32)
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
        rgb_image = norm_image if norm_image.ndim == 3 else np.stack([norm_image] * 3, axis=-1)
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
    def _extract_mixed_slices(self, label_data, axial_axis, num_top=2, num_random=2):
        areas = np.count_nonzero(label_data, axis=tuple(i for i in range(3) if i != axial_axis))
        count = min(num_top + num_random, len(areas))
        top = sorted(range(len(areas)), key=lambda i: (-areas[i], i))[:min(num_top, count)]
        remaining = [i for i in range(len(areas)) if i not in top]
        return top + sorted(random.sample(remaining, count - len(top)))

    def _load_case_data(self, image_path, label_path):
        image = load_volume(image_path)
        label = load_volume(label_path)
        if image.shape != label.shape or not np.allclose(image.affine, label.affine):
            raise ValueError("Image and pseudo-label must share shape and affine")
        img_data = image.get_fdata()
        if str(image_path).endswith(".png"):
            with Image.open(image_path) as png:
                img_data = np.asarray(png.convert("RGB"))
        return img_data, label.get_fdata()

    def _get_axial_axis(self, shape: Tuple[int, ...]) -> int:
        """Heuristically determine axial axis (assumes shortest axis is axial)"""
        return int(np.argmin(shape))

    def _extract_case_name(self, image_filename: str) -> str:
        if image_filename.endswith("_0000.nii.gz"):
            return image_filename[: -len("_0000.nii.gz")]
        elif image_filename.endswith("_0000.nii"):
            return image_filename[: -len("_0000.nii")]
        else:
            return Path(image_filename).stem.removesuffix("_0000")

    def _get_label_filename(self, case_name: str) -> str:
        return f"{case_name}.nii.gz"

    # -----------------------------
    # Grid image utilities
    # -----------------------------
    def _create_grid_image(self, images: List[Image.Image]) -> Image.Image:
        if not images:
            raise ValueError("No images to create grid")

        if len(images) == 1:
            return images[0]
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

    @staticmethod
    def normalize_decision(value):
        return "ACCEPT" if str(value).strip().upper() == "ACCEPT" else "REJECT"

    @classmethod
    def _validate_critic_result(cls, result):
        if not isinstance(result, dict):
            raise ValueError("Critic response must be a JSON object")
        decision = str(result.get("final_decision", "")).strip().upper()
        if decision not in {"ACCEPT", "REJECT", "UNCERTAIN"}:
            raise ValueError("Missing or invalid critic decision")
        result["final_decision"] = cls.normalize_decision(decision)
        for field in ("scores", "reasoning", "suggested_remedy"):
            if not isinstance(result.get(field), dict):
                raise ValueError(f"Critic {field} must be a JSON object")
        criteria = ("anatomical_plausibility", "boundary_precision", "morphological_consistency", "false_positives")
        for key in criteria:
            score = result["scores"].get(key)
            explanation = result["reasoning"].get(key)
            if type(score) is not int or not 1 <= score <= 10:
                raise ValueError(f"Invalid 1-10 score for {key}")
            if not isinstance(explanation, str) or not explanation.strip():
                raise ValueError(f"Missing assessment for {key}")
        remedy = result["suggested_remedy"]
        aliases = {"Zoom": "ZoomSAM3", "Topology": "TopologyCleaner", "Symmetry": "SymmetryChecker"}
        tool = remedy.get("tool")
        if not isinstance(tool, str):
            raise ValueError("Remediation tool must be a string")
        tool = aliases.get(tool, tool)
        if result["final_decision"] == "REJECT":
            if tool not in {"ZoomSAM3", "CLAHE", "TopologyCleaner", "SymmetryChecker"}:
                raise ValueError("REJECT requires a valid remediation tool")
            if not isinstance(remedy.get("rationale"), str) or not remedy["rationale"].strip():
                raise ValueError("REJECT requires an observable remediation rationale")
        result["suggested_remedy"] = {**remedy, "tool": tool if result["final_decision"] == "REJECT" else "None"}
        return result

    def _call_vlm_critic(self, target_organs, organ_label_mapping, slice_indices, grid_base64, case_name,
                         retries=2, retry_backoff_sec=2.0, temperature=0.1, max_tokens=4096):
        from Prompts import PROMPT_REGISTRY
        panels = ", ".join(str(i) for i in slice_indices)
        palette = self._get_boundary_palette(max(organ_label_mapping.values()))
        legend = "; ".join(f"label {organ_label_mapping[organ]} = {organ}, "
                           f"contour RGB {tuple(int(v) for v in palette[organ_label_mapping[organ]])}"
                           for organ in target_organs)
        prompt = f"""Case: {case_name}. Target organs: {', '.join(target_organs)}.
The image contains {len(slice_indices)} observed slice(s), indices: {panels}.
With multiple panels they are ordered top-left, top-right, bottom-left, bottom-right;
unused panels are blank. The first two observations have the largest predicted areas,
and remaining observations sample other depths, including slices without predictions.
Cyan fill (20% opacity) shows predicted tissue, dark contours show class boundaries.
Class legend: {legend}.
Provide four explanations and four integer scores (1-10), then ACCEPT or REJECT.
Insufficient evidence or uncertainty belongs to REJECT and requires one remediation tool.
"""
        messages = [{"role": "system", "content": PROMPT_REGISTRY["REFLECTION_CRITIC"]},
                    {"role": "user", "content": [{"type": "text", "text": prompt},
                     {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{grid_base64}"}}]}]
        client = self._make_thread_client()
        error = ""
        for attempt in range(retries + 1):
            try:
                response = client.chat.completions.create(model=self.model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens)
                raw = response.choices[0].message.content or ""
                start, end = raw.find("{"), raw.rfind("}")
                if start < 0 or end < start:
                    raise ValueError("Critic returned no JSON object")
                return self._validate_critic_result(json.loads(raw[start:end + 1]))
            except Exception as exc:
                error = str(exc)
                self.logger.warning(f"Critic failed for {case_name}, attempt {attempt + 1}: {error}")
                if attempt < retries:
                    time.sleep(retry_backoff_sec * (attempt + 1))
        return {"final_decision": "REJECT", "error": error,
                "suggested_remedy": {"tool": "None", "rationale": "Technical assessment failure"}}

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
        label_ids = list(organ_label_mapping.values())
        if (not label_ids or any(type(i) is not int or i <= 0 for i in label_ids)
                or len(set(label_ids)) != len(label_ids)):
            raise ValueError("Organ label IDs must be distinct positive integers")
        if (not np.all(np.isfinite(lbl_data)) or not np.all(lbl_data == np.floor(lbl_data))
                or not set(np.unique(lbl_data)).issubset({0, *label_ids})):
            raise ValueError("Pseudo-label contains non-finite, fractional or unknown class IDs")
        axial_axis = self._get_axial_axis(lbl_data.shape)

        slice_indices = self._extract_mixed_slices(
            lbl_data, axial_axis, num_top=2, num_random=2
        )

        overlay_images_list = []
        for idx in slice_indices[:4]:
            if str(image_path).endswith(".png"):
                img_slice, lbl_slice = img_data, lbl_data[..., 0]
            else:
                img_slice = np.take(img_data, idx, axis=axial_axis)
                lbl_slice = np.take(lbl_data, idx, axis=axial_axis)
            overlay_images_list.append(self._create_multi_class_overlay(
                img_slice, lbl_slice, opacity=0.2, boundary_thickness=1))

        grid_image = self._create_grid_image(overlay_images_list)
        saved_image_path = self._save_check_image(
            grid_image, case_name, iteration=iteration
        )
        grid_base64 = self._process_image_to_base64(grid_image)

        vlm_start_time = datetime.now()
        vlm_result = self._call_vlm_critic(
            target_organs, organ_label_mapping, slice_indices, grid_base64, case_name,
            retries=getattr(self, "_vlm_retries", 2),
            retry_backoff_sec=getattr(self, "_vlm_retry_backoff", 2.0)
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
    def _evaluate_cases(self, dataset_path, case_names, target_organs, organ_label_mapping,
                        num_slices=4, max_workers=8, iteration=0):
        dataset = Path(dataset_path)
        with (dataset / "dataset.json").open() as stream:
            ending = json.load(stream)["file_ending"]
        check_dir = Path(self.reflection_images_dir) / dataset.name / ("initial" if iteration == 0 else f"iteration_{iteration}")
        check_dir.mkdir(parents=True, exist_ok=True)
        self._current_check_dir = str(check_dir)

        def evaluate(case_name):
            try:
                return self.evaluate_case(str(dataset / "imagesTr" / f"{case_name}_0000{ending}"),
                    str(dataset / "labelsTr" / f"{case_name}{ending}"), target_organs,
                    organ_label_mapping, num_slices, iteration)
            except Exception as exc:
                self.logger.exception(f"Evaluation failed for {case_name}")
                return {"case_name": case_name, "error": str(exc), "iteration": iteration,
                        "vlm_evaluation": {"final_decision": "REJECT", "error": str(exc),
                                           "suggested_remedy": {"tool": "None"}}}

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                evaluations = list(pool.map(evaluate, case_names))
        finally:
            del self._current_check_dir
        for evaluation in evaluations:
            critic = evaluation.setdefault("vlm_evaluation", {})
            critic["final_decision"] = self.normalize_decision(critic.get("final_decision"))
        return evaluations

    def _summarize_evaluations(self, evaluations):
        accepted, rejected = [], []
        for ev in evaluations:
            critic = ev.setdefault("vlm_evaluation", {})
            critic["final_decision"] = self.normalize_decision(critic.get("final_decision"))
            (accepted if critic["final_decision"] == "ACCEPT" else rejected).append(ev["case_name"])
        total = len(evaluations)
        return {"total_evaluated": total, "accepted": len(accepted), "rejected": len(rejected),
                "accepted_cases": sorted(accepted), "rejected_cases": sorted(rejected),
                "acceptance_rate": len(accepted) / total if total else 0,
                "rejection_rate": len(rejected) / total if total else 0,
                "detailed_evaluations": evaluations}

    def reflect_on_dataset(self, constructed_dataset_path, adaptation_plan_path, num_slices_per_case=4,
                           max_workers=8, retries=2, retry_backoff=2.0, iteration=0):
        dataset = Path(constructed_dataset_path)
        with open(adaptation_plan_path) as stream:
            plan = json.load(stream)
        with (dataset / "dataset.json").open() as stream:
            info = json.load(stream)
        ending = info["file_ending"]
        cases = sorted(p.name[:-len("_0000" + ending)] for p in (dataset / "imagesTr").glob("*_0000" + ending))
        if not cases:
            raise ValueError("No target images to evaluate")
        self._vlm_retries, self._vlm_retry_backoff = retries, retry_backoff
        started = datetime.now()
        evaluations = self._evaluate_cases(constructed_dataset_path, cases, plan["target_organs"],
            {k: v for k, v in info["labels"].items() if k != "background"},
            num_slices_per_case, max_workers, iteration)
        summary = self._summarize_evaluations(evaluations)
        summary.update(dataset_name=dataset.name, iteration=iteration,
                       reflection_start_time=started.isoformat(), reflection_end_time=datetime.now().isoformat())
        plan["reflection"] = summary
        plan["current_refinement_plan"] = self.generate_refinement_plan(summary)
        plan.setdefault("reflection_history", []).append({"iteration": iteration, "evaluations": evaluations})
        with open(adaptation_plan_path, "w") as stream:
            json.dump(plan, stream, indent=2)
        return summary

    def _update_dataset_json(self, dataset_path, new_num_training):
        path = Path(dataset_path) / "dataset.json"
        info = json.loads(path.read_text())
        info["numTraining"] = new_num_training
        path.write_text(json.dumps(info, indent=2))

    def generate_refinement_plan(self, reflection_summary):
        actions = {}
        summary = self._summarize_evaluations(reflection_summary.get("detailed_evaluations", []))
        known = {ev["case_name"] for ev in summary["detailed_evaluations"]}
        unassessed = set(reflection_summary.get("rejected_cases", []) + reflection_summary.get("uncertain_cases", [])) - known
        rejected = set(summary["rejected_cases"]) | unassessed
        failures = sorted(unassessed)
        defaults = {"CLAHE": {"clip_limit": 3.0, "tile_grid_size": [8, 8]},
                    "ZoomSAM3": {"margin": 10},
                    "TopologyCleaner": {"min_size": 0, "keep_top_k": 1, "mode": "3d"},
                    "SymmetryChecker": {"margin": 10}}
        for ev in summary["detailed_evaluations"]:
            critic = ev.get("vlm_evaluation", {})
            if self.normalize_decision(critic.get("final_decision")) == "ACCEPT":
                continue
            tool = critic.get("suggested_remedy", {}).get("tool")
            tool = {"Zoom": "ZoomSAM3", "Topology": "TopologyCleaner", "Symmetry": "SymmetryChecker"}.get(tool, tool)
            if tool not in defaults:
                failures.append(ev["case_name"])
                continue
            actions.setdefault(tool, {"cases": [], "params": defaults[tool]})["cases"].append(ev["case_name"])
        return {"generated_at": datetime.now().isoformat(), "actions": actions,
                "failed_cases": failures, "stop_refinement": not rejected,
                "reason": "All samples accepted" if not rejected else "Rejected samples require reassessment"}

    def reflect_specific_cases(self, constructed_dataset_path, case_names, target_organs,
                               organ_label_mapping, num_slices=4, max_workers=8, iteration=1):
        evaluations = self._evaluate_cases(constructed_dataset_path, case_names, target_organs,
                                         organ_label_mapping, num_slices, max_workers, iteration)
        return {"evaluations": evaluations, "cases_evaluated": len(evaluations), "iteration": iteration}

    def merge_reflection_results(self, adaptation_plan_path, new_results):
        with open(adaptation_plan_path) as stream:
            plan = json.load(stream)
        old = plan.get("reflection", {})
        evaluations = {ev["case_name"]: ev for ev in old.get("detailed_evaluations", [])}
        for ev in new_results["evaluations"]:
            evaluations[ev["case_name"]] = ev
        summary = self._summarize_evaluations(list(evaluations.values()))
        summary.update(dataset_name=old.get("dataset_name"), iteration=new_results["iteration"])
        plan["reflection"] = summary
        plan["current_refinement_plan"] = self.generate_refinement_plan(summary)
        plan.setdefault("reflection_history", []).append(new_results)
        with open(adaptation_plan_path, "w") as stream:
            json.dump(plan, stream, indent=2)

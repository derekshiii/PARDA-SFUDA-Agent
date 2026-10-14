"""Frozen native RGB BiomedParse inference in its own environment; no target GT access."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    backend = Path(os.environ.get("BIOMEDPARSE2D_ROOT", Path(__file__).resolve().parent / "BiomedParse")).resolve()
    sys.path.insert(0, str(backend))
    import numpy as np
    import nibabel as nib
    from PIL import Image
    import torch
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utilities.arguments import load_opt_from_config_files
    from utilities.distributed import init_distributed
    from utilities.constants import BIOMED_CLASSES
    from inference_utils.inference import interactive_infer_image

    prompts = list(json.loads(args.prompts).values())
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Non-empty text prompts are required")
    images = sorted((Path(args.dataset) / "imagesTr").glob("*_0000.png"))
    if not images:
        raise ValueError("No PNG training images found")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    options = init_distributed(load_opt_from_config_files([str(backend / "configs/biomedparse_inference.yaml")]))
    model = BaseModel(options, build_model(options)).from_pretrained(args.checkpoint).eval().cuda()
    with torch.inference_mode():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)
        for path in images:
            with Image.open(path) as source:
                image = source.convert("RGB")
            probabilities = np.asarray(interactive_infer_image(model, image, prompts))
            if probabilities.shape != (len(prompts), image.height, image.width) or not np.isfinite(probabilities).all():
                raise ValueError(f"Invalid BiomedParse prediction for {path.name}")
            scores = np.concatenate((np.full((1, image.height, image.width), 0.5), probabilities), axis=0)
            mask = scores.argmax(axis=0).astype(np.uint16)[..., None]
            case = path.name.removesuffix("_0000.png")
            nib.save(nib.Nifti1Image(mask, np.eye(4)), output / f"{case}.nii.gz")
            print(f"RGB_BIOMEDPARSE_OK {case} {mask.shape}", flush=True)


if __name__ == "__main__":
    main()

# PARDA: Perceive–Act–Reflect–Distill

PARDA is an agent framework for **source-free unsupervised domain adaptation (SFUDA)** in medical image segmentation. It uses unlabeled target images, organ names, frozen segmentation foundation models, a vision-language model (VLM), and a pretrained source checkpoint to build a curated target dataset and train an adapted nnUNet student.

Target ground-truth masks are not required or read during adaptation. The implementation supports 3D NIfTI volumes and 2D PNG images, including native RGB inputs.

## Method

1. **Perceive:** inspect target images, identify domain characteristics, construct organ prompts, and select a segmentation strategy and preprocessing settings.
2. **Act:** generate pseudo-labels with BiomedParse, SAM3, or the dual-expert strategy selected by Perception.
3. **Reflect:** assess anatomical plausibility, boundary precision, morphological consistency, and false positives. Each criterion receives an explanation and a score from 1 to 10. Rejected samples undergo tool-based refinement and reassessment, with at most five attempts per sample.
4. **Distill:** retain accepted image–label pairs and fine-tune an nnUNet student initialized from the complete source network, including its segmentation heads.

Reflection uses two decisions: `ACCEPT` and `REJECT`. Acceptance follows the VLM rubric rather than a weighted score cutoff. Each rejected sample continues until acceptance or exhaustion of its refinement budget, independently of the dataset's overall rejection rate.

| Refinement tool | Operation |
|---|---|
| `ZoomSAM3` | Crop label regions with a 10-pixel margin, re-segment with SAM3, and restore the masks to their original coordinates. |
| `CLAHE` | Enhance local contrast and re-segment with SAM3 using the current label's boxes and organ prompts. |
| `TopologyCleaner` | Remove unwanted connected components according to the expected component counts. |
| `SymmetryChecker` | Use paired organs to construct a contralateral region prior and re-segment the incomplete side with SAM3. |

Refinement preserves unprocessed classes and keeps the input label if an operation fails. Reflection reassesses the result after each attempt. Symmetry refinement uses an estimated body midline and mirrored regions; oblique or strongly asymmetric anatomy may require more accurate registration.

## Repository layout

| File | Purpose |
|---|---|
| [Agent_Base.py](Agent_Base.py) | CLI entry point and orchestration of the four stages |
| [Perception.py](Perception.py) | Domain analysis, organ prompts, and adaptation plans |
| [Action.py](Action.py) | Segmentation strategy execution, dataset construction, and refinement |
| [Reflection.py](Reflection.py) | Pseudo-label assessment and refinement recommendations |
| [Prompts.py](Prompts.py) | Perception and Reflection prompt templates |
| [Toolbox.py](Toolbox.py) | Segmentation wrappers and refinement tools |
| [nnUNet2BiomedParse.py](nnUNet2BiomedParse.py) | Image conversion for BiomedParse without target labels |
| [Knowledge_Distillation.py](Knowledge_Distillation.py) | nnUNet preprocessing and training orchestration |
| [train_student.py](train_student.py) | Source initialization and the 300-epoch student training budget |
| [tests/](tests/) | Automated pipeline and regression tests |

## Setup

Run the agent in a Python environment compatible with the BiomedParse 3D backend and CUDA-enabled PyTorch. Install the agent's supporting packages in that environment:

```bash
python -m pip install openai numpy nibabel hydra-core scipy pillow opencv-python tqdm
```

Install BiomedParse 3D, native RGB BiomedParse for PNG inputs, SAM3, and nnUNet v2 with their respective dependencies and model checkpoints. SAM3 and RGB BiomedParse run in separate Python environments; nnUNet can also use a separate environment through `--nnunet-env`. If your endpoint connection uses a SOCKS proxy, install `httpx[socks]` in the agent environment.

**Configure backend paths through environment variables before running.** Set them in the shell that launches the agent; subprocesses inherit these settings:

| Variable | Purpose | Default |
|---|---|---|
| `BIOMEDPARSE3D_ROOT` | BiomedParse 3D repository | `BiomedParsev2/` under this repository |
| `BIOMEDPARSE2D_ROOT` | Native RGB BiomedParse repository | `BiomedParse/` under this repository |
| `BIOMEDPARSE2D_PYTHON` | RGB BiomedParse Python executable | Current Python executable |
| `BIOMEDPARSE2D_CHECKPOINT` | Native RGB 2D checkpoint | `biomedparse_v1.pt` under `BIOMEDPARSE2D_ROOT` |
| `SAM3_ROOT` | SAM3 repository | `sam3/` under this repository |
| `SAM3_PYTHON` | SAM3 Python executable | Current Python executable |

For example, `export SAM3_PYTHON="/path/to/sam3/environment/bin/python"` selects the SAM3 environment. Local exports can be saved in `.env.local` and loaded with `source .env.local`; this file is ignored by Git. Backend repositories, weights, local credentials, generated datasets, and default run outputs are also excluded from version control.

Run commands from the repository root. Configure your VLM endpoint and nnUNet storage locations in the same shell:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="https://<your-vlm-endpoint>/v1"
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

The endpoint must support image inputs and the OpenAI-compatible Chat Completions API. `nnUNet_raw` must be writable because the agent creates a separate adaptation dataset there. Weights, datasets, and backend environments must be supplied separately.

## Target data

Provide the target **training split** in nnUNet format:

```text
Dataset125_Target/
├── dataset.json
└── imagesTr/
    ├── case001_0000.nii.gz
    └── case002_0000.nii.gz
```

Example `dataset.json` for single-modality CT with three target organs:

```json
{
  "channel_names": {"0": "CT"},
  "labels": {"background": 0, "Liver": 1, "Left Kidney": 2, "Right Kidney": 3},
  "numTraining": 2,
  "file_ending": ".nii.gz"
}
```

`labels` defines class names and IDs; a `labelsTr` directory is unnecessary. For NIfTI inputs, supply the selected modality as channel `_0000`. For PNG inputs, use `case001_0000.png`, set `file_ending` to `.png`, and describe the image channels correctly in `channel_names` (for RGB, `{"0": "R", "1": "G", "2": "B"}`). PNG images retain their color through inference, refinement, and student dataset construction.

For distillation, source and target input channel counts, class IDs, and class meanings must match. List `--targets` in the source label order. The caller supplies the paper's 8:2 train/test split; the pipeline adapts only `imagesTr` and does not create or access a held-out test split.

## Run

```bash
python Agent_Base.py \
  --target-data /path/to/Dataset125_Target \
  --targets 'Liver,Left Kidney,Right Kidney' \
  --source-checkpoint /path/to/source/checkpoint_final.pth \
  --biomedparse-checkpoint /path/to/biomedparse_3D_AllData_MultiView_edge.ckpt \
  --sam3-checkpoint /path/to/sam3.pt \
  --nnunet-env /path/to/nnunet/environment \
  --result-dir ./Agent-Result \
  --gpus 0
```

- `--model` selects the VLM model name or endpoint alias; the default is `Qwen3-VL-32B-Instruct`.
- `--base-url` overrides `OPENAI_BASE_URL`.
- `--skip-distillation` runs pseudo-label generation and the complete reflection/refinement loop without training a student; `--source-checkpoint` is then optional.
- `--biomedparse-checkpoint` specifies the 3D checkpoint. Set `BIOMEDPARSE2D_CHECKPOINT` for the native RGB 2D checkpoint.
- `--nnunet-env` selects the nnUNet environment directory; if omitted, training uses executables from the current environment.

## Settings and reproducibility

| Setting | Value |
|---|---|
| Default VLM | Qwen3-VL-32B-Instruct |
| VLM temperature / maximum output tokens | 0.1 / 4096 |
| Refinement budget | Five attempts per rejected sample after the initial assessment |
| Student distillation | 300 epochs, Dice + Cross-Entropy loss |
| Default GPU | 0 |
| Initial low-contrast enhancement | CLAHE when Perception's contrast rating is ≤3; clip limit 2.0 |
| CLAHE refinement default | Clip limit 3.0, grid 8×8 |

Preprocessing uses the source checkpoint's network architecture, patch size, and batch size while retaining target normalization and spacing. Incompatible source weights fail explicitly. The source checkpoint determines the student architecture and parameter count; a checkpoint path alone cannot establish its training provenance.

## Outputs

The agent returns the path of a generated `DatasetXXX_*_PARDA` under `nnUNet_raw`. It contains original target images and the accepted pseudo-labels, with `dataset.json` updated to the retained sample count. Distillation fails explicitly if no accepted samples remain.

The result directory contains timestamped adaptation plans, logs, pseudo-label working files, visual assessment images under `reflection_check_images/`, and rejected or unassessed samples under `excluded_cases/`. Plans record criterion scores, decisions, refinement operations, attempts, retained/excluded cases, and the stopping reason. Original input images are preserved. Student preprocessing and training outputs use `nnUNet_preprocessed` and `nnUNet_results`.

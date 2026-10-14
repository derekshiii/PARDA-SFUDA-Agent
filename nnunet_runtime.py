"""Apply experiment paths before nnUNet imports cache them, then run native preprocessing."""
import os


def configure_paths():
    from nnunetv2 import paths
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        if os.environ.get(name):
            setattr(paths, name, os.environ[name])


if __name__ == "__main__":
    configure_paths()
    from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry
    plan_and_preprocess_entry()

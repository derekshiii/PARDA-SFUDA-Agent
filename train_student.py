"""Run the installed nnUNet CLI with the paper's 300-epoch training budget."""
from nnunet_runtime import configure_paths

configure_paths()
from nnunetv2.run import run_training as training
import torch
from torch.nn.parallel import DistributedDataParallel


_get_trainer = training.get_trainer_from_args


def get_paper_trainer(*args, **kwargs):
    trainer = _get_trainer(*args, **kwargs)
    # Set before initialize() creates the learning-rate schedule.
    trainer.num_epochs = 300
    return trainer


def load_source_weights(network, fname, verbose=False):
    # nnUNet's transfer-learning loader skips segmentation heads; SFUDA initializes the entire source model.
    module = network.module if isinstance(network, DistributedDataParallel) else network
    module = getattr(module, "_orig_mod", module)
    checkpoint = torch.load(fname, map_location="cpu", weights_only=False)
    module.load_state_dict(checkpoint["network_weights"], strict=True)
    print(f"Loaded complete source network, including segmentation heads: {fname}")


# Module-level assignment also applies in nnUNet's spawned GPU workers.
training.get_trainer_from_args = get_paper_trainer
training.load_pretrained_weights = load_source_weights


if __name__ == "__main__":
    training.run_training_entry()

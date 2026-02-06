import os
import numpy as np
import torch
import torch.nn.functional as F
import hydra
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
import gc
import json
from tqdm import tqdm
import nibabel as nib
import multiprocessing as mp
from multiprocessing import Process, Queue
import time

# ==================== Configuration Options ====================
# GPU Configuration
CUDA_DEVICES = ['0', '1']  # List of GPU device IDs to use
NUM_WORKERS = len(CUDA_DEVICES)  # Number of worker processes, typically equals number of GPUs

# Data Path Configuration
INPUT_DIR = "/8TB_HDD_2/Datasets/AMOS_CT-Dual-enhanced"
OUTPUT_DIR = "/8TB_HDD_2/BiomedParsev2/Inference_results/AMOS_CT-Dual-enhanced"
CHECKPOINT_PATH = "/8TB_HDD_2/BiomedParsev2/biomedparse_3D_AllData_MultiView_edge.ckpt"

# Save Options
SAVE_NII = False           # Set to True to save predictions in NIfTI format
SAVE_PROB = False          # Set to True to save probability distributions
NII_OUTPUT_DIR = None      # NIfTI output directory, None uses default subdirectory "nii_predictions"
PROB_OUTPUT_DIR = None     # Probability maps output directory, None uses default subdirectory "probability_maps"

# Model Inference Configuration
SLICE_BATCH_SIZE = 4       # Slice batch size
INTERPOLATE_SIZE = 512     # Interpolation size
NMS_IOU_THRESHOLD = 0.5    # IOU threshold for NMS
SCORE_THRESHOLD = 0.5      # Score threshold
# ==============================================================


def load_case(file_path):
    """Load data from NPZ file.
    
    Args:
        file_path: Path to NPZ file
        
    Returns:
        tuple: (image, text_prompts, gt) where gt is None if not present
    """
    data = np.load(file_path, allow_pickle=True)
    image = data["imgs"]
    text_prompts = data["text_prompts"].item()
    gt = data["gts"] if "gts" in data else None
    return image, text_prompts, gt


def merge_multiclass_masks(masks, ids):
    """Merge multiple binary masks into a single multi-class segmentation mask.
    
    Args:
        masks: Tensor of shape (num_classes, H, W) containing binary masks
        ids: List of class IDs corresponding to each mask
        
    Returns:
        Tensor: Multi-class segmentation mask with shape (H, W)
    """
    # Add background mask with value 0.5
    bg_mask = 0.5 * torch.ones_like(masks[0:1])
    keep_masks = torch.cat([bg_mask, masks], dim=0)
    
    # Use argmax to determine class for each pixel
    class_mask = keep_masks.argmax(dim=0)

    # Remap class indices if they don't match sequential order
    id_map = {j + 1: int(ids[j]) for j in range(len(ids)) if j + 1 != int(ids[j])}
    if len(id_map) > 0:
        orig_mask = class_mask.clone()
        for j in id_map:
            class_mask[orig_mask == j] = id_map[j]

    return class_mask


def postprocess(model_outputs, object_existence, threshold=SCORE_THRESHOLD, do_nms=True):
    """Post-process model outputs to generate final segmentation masks.
    
    Args:
        model_outputs: Raw model output logits
        object_existence: Object existence scores
        threshold: Score threshold for filtering predictions
        do_nms: Whether to apply Non-Maximum Suppression
        
    Returns:
        Tensor: Processed segmentation masks
    """
    if do_nms and model_outputs.shape[0] > 1:
        from utils import slice_nms
        return slice_nms(model_outputs.sigmoid(), object_existence.sigmoid(), 
                        iou_threshold=NMS_IOU_THRESHOLD, score_threshold=threshold)
    
    # Apply sigmoid and threshold
    mask = (model_outputs.sigmoid()) * (
        object_existence.sigmoid() > threshold
    ).int().unsqueeze(-1).unsqueeze(-1)
    return mask


def save_as_nii(mask_array, output_path, affine=None):
    """Save mask array as NIfTI format.
    
    Args:
        mask_array: Numpy array or Tensor containing the segmentation mask
        output_path: Path to save the NIfTI file
        affine: Affine transformation matrix, uses identity matrix if None
    """
    if torch.is_tensor(mask_array):
        mask_array = mask_array.cpu().numpy()
    
    if affine is None:
        affine = np.eye(4)
    
    mask_array = mask_array.astype(np.int16)
    nii_img = nib.Nifti1Image(mask_array, affine)
    nib.save(nii_img, output_path)


def save_probability_maps(prob_maps, output_path, class_ids):
    """Save probability distribution maps.
    
    Args:
        prob_maps: Tensor or array of shape (num_classes, D, H, W)
        output_path: Path to save the probability maps
        class_ids: List of class IDs
    """
    if torch.is_tensor(prob_maps):
        prob_maps = prob_maps.cpu().numpy()
    
    # Convert to float16 to save space
    prob_maps = prob_maps.astype(np.float16)
    
    # Save as compressed NPZ file
    np.savez_compressed(
        output_path,
        probability_maps=prob_maps,
        class_ids=np.array(class_ids)
    )


def worker_process(gpu_id, file_queue, result_queue, progress_queue, input_dir, output_dir, 
                   checkpoint_path, nii_dir, prob_dir):
    """Worker process function that processes files on a specified GPU.
    
    Args:
        gpu_id: GPU device ID to use
        file_queue: Queue containing files to process
        result_queue: Queue for storing processing results
        progress_queue: Queue for progress updates
        input_dir: Input directory path
        output_dir: Output directory path
        checkpoint_path: Path to model checkpoint
        nii_dir: Directory for NIfTI outputs (can be None)
        prob_dir: Directory for probability maps (can be None)
    """
    # Set GPU for current process
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    device = torch.device("cuda:0")  # Use cuda:0 since CUDA_VISIBLE_DEVICES is set
    
    try:
        # Initialize model
        GlobalHydra.instance().clear()
        hydra.initialize(config_path="../BiomedParsev2/configs/model", job_name=f"prediction_gpu_{gpu_id}")
        cfg = compose(config_name="biomedparse_3D")
        model = hydra.utils.instantiate(cfg, _convert_="object")
        model.load_pretrained(checkpoint_path)

        model.to(device)
        model.eval()
        
        # Process files from queue
        processed_count = 0
        while True:
            try:
                file = file_queue.get(timeout=5)  # 5 second timeout
                if file is None:  # Termination signal
                    break
                    
                file_path = os.path.join(input_dir, file)
                
                # Process single file
                result = process_single_file(
                    file_path, file, output_dir, nii_dir, prob_dir,
                    model, device, gpu_id
                )
                
                # Put result in result queue
                if result:
                    result_queue.put(result)
                
                processed_count += 1
                
                # Update progress
                progress_queue.put({'gpu_id': gpu_id, 'file': file, 'status': 'completed'})
                
                # Memory cleanup
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                progress_queue.put({'gpu_id': gpu_id, 'file': file, 'status': 'error', 'error': str(e)})
                continue
    
    except Exception as e:
        progress_queue.put({'gpu_id': gpu_id, 'status': 'worker_error', 'error': str(e)})


def process_single_file(file_path, filename, output_dir, nii_output_dir, 
                       prob_output_dir, model, device, gpu_id):
    """Process a single file - inference only, no metrics computation.
    
    Args:
        file_path: Path to input NPZ file
        filename: Name of the file
        output_dir: Directory for NPZ predictions
        nii_output_dir: Directory for NIfTI outputs (can be None)
        prob_output_dir: Directory for probability maps (can be None)
        model: BiomedParse model instance
        device: Torch device
        gpu_id: GPU ID for logging
        
    Returns:
        dict: Processing result containing filename
    """
    try:
        # Load data
        npz_data = np.load(file_path, allow_pickle=True)
        imgs = npz_data["imgs"]
        text_prompts = npz_data["text_prompts"].item()
        
        # Extract class IDs and construct text prompt
        ids = [int(_) for _ in text_prompts.keys() if _ != "instance_label"]
        ids.sort()
        text = "[SEP]".join([text_prompts[str(i)] for i in ids])
        
        # Preprocess input
        from utils import process_input, process_output
        imgs, pad_width, padded_size, valid_axis = process_input(imgs, INTERPOLATE_SIZE)
        imgs = imgs.to(device).int()

        # Model inference
        input_tensor = {
            "image": imgs.unsqueeze(0),
            "text": [text],
        }

        with torch.no_grad():
            output = model(input_tensor, mode="eval", slice_batch_size=SLICE_BATCH_SIZE)

        # Get raw probability distributions
        mask_logits = output["predictions"]["pred_gmasks"]
        object_existence = output["predictions"]["object_existence"]
        
        # Interpolate to specified size
        mask_logits_resized = F.interpolate(
            mask_logits,
            size=(INTERPOLATE_SIZE, INTERPOLATE_SIZE),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        
        # Save probability maps if enabled
        if SAVE_PROB and prob_output_dir:
            try:
                prob_maps = mask_logits_resized.sigmoid() * object_existence.sigmoid().unsqueeze(-1).unsqueeze(-1)
                prob_maps_list = []
                for i in range(prob_maps.shape[0]):
                    prob_map_single = process_output(prob_maps[i], pad_width, padded_size, valid_axis)
                    prob_maps_list.append(prob_map_single)
                
                if torch.is_tensor(prob_maps_list[0]):
                    prob_maps_original = torch.stack(prob_maps_list, dim=0)
                else:
                    prob_maps_original = np.stack(prob_maps_list, axis=0)
                
                base_filename = os.path.splitext(filename)[0]
                prob_save_path = os.path.join(prob_output_dir, f"{base_filename}_prob.npz")
                save_probability_maps(prob_maps_original, prob_save_path, ids)
            except Exception as e:
                # Log error but continue processing
                print(f"[GPU {gpu_id}] Warning: Failed to save probability maps for {filename}: {e}")

        # Post-process to get final mask
        mask_preds = postprocess(mask_logits_resized, object_existence)
        mask_preds = merge_multiclass_masks(mask_preds, ids)
        mask_preds = process_output(mask_preds, pad_width, padded_size, valid_axis)

        # Save prediction as NPZ format
        save_path = os.path.join(output_dir, filename)
        np.savez_compressed(save_path, segs=mask_preds)

        # Save NIfTI if enabled
        if SAVE_NII and nii_output_dir:
            base_filename = os.path.splitext(filename)[0]
            nii_save_path = os.path.join(nii_output_dir, f"{base_filename}_pred.nii.gz")
            try:
                save_as_nii(mask_preds, nii_save_path)
            except Exception as e:
                print(f"[GPU {gpu_id}] Warning: Failed to save NIfTI for {filename}: {e}")

        # Return filename for progress tracking
        return {'filename': filename}
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error processing {filename}: {e}")
        return None


def main():
    """Main function for multi-GPU inference."""
    tqdm.write("=" * 60)
    tqdm.write("Starting Multi-GPU BiomedParse Inference")
    tqdm.write("=" * 60)
    tqdm.write(f"GPU Devices: {CUDA_DEVICES}")
    tqdm.write(f"Number of Workers: {NUM_WORKERS}")
    tqdm.write(f"Input Directory: {INPUT_DIR}")
    tqdm.write(f"Output Directory: {OUTPUT_DIR}")
    tqdm.write(f"Checkpoint: {CHECKPOINT_PATH}")
    tqdm.write(f"SAVE_NII: {SAVE_NII}")
    tqdm.write(f"SAVE_PROB: {SAVE_PROB}")
    tqdm.write("-" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Setup NIfTI output directory
    nii_dir = None
    if SAVE_NII:
        if NII_OUTPUT_DIR is None:
            nii_dir = os.path.join(OUTPUT_DIR, "nii_predictions")
        else:
            nii_dir = NII_OUTPUT_DIR
        os.makedirs(nii_dir, exist_ok=True)
        tqdm.write(f"NIfTI Output Directory: {nii_dir}")
    
    # Setup probability maps output directory
    prob_dir = None
    if SAVE_PROB:
        if PROB_OUTPUT_DIR is None:
            prob_dir = os.path.join(OUTPUT_DIR, "probability_maps")
        else:
            prob_dir = PROB_OUTPUT_DIR
        os.makedirs(prob_dir, exist_ok=True)
        tqdm.write(f"Probability Maps Output Directory: {prob_dir}")
    
    # Get all files to process
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")]
    tqdm.write(f"\nFound {len(all_files)} files to process")
    
    if not all_files:
        tqdm.write("No .npz files found in input directory")
        return
    
    # Create inter-process communication queues
    file_queue = Queue()
    result_queue = Queue()
    progress_queue = Queue()
    
    # Put files into queue
    for file in all_files:
        file_queue.put(file)
    
    # Add termination signals (one None per worker)
    for _ in range(NUM_WORKERS):
        file_queue.put(None)
    
    # Start worker processes
    processes = []
    for i, gpu_id in enumerate(CUDA_DEVICES):
        p = Process(
            target=worker_process,
            args=(gpu_id, file_queue, result_queue, progress_queue, INPUT_DIR, OUTPUT_DIR, 
                  CHECKPOINT_PATH, nii_dir, prob_dir),
            name=f"GPU-{gpu_id}-Worker"
        )
        p.start()
        processes.append(p)
    
    tqdm.write("\nStarting inference...\n")
    
    # Track progress
    completed_files = []
    
    # Use tqdm to display progress
    with tqdm(total=len(all_files), desc="Processing files", unit="file") as pbar:
        completed = 0
        while completed < len(all_files):
            try:
                # Check for progress updates
                if not progress_queue.empty():
                    progress_info = progress_queue.get(timeout=1)
                    if progress_info.get('status') == 'completed':
                        pbar.update(1)
                        completed += 1
                        file = progress_info.get('file', 'Unknown')
                        gpu = progress_info.get('gpu_id', 'Unknown')
                        tqdm.write(f"[GPU {gpu}] Completed: {file}")
                    elif progress_info.get('status') == 'error':
                        pbar.update(1)
                        completed += 1
                        file = progress_info.get('file', 'Unknown')
                        gpu = progress_info.get('gpu_id', 'Unknown')
                        error = progress_info.get('error', 'Unknown error')
                        tqdm.write(f"[GPU {gpu}] Error processing {file}: {error}")
                
                # Collect results (consume queue to prevent blocking)
                while not result_queue.empty():
                    result = result_queue.get_nowait()
                    if result:
                        completed_files.append(result)
                
                time.sleep(0.1)
                
            except Exception as e:
                continue
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect remaining results
    while not result_queue.empty():
        result = result_queue.get_nowait()
        if result and result not in completed_files:
            completed_files.append(result)
    
    tqdm.write("\n" + "=" * 60)
    tqdm.write("Inference Complete")
    tqdm.write("=" * 60)
    
    # Save processing summary
    summary = {
        'total_files': len(all_files),
        'processed_files': len(completed_files),
        'config': {
            'cuda_devices': CUDA_DEVICES,
            'checkpoint': CHECKPOINT_PATH,
            'slice_batch_size': SLICE_BATCH_SIZE,
            'interpolate_size': INTERPOLATE_SIZE,
            'save_nii': SAVE_NII,
            'save_prob': SAVE_PROB
        },
        'output_directories': {
            'predictions': OUTPUT_DIR,
            'nii_predictions': nii_dir,
            'probability_maps': prob_dir
        },
        'processed_file_list': [f['filename'] for f in completed_files]
    }
    
    # Save summary as JSON
    summary_path = os.path.join(OUTPUT_DIR, 'inference_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    tqdm.write(f"\nProcessed {len(completed_files)}/{len(all_files)} files with {NUM_WORKERS} GPUs")
    tqdm.write(f"Summary saved to: {summary_path}")
    
    if SAVE_NII:
        tqdm.write(f"NIfTI predictions saved to: {nii_dir}")
    
    if SAVE_PROB:
        tqdm.write(f"Probability maps saved to: {prob_dir}")
    
    tqdm.write("\n" + "=" * 60)
    tqdm.write("All Done!")
    tqdm.write("=" * 60)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
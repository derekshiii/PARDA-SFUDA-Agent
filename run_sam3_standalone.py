import os
import sys
import json
import cv2
import torch
import numpy as np
import shutil
import nibabel as nib
import argparse
from tqdm import tqdm
from typing import List, Dict, Any

# 添加SAM3路径
SAM3_DIR = os.path.abspath(os.environ.get("SAM3_ROOT", os.path.join(os.path.dirname(__file__), "sam3")))
if SAM3_DIR not in sys.path:
    sys.path.append(SAM3_DIR)

from nnUNet2BiomedParse import load_volume

class StandaloneSAM3Predictor:
    def __init__(self, checkpoint_path: str, gpus_to_use: List[int]):
        from sam3.model_builder import build_sam3_video_predictor
        print(f"Loading SAM3 model from {checkpoint_path} on GPUs {gpus_to_use}...")
        self.predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path=checkpoint_path
        )
        print("Model loaded.")

    def _load_and_transpose(self, nii_path: str, slice_axis=None):
        """
        加载 NIfTI 并转置。
        关键：必须与 Toolbox.py 中的 _standardize_orientation 逻辑保持一致！
        """
        img_obj = load_volume(nii_path)
        data = img_obj.get_fdata()
        affine = img_obj.affine
        header = img_obj.header
        
        shape = data.shape
        # 寻找最小维度作为 Z (Slice/Depth)
        min_dim_idx = int(np.argmin(shape)) if slice_axis is None else int(slice_axis)
        if min_dim_idx not in (0, 1, 2):
            raise ValueError("Invalid slice axis")
        
        # 目标: (D, H, W)
        permute_order = [0, 1, 2]
        permute_order.pop(min_dim_idx)
        permute_order.insert(0, min_dim_idx)
        
        data_transposed = data.transpose(tuple(permute_order))
        return data_transposed, affine, header, permute_order

    def _nii_to_frames(self, data_transposed, temp_base_dir, case_id, source_image=None):
        save_dir = os.path.join(temp_base_dir, case_id)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        if source_image and str(source_image).endswith(".png"):
            pixels = cv2.imread(str(source_image))
            if pixels is None:
                raise ValueError("Could not load PNG frame")
            h, w = pixels.shape[:2]
            scale = max(1.0, 64 / min(h, w))
            pixels = cv2.resize(pixels, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
            if not cv2.imwrite(os.path.join(save_dir, "00000.jpg"), pixels):
                raise IOError("Could not write SAM3 frame")
            return save_dir, pixels.shape[1], pixels.shape[0]

        # 归一化 (0-255)
        lower = np.percentile(data_transposed, 0.5)
        upper = np.percentile(data_transposed, 99.5)
        
        if (upper - lower) > 1e-5:
            data_norm = (data_transposed - lower) / (upper - lower) * 255.0
        else:
            data_norm = np.zeros_like(data_transposed)
            
        data_norm = np.clip(data_norm, 0, 255).astype(np.uint8)
        
        # 保存每一帧
        # data_transposed 是 (D, H, W)
        D, H, W = data_norm.shape
        # SAM3's CUDA antialias kernel cannot downsample to extremely small ROI frames.
        scale = max(1.0, 64 / min(H, W))
        frame_w, frame_h = int(round(W * scale)), int(round(H * scale))
        img_paths = []
        
        for i in range(D):
            slice_img = data_norm[i, :, :]
            # SAM3 需要 RGB 图像
            slice_rgb = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)
            if scale > 1:
                slice_rgb = cv2.resize(slice_rgb, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            p = os.path.join(save_dir, f"{i:05d}.jpg")
            cv2.imwrite(p, slice_rgb)
            img_paths.append(p)
            
        return save_dir, frame_w, frame_h

    def _normalize_box(self, box, w, h):
        """
        归一化 Box 坐标 [x, y, bw, bh] -> [0-1]
        """
        x, y, bw, bh = box
        
        # Clip
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        
        if bw <= 0 or bh <= 0: return None
        return [x/w, y/h, bw/w, bh/h]

    def process_case(self, case_id, image_path, prompts, output_dir, temp_dir, slice_axis=None):
        session_id = None
        try:
            # 1. 准备数据
            data_T, affine, header, permute_order = self._load_and_transpose(image_path, slice_axis)
            # vid_w, vid_h 是图像的物理宽高
            vid_dir, vid_w, vid_h = self._nii_to_frames(data_T, temp_dir, case_id, source_image=image_path)
            
            num_frames = data_T.shape[0]
            original_h, original_w = data_T.shape[1:]
            
            final_volume = np.zeros(data_T.shape, dtype=np.uint8)
            
            
            # 3. 处理每个 Prompt
            for p in prompts:
                # Each organ has an independent session so previous masks cannot contaminate it.
                resp = self.predictor.handle_request({"type": "start_session", "resource_path": vid_dir})
                session_id = resp["session_id"]
                obj_id = p['obj_id']
                key_slice = p.get('key_slice', 0)
                
                # 检查 slice 是否越界
                if key_slice < 0 or key_slice >= num_frames:
                    raise ValueError(f"Invalid key slice {key_slice} for {num_frames} frames")

                req = {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "obj_id": 1, 
                    "frame_index": key_slice
                }
                
                if 'text' in p:
                    req['text'] = p['text']
                
                if 'box' in p:
                    # p['box'] 是 [x, y, w, h] (基于 H, W 尺寸)
                    norm_box = self._normalize_box(p['box'], original_w, original_h)
                    if norm_box:
                        req['bounding_boxes'] = [norm_box]
                        req['bounding_box_labels'] = [1]
                
                self.predictor.handle_request(req)
                
                # 双向传播
                stream = self.predictor.handle_stream_request({
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "both"
                })
                
                # 收集 Mask
                for r in stream:
                    idx = r['frame_index']
                    if not 0 <= idx < num_frames:
                        raise ValueError("SAM3 returned an invalid frame index")
                    masks = r['outputs']['out_binary_masks']
                    if len(masks) > 0:
                        if isinstance(masks, torch.Tensor): masks = masks.cpu().numpy()
                        m = np.any(np.asarray(masks), axis=0)
                        
                        # 确保 Mask 尺寸匹配
                        if m.shape != (original_h, original_w):
                             m = cv2.resize(m.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                        
                        # 填充到结果 Volume
                        final_volume[idx, final_volume[idx] == obj_id] = 0
                        final_volume[idx, (m > 0) & (final_volume[idx] == 0)] = obj_id

                self.predictor.handle_request({"type": "close_session", "session_id": session_id})
                session_id = None
            
            # 4. 保存结果 (还原维度)
            inv_order = np.argsort(permute_order)
            final_orig = final_volume.transpose(tuple(inv_order))
            
            out_name = f"{case_id}.nii.gz"
            nib.save(nib.Nifti1Image(final_orig, affine, header), os.path.join(output_dir, out_name))
            print(f"Saved: {out_name}")
            
            # 清理临时图片
            shutil.rmtree(vid_dir)
            
        finally:
            if session_id:
                self.predictor.handle_request({"type": "close_session", "session_id": session_id})
            if 'vid_dir' in locals() and os.path.isdir(vid_dir):
                shutil.rmtree(vid_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_config", type=str, required=True, help="Path to JSON config with prompts")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=os.path.join(SAM3_DIR, "sam3", "sam3.pt"))
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()
    
    gpus = [int(x) for x in args.gpus.split(',')]
    
    # Init Predictor
    predictor = StandaloneSAM3Predictor(args.checkpoint, gpus)
    
    # Load Config
    with open(args.json_config, 'r') as f:
        config = json.load(f)
        
    temp_dir = os.path.join(args.output_dir, "temp_frames")
    
    try:
        for case_name, info in tqdm(config.items(), desc="Running SAM3"):
            predictor.process_case(case_name, info['image'], info['prompts'], args.output_dir,
                                   temp_dir, slice_axis=info.get("slice_axis"))
    finally:
        predictor.predictor.shutdown()

if __name__ == "__main__":
    main()

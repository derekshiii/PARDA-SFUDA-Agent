# 创建测试脚本 test_zoom_sam3.py
from Toolbox import ZoomSAM3Refiner
import logging

# 设置日志
logger = logging.getLogger("test")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(handler)

# 初始化 Refiner
refiner = ZoomSAM3Refiner(
    sam3_python_path="/home/syl/miniconda3/envs/sam3/bin/python",
    sam3_script_path="run_sam3_standalone.py",
    sam3_checkpoint="/8TB_HDD_2/sam3/sam3/sam3.pt",
    temp_root="./test_zoom_sam3_temp",
    logger=logger
)

# 测试数据
test_image = "/8TB_HDD_2/nnUNetFrame/nnUNet_raw/Dataset015_MR_CHAOS-T2-test/imagesTr/MR_CHAOS-T2_5_T2SPIR_0000.nii.gz"
test_label = "/8TB_HDD_2/nnUNetFrame/nnUNet_raw/Dataset015_MR_CHAOS-T2-test/labelsTr/MR_CHAOS-T2_5_T2SPIR.nii.gz"  # 粗糙的初始标签
output_refined = "./MR_CHAOS-T2_5_T2SPIR.nii.gz"

target_map = {
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Liver"
}

# 运行
try:
    refiner.run(
        image_path=test_image,
        coarse_label_path=test_label,
        output_path=output_refined,
        target_map=target_map,
        margin=10,
        gpus=[0, 1],
        cleanup=False  # 保留临时文件以便检查
    )
    print(f"\n✓ Success! Refined label saved to: {output_refined}")
except Exception as e:
    print(f"\n✗ Failed: {e}")
    import traceback
    traceback.print_exc()
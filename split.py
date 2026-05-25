import json
import random
import os
import re
from copy import deepcopy


# =========================
# 1. 参数设置
# =========================

INPUT_JSON = r"/home/shuangy/DEIMv2/data/waterway/result.json"      # 原始 COCO json 文件
OUTPUT_DIR = r"/home/shuangy/DEIMv2/data/waterway"           # 输出文件夹

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

TRAIN_JSON_NAME = "train.json"
TEST_JSON_NAME = "test.json"


# =========================
# 2. file_name 路径清理函数
# =========================

def clean_file_name(file_name: str) -> str:
    """
    删除 COCO images 中 file_name 前缀里的 G:/jpg\\ 或类似路径。
    兼容以下情况：
    G:/jpg\\xxx.jpg
    G:\\jpg\\xxx.jpg
    G:/jpg/xxx.jpg
    G:\\/jpg\\xxx.jpg
    """

    if file_name is None:
        return file_name

    file_name = str(file_name)

    # 统一处理可能出现的转义形式
    patterns = [
        r"^G:/jpg\\",
        r"^G:/jpg/",
        r"^G:\\jpg\\",
        r"^G:\\jpg/",
        r"^G:\\/jpg\\",
        r"^G:\\/jpg/",
    ]

    for p in patterns:
        file_name = re.sub(p, "", file_name)

    # 如果还存在完整路径，也可只保留文件名
    # 如果你不希望这样，可注释掉下面这两行
    file_name = file_name.replace("\\", "/")
    if file_name.startswith("G:\/jpg/"):
        file_name = file_name.replace("G:\/jpg/", "", 1)

    return file_name


# =========================
# 3. COCO 数据划分函数
# =========================

def split_coco_json(input_json, output_dir, train_ratio=0.8, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    if "images" not in coco:
        raise KeyError("输入 JSON 中未找到 'images' 字段。")

    if "annotations" not in coco:
        raise KeyError("输入 JSON 中未找到 'annotations' 字段。")

    images = deepcopy(coco["images"])
    annotations = deepcopy(coco["annotations"])

    # 清理 images 中的 file_name
    for img in images:
        if "file_name" in img:
            img["file_name"] = clean_file_name(img["file_name"])

    # 固定随机种子，保证可复现
    random.seed(seed)
    random.shuffle(images)

    total_num = len(images)
    train_num = int(total_num * train_ratio)

    train_images = images[:train_num]
    test_images = images[train_num:]

    train_image_ids = set(img["id"] for img in train_images)
    test_image_ids = set(img["id"] for img in test_images)

    train_annotations = [
        ann for ann in annotations
        if ann.get("image_id") in train_image_ids
    ]

    test_annotations = [
        ann for ann in annotations
        if ann.get("image_id") in test_image_ids
    ]

    # 构建 train COCO json
    train_coco = deepcopy(coco)
    train_coco["images"] = train_images
    train_coco["annotations"] = train_annotations

    # 构建 test COCO json
    test_coco = deepcopy(coco)
    test_coco["images"] = test_images
    test_coco["annotations"] = test_annotations

    train_json_path = os.path.join(output_dir, TRAIN_JSON_NAME)
    test_json_path = os.path.join(output_dir, TEST_JSON_NAME)

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)

    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_coco, f, ensure_ascii=False, indent=2)

    print("COCO 数据划分完成。")
    print(f"原始图像数量: {total_num}")
    print(f"训练集图像数量: {len(train_images)}")
    print(f"测试集图像数量: {len(test_images)}")
    print(f"训练集标注数量: {len(train_annotations)}")
    print(f"测试集标注数量: {len(test_annotations)}")
    print(f"训练集 JSON 保存至: {train_json_path}")
    print(f"测试集 JSON 保存至: {test_json_path}")


# =========================
# 4. 主函数
# =========================

if __name__ == "__main__":
    split_coco_json(
        input_json=INPUT_JSON,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        seed=RANDOM_SEED
    )
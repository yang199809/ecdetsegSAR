import json
from pathlib import Path


def remap_coco_categories(input_json, output_json, mapping):
    input_json = Path(input_json)
    output_json = Path(output_json)

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 修改 annotations
    for ann in data["annotations"]:
        old_id = ann["category_id"]
        if old_id not in mapping:
            raise ValueError(f"Unknown category_id {old_id} in annotation id={ann.get('id')}")
        ann["category_id"] = mapping[old_id]

    # 修改 categories
    old_categories = data.get("categories", [])
    new_categories = []

    used_new_ids = set()
    for cat in old_categories:
        old_id = cat["id"]
        if old_id in mapping:
            new_cat = dict(cat)
            new_cat["id"] = mapping[old_id]
            if new_cat["id"] not in used_new_ids:
                new_categories.append(new_cat)
                used_new_ids.add(new_cat["id"])

    new_categories = sorted(new_categories, key=lambda x: x["id"])
    data["categories"] = new_categories

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved fixed COCO json to: {output_json}")
    print("New categories:")
    for cat in new_categories:
        print(cat)


if __name__ == "__main__":
    # 3类：1,2,3 -> 0,1,2
    mapping = {
        1: 0,
    }

    remap_coco_categories(
        input_json="/home/yangshuang/EdgeCrafter-main/ecdetseg/data/hrsid/annotations/train2017.json",
        output_json="/home/yangshuang/EdgeCrafter-main/ecdetseg/data/hrsid/annotations/train2017_fixed.json",
        mapping=mapping,
    )

    remap_coco_categories(
        input_json="/home/yangshuang/EdgeCrafter-main/ecdetseg/data/hrsid/annotations/test2017.json",
        output_json="/home/yangshuang/EdgeCrafter-main/ecdetseg/data/hrsid/annotations/test2017_fixed.json",
        mapping=mapping,
    )
"""
Extract instances of target classes from a COCO-format dataset into a
reusable "instance pool" for copy-paste augmentation.
 
Usage:
    python extract_instance_pool.py \
        --ann /path/to/annotations/instances_train.json \
        --img-dir /path/to/train/images \
        --classes transferpottor_glass transferpottor_plastic \
        --out-dir ./copypaste_pool \
        --min-area 400
 
Each extracted instance is saved as an RGBA png (alpha channel = the
instance mask) under <out-dir>/<class_name>/, plus an index.json that
the TargetedCopyPaste transform (copy_paste_transform.py) reads at
training time.
"""
import argparse
import json
import os
 
import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
 
 
def ann_to_binary_mask(ann, height, width):
    """Convert a COCO annotation's segmentation field to a binary mask (H, W)."""
    segm = ann["segmentation"]
    if isinstance(segm, list):
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm["counts"], list):
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        rle = segm
    return maskUtils.decode(rle).astype(np.uint8)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ann", default="./MMdetection/data/MedBin_Dataset/train/_annotations.coco.json",
        help="COCO-format annotation json")
    parser.add_argument(
        "--img-dir", default="./MMdetection/data/MedBin_Dataset/train",
        help="folder containing the images")
    parser.add_argument(
        "--classes", nargs="+",
        default=["transferpettor_glass", "transferpettor_plastic"],
        help="class names to extract, e.g. transferpettor_glass transferpettor_plastic")
    parser.add_argument(
        "--out-dir", default="./copypaste_pool")
    parser.add_argument("--min-area", type=int, default=200,
                         help="skip instances smaller than this many pixels — avoids near-empty crops")
    args = parser.parse_args()
 
    coco = COCO(args.ann)
    name_to_id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
 
    missing = [c for c in args.classes if c not in name_to_id]
    if missing:
        raise ValueError(f"Class names not found in annotations: {missing}. "
                          f"Available: {sorted(name_to_id.keys())}")
 
    os.makedirs(args.out_dir, exist_ok=True)
    index = {}
    total = 0
 
    for cls_name in args.classes:
        cat_id = name_to_id[cls_name]
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        cls_dir = os.path.join(args.out_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        entries = []
 
        for ann in coco.loadAnns(ann_ids):
            if ann.get("area", 0) < args.min_area:
                continue
 
            img_info = coco.loadImgs(ann["image_id"])[0]
            img_path = os.path.join(args.img_dir, img_info["file_name"])
            if not os.path.exists(img_path):
                continue
 
            image = np.array(Image.open(img_path).convert("RGB"))
            h, w = img_info["height"], img_info["width"]
            mask = ann_to_binary_mask(ann, h, w)
 
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
 
            crop_rgb = image[y0:y1, x0:x1]
            crop_mask = mask[y0:y1, x0:x1] * 255
 
            rgba = np.dstack([crop_rgb, crop_mask]).astype(np.uint8)
            inst_id = f"{cls_name}_{ann['id']}"
            out_path = os.path.join(cls_dir, f"{inst_id}.png")
            Image.fromarray(rgba, mode="RGBA").save(out_path)
 
            entries.append({"id": inst_id, "path": out_path,
                             "w": int(x1 - x0), "h": int(y1 - y0)})
            total += 1
 
        index[cls_name] = entries
        print(f"{cls_name}: extracted {len(entries)} instances")
 
    with open(os.path.join(args.out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
 
    print(f"Done. {total} instances written to {args.out_dir}")
 
 
if __name__ == "__main__":
    main()
 

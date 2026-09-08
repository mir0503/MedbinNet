"""
Targeted copy-paste augmentation for MMDetection (mmdet >= 3.x pipelines).
 
Pastes pre-extracted instances of chosen weak classes onto training images,
to boost their effective sample count without touching classes that are
already performing well. Designed to plug into an MMDetection train_pipeline.
 
Requires the instance pool built by extract_instance_pool.py first.
 
Add to your config after the load/annotation steps, e.g.:
 
    custom_imports = dict(imports=['copy_paste_transform'], allow_failed_imports=False)
 
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
        dict(type='TargetedCopyPaste',
             pool_dir='./copypaste_pool',
             max_paste=2,
             paste_prob=0.5,
             scale_range=(0.6, 1.3),
             blend_blur=3,
             class_name_to_id={'transferpottor_glass': 12, 'transferpottor_plastic': 13}),
        dict(type='Resize', scale=(640, 640), keep_ratio=True),
        ...
    ]
 
class_name_to_id must match the label ids your dataset config already
uses (i.e. the index of each class name in metainfo['classes']).
 
Note: this targets the mmdet 3.x BaseTransform / dict-results pipeline API.
If you're on mmdet 2.x, the results dict keys and mask container class
differ slightly (gt_masks was a PolygonMasks/BitmapMasks under a different
import path) — flag that and I can adapt this for your exact version.
"""
import json
import os
import random
 
import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks
 
 
@TRANSFORMS.register_module()
class TargetedCopyPaste(BaseTransform):
    def __init__(self,
                 pool_dir,
                 max_paste=2,
                 paste_prob=0.5,
                 scale_range=(0.6, 1.3),
                 rotate_range=(-15, 15),
                 blend_blur=3,
                 max_iou_with_existing=0.3,
                 class_name_to_id=None):
        """
        Args:
            pool_dir: folder produced by extract_instance_pool.py.
            max_paste: max number of instances pasted per image.
            paste_prob: probability this augmentation is applied at all
                for a given image.
            scale_range: random scale factor applied to each pasted
                instance relative to its original crop size.
            rotate_range: random rotation in degrees applied before paste.
            blend_blur: odd kernel size for Gaussian-blurring the alpha
                mask edge before compositing, so the paste doesn't leave
                a hard visible seam the model could learn to key on.
            max_iou_with_existing: skip a candidate placement if it would
                overlap an existing ground-truth box by more than this.
            class_name_to_id: REQUIRED dict mapping class name -> label id,
                matching your dataset config's class ordering.
        """
        with open(os.path.join(pool_dir, "index.json")) as f:
            self.pool = json.load(f)
        self.pool_classes = list(self.pool.keys())
        self.max_paste = max_paste
        self.paste_prob = paste_prob
        self.scale_range = scale_range
        self.rotate_range = rotate_range
        self.blend_blur = blend_blur
        self.max_iou_with_existing = max_iou_with_existing
 
        if class_name_to_id is None:
            raise ValueError(
                "class_name_to_id is required — pass the same class->id "
                "mapping used by your dataset config (metainfo['classes']).")
        self.class_name_to_id = class_name_to_id
 
    def _load_instance(self, cls_name):
        entry = random.choice(self.pool[cls_name])
        rgba = cv2.imread(entry["path"], cv2.IMREAD_UNCHANGED)
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
        return rgba
 
    def _transform_instance(self, rgba):
        h, w = rgba.shape[:2]
        scale = random.uniform(*self.scale_range)
        angle = random.uniform(*self.rotate_range)
 
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
 
        center = (new_w / 2, new_h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        rot_w = int(new_h * sin + new_w * cos)
        rot_h = int(new_h * cos + new_w * sin)
        M[0, 2] += (rot_w / 2) - center[0]
        M[1, 2] += (rot_h / 2) - center[1]
        rgba = cv2.warpAffine(rgba, M, (rot_w, rot_h),
                               flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        return rgba
 
    def _feather_alpha(self, alpha):
        if self.blend_blur and self.blend_blur > 1:
            k = self.blend_blur | 1  # force odd kernel size
            alpha = cv2.GaussianBlur(alpha.astype(np.float32), (k, k), 0)
        return np.clip(alpha, 0, 255).astype(np.uint8)
 
    @staticmethod
    def _bbox_iou(box_a, box_b):
        xa0, ya0, xa1, ya1 = box_a
        xb0, yb0, xb1, yb1 = box_b
        ix0, iy0 = max(xa0, xb0), max(ya0, yb0)
        ix1, iy1 = min(xa1, xb1), min(ya1, yb1)
        iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
        inter = iw * ih
        area_a = max(0, xa1 - xa0) * max(0, ya1 - ya0)
        area_b = max(0, xb1 - xb0) * max(0, yb1 - yb0)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
 
    def transform(self, results):
        if random.random() > self.paste_prob:
            return results
 
        img = results["img"]
        H, W = img.shape[:2]
        n_paste = random.randint(1, self.max_paste)
 
        gt_bboxes = results["gt_bboxes"]
        existing_boxes = gt_bboxes.tensor.numpy().tolist() \
            if hasattr(gt_bboxes, "tensor") else np.array(gt_bboxes).tolist()
 
        new_boxes, new_labels, new_masks = [], [], []
 
        for _ in range(n_paste):
            cls_name = random.choice(self.pool_classes)
            rgba = self._load_instance(cls_name)
            rgba = self._transform_instance(rgba)
            ph, pw = rgba.shape[:2]
            if ph >= H or pw >= W:
                continue
 
            x0 = random.randint(0, W - pw)
            y0 = random.randint(0, H - ph)
            candidate_box = [x0, y0, x0 + pw, y0 + ph]
 
            if any(self._bbox_iou(candidate_box, b) > self.max_iou_with_existing
                   for b in existing_boxes):
                continue
 
            rgb = rgba[..., :3]
            alpha = self._feather_alpha(rgba[..., 3])
            alpha_f = (alpha / 255.0)[..., None]
 
            roi = img[y0:y0 + ph, x0:x0 + pw].astype(np.float32)
            blended = rgb.astype(np.float32) * alpha_f + roi * (1 - alpha_f)
            img[y0:y0 + ph, x0:x0 + pw] = blended.astype(np.uint8)
 
            full_mask = np.zeros((H, W), dtype=np.uint8)
            full_mask[y0:y0 + ph, x0:x0 + pw] = (alpha > 127).astype(np.uint8)
 
            new_boxes.append(candidate_box)
            new_labels.append(self.class_name_to_id[cls_name])
            new_masks.append(full_mask)
            existing_boxes.append(candidate_box)
 
        if new_boxes:
            results["img"] = img
 
            old_boxes_arr = gt_bboxes.numpy() if hasattr(gt_bboxes, "numpy") else np.array(gt_bboxes)
            all_boxes = np.concatenate(
                [old_boxes_arr, np.array(new_boxes, dtype=np.float32)], axis=0)
            results["gt_bboxes"] = type(gt_bboxes)(all_boxes)
 
            all_labels = np.concatenate(
                [np.array(results["gt_bboxes_labels"]),
                 np.array(new_labels, dtype=np.int64)])
            results["gt_bboxes_labels"] = all_labels
 
            if "gt_masks" in results:
                old_masks = results["gt_masks"].masks
                stacked = np.concatenate(
                    [old_masks, np.stack(new_masks, axis=0)], axis=0)
                results["gt_masks"] = BitmapMasks(stacked, H, W)
 
        return results
 

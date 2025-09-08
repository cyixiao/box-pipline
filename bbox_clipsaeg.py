#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, pathlib, re
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# ================== 路径配置 ==================
ROOT = "/home/cyixiao/Project/videollm/pipline"
IN_META  = f"{ROOT}/datasets/keyframes_minerva_gpt4o_labels.json"              # 含 frames + frame_labels
OUT_META = f"{ROOT}/datasets/keyframes_minerva_gpt4o_boxes.json"   # 输出（与 GPT 版本同名/同结构）

# ================== 设备 & 模型 ==================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIPSEG_NAME = "CIDAS/clipseg-rd64-refined"

# ================== 写回尺度 ==================
SCALE_MAX = 1000  # 0-1000 的整数坐标

# ================== 分割/筛选参数（可按需微调） ==================
# 阈值扫描：从多个阈值里挑一个“更紧且可靠”的掩码
# THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
THRESHOLDS = [0.25, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
AREA_MIN = 0.002      # 掩码面积占比下限
AREA_MAX = 0.90      # 掩码面积占比上限（避免整图）
CENTER_BONUS = 0.05  # 掩码重心越靠近图像中心加一点分
SIZE_PRIOR = 0.25    # 偏好中等面积（越接近此值加分）

# ================ 工具函数 =================
def ensure_parent(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return [x1, y1, x2, y2]

def pix_to_1000(box_pix: List[int], W: int, H: int) -> List[int]:
    x1, y1, x2, y2 = box_pix
    return [
        int(round(x1 / W * SCALE_MAX)),
        int(round(y1 / H * SCALE_MAX)),
        int(round(x2 / W * SCALE_MAX)),
        int(round(y2 / H * SCALE_MAX)),
    ]

def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    x2 = max(x2, x1 + 1); y2 = max(y2, y1 + 1)
    return [x1, y1, x2, y2]

def score_mask(prob_map: torch.Tensor, mask: np.ndarray, W: int, H: int) -> float:
    """越大越好：均值置信度 + 居中 + 面积合理性"""
    h, w = mask.shape
    area = mask.sum()
    ar = float(area) / (h * w + 1e-6)
    if ar < AREA_MIN or ar > AREA_MAX:  # 过小/过大丢弃
        return -1e9
    if area == 0:
        return -1e9
    mean_prob = float(prob_map[mask].mean().item())
    ys, xs = np.where(mask)
    bx, by = (xs.mean() if len(xs) else w/2), (ys.mean() if len(ys) else h/2)
    cx, cy = w / 2.0, h / 2.0
    d = ((bx - cx)**2 + (by - cy)**2) ** 0.5
    d_norm = d / (max(w, h) + 1e-6)
    center_score = (1.0 - d_norm) * CENTER_BONUS
    size_score = (1.0 - abs(ar - SIZE_PRIOR)) * 0.1
    return mean_prob + center_score + size_score

# ================ CLIPSeg 包装 ================
class ClipSeg:
    def __init__(self, name: str, device: str):
        self.processor = CLIPSegProcessor.from_pretrained(name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(name).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def infer_one_label(self, image_pil: Image.Image, label: str) -> Optional[List[int]]:
        """
        返回像素坐标 bbox [x1,y1,x2,y2]；失败返回 None
        """
        if not label or not str(label).strip():
            return None
        text = [str(label).strip()]
        inputs = self.processor(text=text, images=image_pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # may be [N,1,Hc,Wc] or [N,Hc,Wc]
        if logits.ndim == 3:      # [N,Hc,Wc] -> [N,1,Hc,Wc]
            logits = logits.unsqueeze(1)
        elif logits.ndim != 4:
            return None

        W, H = image_pil.size
        logits_up = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)  # [N,1,H,W]
        prob_up = torch.sigmoid(logits_up)[0, 0]  # [H,W]
        p_np = prob_up.detach().cpu().numpy()

        best_score = -1e9
        best_box = None
        for t in THRESHOLDS:
            mask = (p_np >= t)
            box = mask_to_bbox(mask)
            if box is None:
                continue
            score = score_mask(prob_up, mask, W, H)
            if score > best_score:
                best_score, best_box = score, clamp_xyxy(*box, W=W, H=H)
        return best_box

# ================ 主流程 =================
def main():
    seg = ClipSeg(CLIPSEG_NAME, DEVICE)

    if not pathlib.Path(IN_META).exists():
        raise FileNotFoundError(f"missing: {IN_META}")
    items: List[Dict[str, Any]] = json.load(open(IN_META, "r"))

    updated = []
    for rec in items:
        vid = rec.get("video_id")
        frames: List[str] = rec.get("frames", []) or []
        labels: List[str] = rec.get("frame_labels", []) or []

        if not frames:
            updated.append(rec); continue

        bbox_pix_list:  List[List[int]] = []
        bbox_1000_list: List[List[int]] = []
        fallback = 0

        for i, fp in enumerate(frames):
            if not pathlib.Path(fp).exists():
                bbox_pix_list.append([0,0,0,0])
                bbox_1000_list.append([0,0,SCALE_MAX,SCALE_MAX])
                fallback += 1
                continue

            img = Image.open(fp).convert("RGB")
            W, H = img.size

            # 优先使用 frame_labels[i]；如果缺失，则简单兜底为整图
            label = labels[i] if i < len(labels) else None
            box_pix = seg.infer_one_label(img, label) if label else None

            if box_pix is None:
                box_pix = [0, 0, W-1, H-1]
                fallback += 1

            bbox_pix_list.append(box_pix)
            bbox_1000_list.append(pix_to_1000(box_pix, W, H))

        # 写回（与 GPT 版本一致）
        rec.pop("bbox", None)
        rec["bbox_xyxy_1000"] = bbox_1000_list
        rec["bbox_xyxy_pix"]  = bbox_pix_list
        updated.append(rec)

        print(f"✓ {vid}: frames={len(frames)}  fallback={fallback}")
        for i, (n, p) in enumerate(zip(bbox_1000_list, bbox_pix_list), 1):
            print(f"  frame_{i}: norm1000=[{n[0]},{n[1]},{n[2]},{n[3]}]  pix=[{p[0]},{p[1]},{p[2]},{p[3]}]")

    ensure_parent(OUT_META)
    with open(OUT_META, "w") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"\nDONE. Saved to {OUT_META}. {len(updated)} samples updated.")

if __name__ == "__main__":
    main()
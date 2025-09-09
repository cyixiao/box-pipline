#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, pathlib, re
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ================== 路径配置 ==================
ROOT = "/home/cyixiao/Project/videollm/pipline"
IN_META  = f"{ROOT}/datasets/train/minerva_labels.json"   # 含 frames (+可选 frame_labels)
OUT_META = f"{ROOT}/datasets/train/minerva_boxes.json"    # 输出（写回 bbox_xyxy_*）

# ================== 模型与设备 ==================
MODEL_ID = "IDEA-Research/grounding-dino-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================== 检测与阈值 ==================
# 文本必须小写 + 每个短语以 '.' 结尾（官方强烈建议）
BOX_THRESHOLD  = 0.30   # 框分数阈值（可调低到 0.30 以照顾小目标）
TEXT_THRESHOLD = 0.25   # 文本匹配阈值

# ================== 工具函数 ==================
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

def clean_label(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)   # 去标点
    s = re.sub(r"\s+", " ", s)
    return s

def build_query_text(label: str) -> str:
    """
    Grounding DINO 约定：每个短语小写、以 '.' 结尾；可以多个短语用 '. ' 串起来。
    我们这里每次只查一个主标签；若失败再用同义词或 FALLBACK_LABELS。
    """
    label = clean_label(label)
    if not label:
        return ""
    if not label.endswith("."):
        label = label + "."
    return label

def pick_best_box(results: Dict[str, Any], want_label: str) -> Optional[List[float]]:
    """
    在 post_process 结果里，筛选出与 want_label 匹配且分数最高的一框。
    results 结构如：{'boxes': Tensor[N,4], 'labels': [...], 'scores': Tensor[N]}
    """
    if not results or "boxes" not in results:
        return None
    boxes = results["boxes"]    # [N,4], xyxy (像素，已按 target_sizes 反缩放)
    labels = results.get("labels", [])
    scores = results.get("scores", None)

    best_idx = -1
    best_score = -1.0
    for i in range(len(boxes)):
        lab_i = str(labels[i]).lower().strip() if i < len(labels) else ""
        if lab_i != want_label:
            continue
        sc = float(scores[i]) if scores is not None else 0.0
        if sc > best_score:
            best_score = sc
            best_idx = i
    if best_idx < 0:
        return None
    return boxes[best_idx].tolist()  # [x1,y1,x2,y2] in pixels

# ================== 推理核心 ==================
class GroundingDINO:
    def __init__(self, model_id: str, device: str):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device

    @torch.no_grad()
    def detect_one(self, image: Image.Image, label: str) -> Optional[List[int]]:
        """
        用单个标签做一次检测，返回像素坐标的 xyxy；失败返回 None
        """
        want = clean_label(label)
        query = build_query_text(want)
        if not query:
            return None

        W, H = image.size
        inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        processed = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(H, W)],  # 注意顺序是 (height, width)
        )
        # 这里只有一张图，取第 0 个结果
        res0 = processed[0] if processed else None
        if not res0:
            return None

        # labels 是文本短语；因为我们只传了一个短语，直接挑分最高的也可。
        # 但为了一致性，这里严格匹配 want。
        box = pick_best_box(res0, want)
        if box is None:
            return None

        x1, y1, x2, y2 = box
        return clamp_xyxy(x1, y1, x2, y2, W, H)

    @torch.no_grad()
    def detect_with_fallbacks(self, image: Image.Image, primary: Optional[str]) -> Optional[List[int]]:
        """
        只使用提供的 primary label；若无或检测失败则返回 None。
        """
        if not primary or not str(primary).strip():
            return None
        return self.detect_one(image, primary)

# ================== 主流程 ==================
def main():
    detector = GroundingDINO(MODEL_ID, DEVICE)

    if not pathlib.Path(IN_META).exists():
        raise FileNotFoundError(f"missing: {IN_META}")
    items: List[Dict[str, Any]] = json.load(open(IN_META, "r"))

    total_frames = 0
    total_fallbacks = 0

    updated = []
    for rec in items:
        vid = rec.get("video_id")
        frames: List[str] = rec.get("frames", []) or []
        labels: List[str] = rec.get("frame_labels", []) or []

        if not frames:
            updated.append(rec); continue

        bbox_pix_list:  List[List[int]] = []
        fallback = 0
        missed = 0

        print(f"\n✓ {vid}: {len(frames)} frames")
        for i, fp in enumerate(frames):
            total_frames += 1
            if not pathlib.Path(fp).exists():
                bbox_pix_list.append([0,0,0,0])
                missed += 1
                print(f"  frame_{i+1}: missing -> full image")
                continue

            img = Image.open(fp).convert("RGB")
            W, H = img.size

            label = labels[i] if i < len(labels) else None
            box_pix = detector.detect_with_fallbacks(img, label)

            if box_pix is None:
                # 兜底：整图
                box_pix = [0, 0, W-1, H-1]
                fallback += 1
                total_fallbacks += 1
                print(f"  frame_{i+1}: label='{label}' -> fallback full image")
            else:
                print(f"  frame_{i+1}: label='{label}' -> box={box_pix}")

            bbox_pix_list.append(box_pix)

        # 写回
        rec.pop("bbox", None)
        rec["bbox_xyxy_pix"]  = bbox_pix_list
        updated.append(rec)

        print(f"  summary: fallback_full_image={fallback}, missing_files={missed}")

    ensure_parent(OUT_META)
    with open(OUT_META, "w") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"\nDONE. Saved to {OUT_META}. {len(updated)} samples updated.")

    if total_frames > 0:
        percent = total_fallbacks / total_frames * 100
        print(f"Fallback to full image in {total_fallbacks}/{total_frames} frames ({percent:.2f}%).")

if __name__ == "__main__":
    main()
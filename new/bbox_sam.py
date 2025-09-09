#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, pathlib, re, argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

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
    我们这里每次只查一个主标签。
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

# ================== 模型封装 ==================
class GroundingDINO:
    def __init__(self, model_id: str, device: str, box_threshold: float, text_threshold: float):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

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
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(H, W)],  # 注意顺序是 (height, width)
        )
        # 这里只有一张图，取第 0 个结果
        res0 = processed[0] if processed else None
        if not res0:
            return None

        # 严格匹配 want（labels 里是文本短语）
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

# ================== CLI ==================
def parse_args():
    ap = argparse.ArgumentParser(description="Zero-shot object detection for keyframes (Grounding DINO), write pixel xyxy boxes.")
    # 路径
    ap.add_argument("--in-meta", required=True, help="输入 JSON（含帧路径、可选帧标签）")
    ap.add_argument("--output", required=True, help="输出 JSON（写入 bbox 字段）")
    # 字段映射
    ap.add_argument("--field.frames", dest="f_frames", required=True, help="字段名：frames 列表")
    ap.add_argument("--field.frame_labels", dest="f_frame_labels", required=True, help="字段名：frame_labels 列表")
    ap.add_argument("--field.bboxes-out", dest="f_bboxes_out", default="bbox_pix", help="输出 bbox 字段名（默认：bbox_pix）")
    # 模型与阈值
    ap.add_argument("--model-id", default="IDEA-Research/grounding-dino-base", help="模型ID（默认：IDEA-Research/grounding-dino-base）")
    ap.add_argument("--box-threshold", type=float, default=0.30, help="框分数阈值（默认 0.30）")
    ap.add_argument("--text-threshold", type=float, default=0.25, help="文本匹配阈值（默认 0.25）")
    return ap.parse_args()

# ================== 主流程 ==================
def main():
    args = parse_args()

    # 设备自动选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = GroundingDINO(args.model_id, device, args.box_threshold, args.text_threshold)

    if not pathlib.Path(args.in_meta).exists():
        raise FileNotFoundError(f"missing: {args.in_meta}")
    items: List[Dict[str, Any]] = json.load(open(args.in_meta, "r"))

    total_frames = 0
    total_fallbacks = 0

    updated = []
    for rec in items:
        frames: List[str] = rec.get(args.f_frames, []) or []
        labels: List[str] = rec.get(args.f_frame_labels, []) or []

        if not frames:
            updated.append(rec); continue

        bbox_pix_list:  List[List[int]] = []
        fallback = 0
        missed = 0

        print(f"\n✓ frames={len(frames)}")
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
        rec[args.f_bboxes_out]  = bbox_pix_list
        updated.append(rec)

        print(f"  summary: fallback_full_image={fallback}, missing_files={missed}")

    ensure_parent(args.output)
    with open(args.output, "w") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"\nDONE. Saved to {args.output}. {len(updated)} samples updated.")

    if total_frames > 0:
        percent = total_fallbacks / total_frames * 100
        print(f"Fallback to full image in {total_fallbacks}/{total_frames} frames ({percent:.2f}%).")

if __name__ == "__main__":
    main()
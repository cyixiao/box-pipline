#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, pathlib
from typing import List, Dict, Any
from PIL import Image, ImageDraw

# ================== 路径配置（按需改） ==================
ROOT = "/home/cyixiao/Project/videollm/pipline"

# 关键帧 + 坐标的元数据（上一步生成的）
IN_KEYFRAMES_META = f"{ROOT}/datasets/train/minerva_boxes.json"

# 之前保存的抽样元数据（需要在里面新增两个字段）
SAMPLED_META_IN  = f"{ROOT}/datasets/train/minerva.json"
SAMPLED_META_OUT = f"{ROOT}/datasets/train/minerva_latent.json"   # 想覆盖就把 OUT 改成 IN

# 输出图片目录
OUT_ENLARGE_DIR = f"{ROOT}/datasets/train/latent/enlarge"
OUT_BBOX_DIR    = f"{ROOT}/datasets/train/latent/bbox"

# ================== 可调参数 ==================
DRAW_WIDTH = 3            # 画框线宽
PADDING_PIX = 0           # enlarge 裁剪时的四周 padding，像素（0 表示无 padding）
RESIZE_SHORT = 0          # enlarge 输出短边统一到该尺寸（0 表示不缩放）

# =================================================
def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W - 1))
    y1 = max(0, min(int(y1), H - 1))
    x2 = max(0, min(int(x2), W - 1))
    y2 = max(0, min(int(y2), H - 1))
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2

def add_padding(x1, y1, x2, y2, pad, W, H):
    if pad <= 0:
        return clamp_box(x1, y1, x2, y2, W, H)
    return clamp_box(x1 - pad, y1 - pad, x2 + pad, y2 + pad, W, H)

def save_enlarge(img_path: str, box_pix: List[int], out_path: str):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = clamp_box(*box_pix, W, H)
    x1, y1, x2, y2 = add_padding(x1, y1, x2, y2, PADDING_PIX, W, H)
    crop = img.crop((x1, y1, x2, y2))
    if RESIZE_SHORT and min(crop.size) > 0:
        w, h = crop.size
        if w <= h:
            new_w = RESIZE_SHORT
            new_h = int(round(h * (RESIZE_SHORT / w)))
        else:
            new_h = RESIZE_SHORT
            new_w = int(round(w * (RESIZE_SHORT / h)))
        crop = crop.resize((new_w, new_h), Image.BICUBIC)
    ensure_dir(str(pathlib.Path(out_path).parent))
    crop.save(out_path)

def save_bbox_vis(img_path: str, box_pix: List[int], out_path: str):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box_pix
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=DRAW_WIDTH)
    ensure_dir(str(pathlib.Path(out_path).parent))
    img.save(out_path)

def main():
    # 读取 keyframes + bboxes
    if not pathlib.Path(IN_KEYFRAMES_META).exists():
        raise FileNotFoundError(f"missing: {IN_KEYFRAMES_META}")
    kf_data: List[Dict[str, Any]] = json.load(open(IN_KEYFRAMES_META, "r"))

    # 建索引：video_id -> (frames, bbox_pix_list)
    kf_index = {}
    for rec in kf_data:
        vid = rec.get("video_id")
        frames = rec.get("frames", [])
        bxy_pix = rec.get("bbox_xyxy_pix") or rec.get("bbox_xyxy_pix".lower())  # 容错
        if not isinstance(frames, list) or not isinstance(bxy_pix, list):
            continue
        kf_index[vid] = (frames, bxy_pix)

    # 读取 sampled_meta
    if not pathlib.Path(SAMPLED_META_IN).exists():
        raise FileNotFoundError(f"missing: {SAMPLED_META_IN}")
    sm_data: List[Dict[str, Any]] = json.load(open(SAMPLED_META_IN, "r"))

    updated = []
    vid_counter: Dict[str, int] = {}
    for item in sm_data:
        vid = item.get("video_id")
        # 为同一视频的第几个问题分配递增编号（_1, _2, ...）
        qidx = vid_counter.get(vid, 0) + 1
        vid_counter[vid] = qidx
        vid_q = f"{vid}_{qidx}"
        if vid not in kf_index:
            # 没有关键帧/坐标的样本直接保留原样
            updated.append(item)
            continue

        frames, bboxes_pix = kf_index[vid]
        n = min(len(frames), len(bboxes_pix))
        enlarge_paths: List[str] = []
        bbox_paths:    List[str] = []

        for i in range(n):
            frame_path = frames[i]
            box = bboxes_pix[i]
            # 输出文件名（每个问题单独文件夹：video_id_1, video_id_2, ...；文件名为 keyframe_1.png ...）
            subdir_enlarge_abs = pathlib.Path(OUT_ENLARGE_DIR) / vid_q
            subdir_bbox_abs    = pathlib.Path(OUT_BBOX_DIR) / vid_q

            # 相对路径前缀，写入 JSON 时不带绝对路径，只以 /latent/... 开头
            rel_enlarge_prefix = f"/latent/enlarge/{vid_q}"
            rel_bbox_prefix    = f"/latent/bbox/{vid_q}"

            filename = f"keyframe_{i+1}.png"
            out_enlarge_abs = str(subdir_enlarge_abs / filename)
            out_bbox_abs    = str(subdir_bbox_abs / filename)
            out_enlarge_rel = f"{rel_enlarge_prefix}/{filename}"
            out_bbox_rel    = f"{rel_bbox_prefix}/{filename}"

            # 生成并保存
            try:
                save_enlarge(frame_path, box, out_enlarge_abs)
                save_bbox_vis(frame_path, box, out_bbox_abs)
                # 写回 JSON 时仅保存相对路径（/latent/...）
                enlarge_paths.append(out_enlarge_rel)
                bbox_paths.append(out_bbox_rel)
            except Exception as e:
                print(f"⚠️ {vid} frame_{i+1} failed: {e}")

        # 把两组路径写回到该样本
        # ——键名严格按你要求（注意 "boundingbox _latent" 中间有空格）
        item["enlarge_latent"] = enlarge_paths
        item["boundingbox _latent"] = bbox_paths

        updated.append(item)
        print(f"✓ {vid_q}: enlarge={len(enlarge_paths)}  bbox={len(bbox_paths)}")

    # 保存新版本 sampled_meta
    out_path = SAMPLED_META_OUT
    pathlib.Path(pathlib.Path(out_path).parent).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"\nDONE. Saved to {out_path}. Total {len(updated)} samples.")

if __name__ == "__main__":
    # 准备输出目录
    ensure_dir(OUT_ENLARGE_DIR)
    ensure_dir(OUT_BBOX_DIR)
    main()
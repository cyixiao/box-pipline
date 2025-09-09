import argparse, json, pathlib
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def ensure_parent(p: str):
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)

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

def save_enlarge(img_path: str, box_pix: List[int], out_path: str,
                 padding_pix: int, resize_short: int):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = clamp_box(*box_pix, W, H)
    x1, y1, x2, y2 = add_padding(x1, y1, x2, y2, padding_pix, W, H)
    crop = img.crop((x1, y1, x2, y2))
    if resize_short and min(crop.size) > 0:
        w, h = crop.size
        if w <= h:
            new_w = resize_short
            new_h = int(round(h * (resize_short / w)))
        else:
            new_h = resize_short
            new_w = int(round(w * (resize_short / h)))
        crop = crop.resize((new_w, new_h), Image.BICUBIC)
    ensure_parent(out_path)
    crop.save(out_path)

def save_bbox_vis(img_path: str, box_pix: List[int], out_path: str, draw_width: int):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = box_pix
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=draw_width)
    ensure_parent(out_path)
    img.save(out_path)

# ================ CLI ================
def parse_args():
    ap = argparse.ArgumentParser(description="Generate enlarge crops and bbox-visualization images, and write relative paths back to metadata.")
    # 路径
    ap.add_argument("--in-keyframes", required=True, help="输入 JSON（包含 frames 和 bboxes）")
    ap.add_argument("--in-meta", required=True, help="输入原始样本 JSON（需写回输出路径）")
    ap.add_argument("--output", required=True, help="输出 JSON（写入 enlarge/bbox 相对路径）")
    ap.add_argument("--out-enlarge-dir", required=True, help="enlarge 裁剪图输出目录（绝对路径）")
    ap.add_argument("--out-bbox-dir", required=True, help="bbox 可视化图输出目录（绝对路径）")

    # 字段映射
    ap.add_argument("--field.video_id", dest="f_video_id", required=True, help="字段名：视频ID")
    ap.add_argument("--field.question_id", dest="f_question_id", required=True, help="字段名：问题ID（用于匹配 keyframes 记录）")
    ap.add_argument("--field.frames", dest="f_frames", required=True, help="字段名：在 keyframes JSON 中的帧列表")
    ap.add_argument("--field.bboxes", dest="f_bboxes", required=True, help="字段名：在 keyframes JSON 中的 bbox 列表（像素 xyxy）")
    ap.add_argument("--field.enlarge-out", dest="f_enlarge_out", default="enlarge_latent", help="写回到样本 JSON 的 enlarge 字段名（默认：enlarge_latent）")
    ap.add_argument("--field.bbox-out", dest="f_bbox_out", default="boundingbox _latent", help="写回到样本 JSON 的 bbox 字段名（默认：'boundingbox _latent'，注意有空格）")

    # 相对路径与文件命名
    ap.add_argument("--rel-root-enlarge", default="/latent/enlarge", help="写回 JSON 的 enlarge 相对根前缀（默认：/latent/enlarge）")
    ap.add_argument("--rel-root-bbox", default="/latent/bbox", help="写回 JSON 的 bbox 相对根前缀（默认：/latent/bbox）")
    ap.add_argument("--filename-template", default="keyframe_{i}.png", help="输出文件名模板，支持 {i}（从1开始）")

    # 图像可视化与裁剪
    ap.add_argument("--draw-width", type=int, default=3, help="画框线宽（默认 3）")
    ap.add_argument("--padding-pix", type=int, default=0, help="enlarge 裁剪 padding 像素（默认 0）")
    ap.add_argument("--resize-short", type=int, default=0, help="enlarge 输出短边统一到该尺寸（0 表示不缩放）")
    return ap.parse_args()

# ================ 主流程 ================
def main():
    args = parse_args()

    # 准备输出目录
    ensure_dir(args.out_enlarge_dir)
    ensure_dir(args.out_bbox_dir)

    # 读取 keyframes+bboxes
    if not pathlib.Path(args.in_keyframes).exists():
        raise FileNotFoundError(f"missing: {args.in_keyframes}")
    kf_data: List[Dict[str, Any]] = json.load(open(args.in_keyframes, "r"))

    # 建索引： (video_id, question_id) -> (frames, bboxes)
    kf_index: Dict[Tuple[str, str], Tuple[List[str], List[List[int]]]] = {}
    for rec in kf_data:
        vid = str(rec.get(args.f_video_id))
        qid = str(rec.get(args.f_question_id, ""))
        frames = rec.get(args.f_frames, []) or []
        bxy_pix = rec.get(args.f_bboxes, []) or []
        if not vid or not isinstance(frames, list) or not isinstance(bxy_pix, list):
            continue
        kf_index[(vid, qid)] = (frames, bxy_pix)

    # 读取样本 meta（按其顺序生成 qidx 并写回）
    if not pathlib.Path(args.in_meta).exists():
        raise FileNotFoundError(f"missing: {args.in_meta}")
    sm_data: List[Dict[str, Any]] = json.load(open(args.in_meta, "r"))

    updated = []
    dropped_missing = 0      # (vid,qid) not found in keyframes index
    dropped_no_enlarge = 0   # processed but produced 0 enlarge images
    vid_counter: Dict[str, int] = {}

    for item in sm_data:
        vid = str(item.get(args.f_video_id))
        qid = str(item.get(args.f_question_id, ""))

        qidx = vid_counter.get(vid, 0) + 1
        vid_counter[vid] = qidx
        subdir_name = f"{vid}_{qidx}" 
        key = (vid, qid)
        if key not in kf_index:
            dropped_missing += 1
            print(f"✗ DROP {subdir_name}: no keyframes/bboxes for (vid={vid}, qid={qid})")
            continue

        frames, bboxes_pix = kf_index[key]
        n = min(len(frames), len(bboxes_pix))
        enlarge_paths: List[str] = []
        bbox_paths:    List[str] = []

        for i in range(n):
            frame_path = frames[i]
            box = bboxes_pix[i]
            # 绝对输出目录
            subdir_enlarge_abs = pathlib.Path(args.out_enlarge_dir) / subdir_name
            subdir_bbox_abs    = pathlib.Path(args.out_bbox_dir) / subdir_name

            # 相对路径（写回 JSON）
            rel_enlarge_prefix = f"{args.rel_root_enlarge.rstrip('/')}/{subdir_name}"
            rel_bbox_prefix    = f"{args.rel_root_bbox.rstrip('/')}/{subdir_name}"

            filename = args.filename_template.format(i=i+1)
            out_enlarge_abs = str(subdir_enlarge_abs / filename)
            out_bbox_abs    = str(subdir_bbox_abs / filename)
            out_enlarge_rel = f"{rel_enlarge_prefix}/{filename}"
            out_bbox_rel    = f"{rel_bbox_prefix}/{filename}"

            try:
                save_enlarge(frame_path, box, out_enlarge_abs, args.padding_pix, args.resize_short)
                save_bbox_vis(frame_path, box, out_bbox_abs, args.draw_width)
                enlarge_paths.append(out_enlarge_rel)
                bbox_paths.append(out_bbox_rel)
            except Exception as e:
                print(f"{subdir_name} frame_{i+1} failed: {e}")

        if len(enlarge_paths) == 0:
            dropped_no_enlarge += 1
            print(f"✗ DROP {subdir_name}: no enlarge saved")
            continue

        item[args.f_enlarge_out] = enlarge_paths
        item[args.f_bbox_out]    = bbox_paths

        updated.append(item)
        print(f"✓ {subdir_name}: enlarge={len(enlarge_paths)}  bbox={len(bbox_paths)}")

    ensure_parent(args.output)
    with open(args.output, "w") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    total_dropped = dropped_missing + dropped_no_enlarge
    print(f"Dropped samples: {total_dropped} (missing kf/bbox: {dropped_missing}, no enlarge saved: {dropped_no_enlarge}). Kept: {len(updated)}")
    print(f"\nDONE. Saved to {args.output}. Total {len(updated)} samples.")

if __name__ == "__main__":
    main()
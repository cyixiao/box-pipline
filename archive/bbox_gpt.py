import os, json, pathlib, base64
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
from openai import OpenAI

ROOT = "/home/cyixiao/Project/videollm/pipline"
IN_META = f"{ROOT}/datasets/keyframes_meta_gpt4o.json"
OUT_META = f"{ROOT}/datasets/keyframes_meta_gpt4o_with_boxes.json"

MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_BOXES_PER_FRAME = 1
CONF_CLAMP = (0, 1000)

def ensure_parent(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def pil_size(img_path: str) -> Tuple[int, int]:
    with Image.open(img_path) as im:
        return im.size  # (W, H)

def image_to_data_url(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    ext = pathlib.Path(img_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"

def clamp1000(x: float) -> int:
    lo, hi = CONF_CLAMP
    try:
        v = float(x)
    except Exception:
        v = 0.0
    v = int(round(v))
    return max(lo, min(hi, v))

def norm_to_pixel(box: List[float], W: int, H: int) -> List[int]:
    x1 = int(round(clamp1000(box[0]) / 1000.0 * W))
    y1 = int(round(clamp1000(box[1]) / 1000.0 * H))
    x2 = int(round(clamp1000(box[2]) / 1000.0 * W))
    y2 = int(round(clamp1000(box[3]) / 1000.0 * H))

    x1, x2 = max(0, min(x1, W-1)), max(0, min(x2, W-1))
    y1, y2 = max(0, min(y1, H-1)), max(0, min(y2, H-1))
    if x2 <= x1: x2 = min(W-1, x1 + 1)
    if y2 <= y1: y2 = min(H-1, y1 + 1)
    return [x1, y1, x2, y2]

def build_prompt(question: str, answer_choices: List[str]) -> Dict[str, Any]:
    lines = [
        "You are given a single video keyframe image and a QA task.",
        "Goal: return bounding boxes for the MINIMUM set of regions that are essential to answer the question.",
        f"Question: {question}",
    ]
    if answer_choices:
        lines.append("Choices:")
        for i, ch in enumerate(answer_choices):
            lines.append(f"- {chr(65+i)}. {ch}")
    lines += [
        "",
        "Rules:",
        "- Return exactly 1 box.",
        "- Always return one box even if uncertain; choose the most relevant region.",
        "- Coordinates must be integers in [0,1000] relative to the full image: [x1,y1,x2,y2], do NOT round to the nearest 10/50/100 unless the boundary truly aligns there.",
        "- Ensure x1 < x2 and y1 < y2; the box should tightly cover the key object/part needed for answering.",
        "JSON ONLY, no extra text. Schema: {\"bbox\":[x1,y1,x2,y2]}",
        "Example: {\"bbox\":[537,312,948,603]}"
    ]
    return {"type": "text", "text": "\n".join(lines)}

def call_gpt4o_for_box(client: OpenAI, img_path: str, question: str, choices: List[str]) -> Optional[List[int]]:
    content = [build_prompt(question, choices),
               {"type": "image_url", "image_url": {"url": image_to_data_url(img_path)}}]
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
    )
    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
        box = data.get("bbox")
        if not (isinstance(box, list) and len(box) == 4):
            return None
        norm1000 = [clamp1000(v) for v in box]
        return norm1000
    except Exception:
        return None


def main():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. `export OPENAI_API_KEY=...`")
    client = OpenAI(api_key=api_key)

    with open(IN_META, "r") as f:
        items = json.load(f)

    updated = []
    for rec in items:
        vid = rec.get("video_id")
        q = rec.get("question", "")
        frames: List[str] = rec.get("frames", [])
        if not frames:
            updated.append(rec); continue

        choices = []
        for i in range(5):
            k = f"answer_choice_{i}"
            if k in rec and rec[k]:
                choices.append(rec[k])

        bbox_1000_list: List[List[int]] = []
        bbox_pix_list:  List[List[int]] = []
        fallback_count = 0

        for fp in frames:
            if not pathlib.Path(fp).exists():
                box_norm1000 = [0, 0, 1000, 1000]
                fallback_count += 1
                W, H = (1, 1)
                box_pix = norm_to_pixel(box_norm1000, W, H)
                bbox_1000_list.append(box_norm1000)
                bbox_pix_list.append(box_pix)
                continue

            box_norm1000 = call_gpt4o_for_box(client, fp, q, choices)
            W, H = pil_size(fp)
            if box_norm1000 is None:
                box_norm1000 = [0, 0, 1000, 1000]
                fallback_count += 1
            box_pix = norm_to_pixel(box_norm1000, W, H)
            bbox_1000_list.append(box_norm1000)
            bbox_pix_list.append(box_pix)

        rec.pop("bbox", None)
        rec["bbox_xyxy_1000"] = bbox_1000_list    # [[x1,y1,x2,y2], ...] per frame (0-1000 ints)
        rec["bbox_xyxy_pix"]  = bbox_pix_list     # [[x1,y1,x2,y2], ...] per frame (pixels)
        updated.append(rec)
        print(f"✓ {vid}: frames={len(frames)}  fallback={fallback_count}")
        for i, (n, p) in enumerate(zip(bbox_1000_list, bbox_pix_list), 1):
            print(
                "  frame_{}: norm1000=[{},{},{},{}]  pix=[{},{},{},{}]".format(
                    i, n[0], n[1], n[2], n[3], p[0], p[1], p[2], p[3]
                )
            )

    ensure_parent(OUT_META)
    with open(OUT_META, "w") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    print(f"\nDONE. Saved to {OUT_META}. {len(updated)} samples updated.")

if __name__ == "__main__":
    main()
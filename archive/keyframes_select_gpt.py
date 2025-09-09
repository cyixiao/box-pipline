import os
import io
import json
import base64
import math
import pathlib
import subprocess
from typing import List, Tuple, Dict

from openai import OpenAI

ROOT = "/home/cyixiao/Project/videollm/pipline"
SAMPLED_META = f"{ROOT}/datasets/sampled_meta.json"
VIDEO_DIR = f"{ROOT}/videos/raw"
OUT_KEYFRAMES = f"{ROOT}/latent/keyframes"
OUT_CANDIDATES = f"{ROOT}/latent/candidates"
OUT_META = f"{ROOT}/datasets/keyframes_meta_gpt4o.json"

MODEL = "gpt-4o"
K_CANDIDATES = 100
N_SELECT = 4
TEMPERATURE = 0.0
CANDIDATE_MAX_WH = 512
CANDIDATE_JPEG_Q = 4

def ensure_dir(p: str):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out.strip(), err.strip()

def ffprobe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    ret, out, err = run_cmd(cmd)
    if ret != 0 or not out:
        raise RuntimeError(f"ffprobe failed for {video_path}: {err}")
    try:
        return float(out)
    except Exception as e:
        raise RuntimeError(f"parse duration failed for {video_path}: {out}") from e

def extract_frame_at(video_path: str, time_sec: float, out_path: str) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{time_sec:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-y", out_path
    ]
    ret, _, err = run_cmd(cmd)
    if ret != 0:
        raise RuntimeError(f"ffmpeg extract failed: {err}")

# Helper for extracting candidate frame (resize+JPEG)
def extract_candidate_frame(video_path: str, time_sec: float, out_path: str, max_wh: int = CANDIDATE_MAX_WH, q: int = CANDIDATE_JPEG_Q) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{time_sec:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-vf", f"scale='min({max_wh},iw)':'min({max_wh},ih)':force_original_aspect_ratio=decrease",
        "-q:v", str(q),
        "-y", out_path
    ]
    ret, _, err = run_cmd(cmd)
    if ret != 0:
        raise RuntimeError(f"ffmpeg candidate-frame extract failed: {err}")

def gen_uniform_times(duration: float, k: int) -> List[float]:
    if duration <= 0:
        return []
    start_pad = min(1.0, duration * 0.02)
    end_pad = min(1.0, duration * 0.02)
    usable = max(0.0, duration - start_pad - end_pad)
    if usable <= 0:
        return [duration / 2.0] * k
    times = [start_pad + (i + 1) * usable / (k + 1) for i in range(k)]
    return times

def image_to_data_url(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    suffix = pathlib.Path(img_path).suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"

def build_messages(question: str,
                   answer_choices: List[str],
                   candidate_infos: List[Tuple[int, float, str]],
                   answer: str) -> List[Dict]:
    header_lines = [
        "You are given a video represented by candidate frames.",
        "Your task: pick EXACTLY 4 frames that are most useful to answer the question,",
        "and provide, for each selected frame, ONE short key object label (a visible noun phrase, lowercase, 1-3 words).",
        "Return ONLY valid JSON with this schema:",
        '{"selected": [i0, i1, i2, i3], "labels": ["label_for_i0", "label_for_i1", "label_for_i2", "label_for_i3"]}',
        "Rules:",
        "- Indices must come from the provided candidate indices; they must be 4 unique integers.",
        "- labels must be aligned with the same order as `selected`.",
        "- Each label must be a concrete, visible object (noun phrase). No verbs or abstract words (e.g., 'playing', 'activity', 'background').",
        "- Use lowercase english; keep it short (<=3 words).",
        "- JSON only. No extra text.",
        "",
        f"Question: {question}",
    ]
    if answer_choices:
        header_lines.append("Choices:")
        for i, ch in enumerate(answer_choices):
            header_lines.append(f"- {chr(65+i)}. {ch}")
    # Include the correct answer to stabilize selection
    if answer:
        header_lines.append("")
        header_lines.append(f"Correct answer: {answer}")

    header_text = "\n".join(header_lines)

    content = [{"type": "text", "text": header_text}]

    for idx, t, img_path in candidate_infos:
        content.append({"type": "text", "text": f"frame_index: {idx}, t={t:.2f}s"})
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(img_path)}
        })

    messages = [{"role": "user", "content": content}]
    return messages

def call_gpt4o_select_indices(client: OpenAI,
                              messages: List[Dict]) -> Tuple[List[int], List[str]]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
        selected = data.get("selected", [])
        labels = data.get("labels", [])
        # ensure 4 unique indices and aligned labels
        selected = list(dict.fromkeys(int(i) for i in selected))[:N_SELECT]
        if len(selected) != N_SELECT or not isinstance(labels, list) or len(labels) < N_SELECT:
            return [], []
        labels = [str(l).strip().lower() for l in labels[:N_SELECT]]
        return selected, labels
    except Exception:
        return [], []

def main():
    ensure_dir(OUT_KEYFRAMES)
    ensure_dir(OUT_CANDIDATES)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Please `export OPENAI_API_KEY=...` first.")

    client = OpenAI(api_key=api_key)

    with open(SAMPLED_META, "r") as f:
        items = json.load(f)

    all_results = []

    for it in items:
        vid = it["video_id"]
        q = it.get("question", "")
        answer = it.get("answer")
        video_path = f"{VIDEO_DIR}/{vid}.mp4"
        if not os.path.exists(video_path):
            print(f"Missing video file: {video_path}, skipped")
            continue

        choices = []
        for i in range(5):
            key = f"answer_choice_{i}"
            if key in it and it[key]:
                choices.append(it[key])

        print(f"\n>>>Processing {vid} ...")
        try:
            duration = ffprobe_duration(video_path)
        except Exception as e:
            print(f"ffprobe failed: {e}")
            continue

        cand_times = gen_uniform_times(duration, K_CANDIDATES)

        cand_dir = pathlib.Path(OUT_CANDIDATES) / vid
        ensure_dir(str(cand_dir))
        cand_infos: List[Tuple[int, float, str]] = []
        for idx, t in enumerate(cand_times):
            out_jpg = str(cand_dir / f"cand_{idx:02d}.jpg")
            try:
                extract_candidate_frame(video_path, t, out_jpg)
                cand_infos.append((idx, t, out_jpg))
            except Exception as e:
                print(f"Candidate frame extraction failed idx={idx}, t={t:.2f}s: {e}")

        if len(cand_infos) < max(N_SELECT, 4):
            print(f"Too few candidate frames ({len(cand_infos)}), skipped {vid}")
            continue

        messages = build_messages(q, choices, cand_infos, answer=answer)
        selected_idx, selected_labels = call_gpt4o_select_indices(client, messages)

        if len(selected_idx) < N_SELECT or len(selected_labels) < N_SELECT:
            print(f"Model did not return enough indices or labels (got indices {selected_idx}, labels {selected_labels}), skipped {vid}")
            continue

        frame_out_paths = []
        keyframe_dir = pathlib.Path(OUT_KEYFRAMES) / vid
        ensure_dir(str(keyframe_dir))
        for i, idx in enumerate(selected_idx[:N_SELECT], start=1):
            if idx < 0 or idx >= len(cand_infos):
                continue
            t = cand_infos[idx][1]
            out_png = str(keyframe_dir / f"{vid}_frame_{i}.png")
            try:
                extract_frame_at(video_path, t, out_png)
                frame_out_paths.append(out_png)
            except Exception as e:
                print(f"Keyframe extraction failed idx={idx}, t={t:.2f}s: {e}")

        if len(frame_out_paths) < N_SELECT:
            print(f"Final keyframes less than 4, skipped {vid}")
            continue

        print(f"Selected indices: {selected_idx}")
        print(f"Frame labels: {selected_labels}")

        rec = dict(it)  # preserve all original fields
        rec["frames"] = frame_out_paths
        rec["frame_labels"] = selected_labels  # aligned with frames order
        all_results.append(rec)

        print(f"{vid} keyframes done -> {frame_out_paths}")

    with open(OUT_META, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDONE. Saved to {OUT_META}. Total: {len(all_results)}")

if __name__ == "__main__":
    main()
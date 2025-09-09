import os, io, json, base64, pathlib, subprocess, re, argparse
from typing import List, Tuple, Dict, Optional

from openai import OpenAI

# ================== 基础工具 ==================
def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

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

def extract_candidate_frame(video_path: str, time_sec: float, out_path: str,
                            max_wh: int, q: int) -> None:
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

def image_to_data_url(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    mime = "image/jpeg" if img_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return f"data:{mime};base64,{b64}"

def find_video_file(video_dir: str, video_id: str) -> Optional[str]:
    exts = [".mp4", ".mkv", ".webm", ".m4v", ".mov"]
    for ext in exts:
        p = pathlib.Path(video_dir) / f"{video_id}{ext}"
        if p.exists():
            return str(p)
    return None

# ================== 时间解析 ==================
def time_to_seconds(s: str) -> Optional[float]:
    """支持 hh:mm:ss 和 mm:ss"""
    try:
        parts = s.strip().split(":")
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + sec
        elif len(parts) == 2:
            m, sec = int(parts[0]), int(parts[1])
            return m * 60 + sec
    except Exception:
        return None
    return None

def extract_times_from_reasoning(reasoning: str) -> List[float]:
    """
    1) 捕获区间并加入两端点: "00:50 - 00:52", "between 1:06-1:07", "03:37 to 03:41"
    2) 修复奇怪写法: "01:20-01-26" -> 01:20 到 01:26（此规则易引入误报，默认关闭）
    3) 捕获单点: "mm:ss", "hh:mm:ss"
    """
    if not reasoning:
        return []

    s = reasoning
    s = s.replace("–", "-").replace("—", "-")  # 统一 dash
    times: List[float] = []

    # case A: hh:mm:ss - hh:mm:ss
    for a, b in re.findall(r"(\d{1,2}:\d{2}:\d{2})\s*(?:-|to)\s*(\d{1,2}:\d{2}:\d{2})", s, flags=re.IGNORECASE):
        ta, tb = time_to_seconds(a), time_to_seconds(b)
        if ta is not None: times.append(ta)
        if tb is not None: times.append(tb)

    # case B: mm:ss - mm:ss
    for a, b in re.findall(r"(\d{1,2}:\d{2})\s*(?:-|to)\s*(\d{1,2}:\d{2})", s, flags=re.IGNORECASE):
        ta, tb = time_to_seconds(a), time_to_seconds(b)
        if ta is not None: times.append(ta)
        if tb is not None: times.append(tb)

    # case D: 单独的 hh:mm:ss
    for t in re.findall(r"\b(\d{1,2}:\d{2}:\d{2})\b", s):
        ts = time_to_seconds(t)
        if ts is not None:
            times.append(ts)

    # case E: 单独的 mm:ss
    for t in re.findall(r"\b(\d{1,2}:\d{2})\b", s):
        ts = time_to_seconds(t)
        if ts is not None:
            times.append(ts)

    # 去重并排序
    uniq = sorted(set(times))
    return uniq

# ================== LLM 选择补帧 ==================
def build_messages_for_fill(question: str,
                            answer_choices: List[str],
                            answer: str,
                            candidate_infos: List[Tuple[int, float, str]],
                            need_n: int) -> List[Dict]:
    lines = [
        "You are given a video represented by candidate frames.",
        f"Task: pick EXACTLY {need_n} frames from the candidates below that help answer the question.",
        'Return ONLY JSON with this schema: {"selected":[i0,i1,...]}',
        "Rules:",
        "- Indices must come from the provided candidate list; they must be unique integers.",
        "- JSON only. No extra text.",
        "",
        f"Question: {question}",
    ]
    if answer_choices:
        lines.append("Choices:")
        for i, ch in enumerate(answer_choices):
            lines.append(f"- {chr(65+i)}. {ch}")
    if answer:
        lines.append("")
        lines.append(f"Correct answer: {answer}")

    content = [{"type": "text", "text": "\n".join(lines)}]
    for idx, t, path in candidate_infos:
        content.append({"type": "text", "text": f"frame_index: {idx}, t={t:.2f}s"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})

    return [{"role": "user", "content": content}]

def call_gpt4o_select_indices(client: OpenAI, model: str, temperature: float, messages: List[Dict], need_n: int) -> List[int]:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content
    try:
        data = json.loads(text)
        selected = data.get("selected", [])
        sel = list(dict.fromkeys(int(i) for i in selected))[:need_n]
        if len(sel) != need_n:
            return []
        return sel
    except Exception:
        return []

# ================== CLI 参数 ==================
def parse_args():
    ap = argparse.ArgumentParser(description="Extract keyframes from videos using reasoning timestamps, optionally LLM-fill from candidates.")
    # 路径
    ap.add_argument("--in-meta", required=True, help="输入元数据 JSON")
    ap.add_argument("--video-dirs", required=True, help="视频目录（单个 dir）")
    ap.add_argument("--out-keyframes-dir", required=True, help="关键帧输出目录")
    ap.add_argument("--out-candidates-dir", required=True, help="候选帧输出目录")
    ap.add_argument("--output", required=True, help="带 frames 字段的输出 JSON")
    # 字段映射
    ap.add_argument("--field.video_id", dest="f_video_id", required=True, help="字段名：视频ID")
    ap.add_argument("--field.question_id", dest="f_question_id", required=True, help="字段名：问题ID（用于子目录命名）")
    ap.add_argument("--field.question", dest="f_question", required=True, help="字段名：问题文本")
    ap.add_argument("--field.answer", dest="f_answer", required=True, help="字段名：标准答案")
    ap.add_argument("--field.reasoning", dest="f_reasoning", required=True, help="字段名：推理/解析文本")
    ap.add_argument("--field.choices-prefix", dest="f_choices_prefix", default="answer_choice_",
                    help="选项字段前缀（默认：answer_choice_，聚合 answer_choice_0..4）")
    # 策略参数
    ap.add_argument("--min-select", type=int, default=4, help="最少关键帧数量，不足则用 LLM 补齐（默认 4）")
    ap.add_argument("--k-candidates", type=int, default=100, help="候选帧采样数量 K（默认 100）")
    ap.add_argument("--candidate-max-wh", type=int, default=512, help="候选缩略图最大片边（默认 512）")
    ap.add_argument("--candidate-jpeg-q", type=int, default=4, help="候选缩略图 JPEG 质量（默认 4）")
    ap.add_argument("--close-sec", type=float, default=0.5, help="候选与 GT 的最小间隔秒（默认 0.5）")
    ap.add_argument("--model", default="gpt-4o", help="用于补帧选择的 LLM 模型（默认 gpt-4o）")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM 采样温度（默认 0.0）")
    return ap.parse_args()

# ================== 主逻辑 ==================
def main():
    args = parse_args()

    ensure_dir(args.out_keyframes_dir)
    ensure_dir(args.out_candidates_dir)

    # 只有在需要补帧时才需要 OPENAI_API_KEY
    api_key = os.environ.get("OPENAI_API_KEY", "")
    client = OpenAI(api_key=api_key) if api_key else None

    # 读入 meta
    meta_path = pathlib.Path(args.in_meta)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta: {args.in_meta}")
    items = json.load(open(meta_path, "r"))

    all_results = []
    need_llm_total = 0
    gt_total_saved = 0

    for it in items:
        vid = str(it.get(args.f_video_id))
        qkey = str(it.get(args.f_question_id, vid))
        q = it.get(args.f_question, "")
        answer = it.get(args.f_answer)
        reasoning = it.get(args.f_reasoning, "")

        video_path = find_video_file(args.video_dirs, vid)
        if not video_path:
            print(f"[skip] video not found for {vid}")
            continue

        print(f"\n>>> {vid}")
        try:
            duration = ffprobe_duration(video_path)
        except Exception as e:
            print(f"[ffprobe error] {e}")
            continue

        # 1) 解析 reasoning 时间点
        times = extract_times_from_reasoning(reasoning)
        # clamp 到视频长度内
        clamped: List[float] = []
        for t in times:
            if t < 0:
                t = 0.0
            if t > duration:
                t = max(0.0, duration - 0.001)
            clamped.append(t)
        # 去重并排序
        gt_times = sorted(set(round(t, 3) for t in clamped))
        print(f"  reasoning times found: {len(gt_times)} -> {gt_times[:10]}{' ...' if len(gt_times)>10 else ''}")

        keyframe_dir = pathlib.Path(args.out_keyframes_dir) / qkey
        ensure_dir(str(keyframe_dir))

        # 2) 先把 reasoning 的时间点全部截帧保存（如果 >=min_select，就全存完）
        frame_out_paths: List[str] = []
        for i, t in enumerate(gt_times, start=1):
            out_png = str(keyframe_dir / f"keyframe_{i}.png")
            try:
                extract_frame_at(video_path, t, out_png)
                frame_out_paths.append(out_png)
            except Exception as e:
                print(f"  [gt extract fail] t={t:.2f}s: {e}")

        gt_total_saved += len(frame_out_paths)

        # 3) 如果少于 min_select，用 LLM 从候选帧里补到 min_select
        if len(frame_out_paths) < args.min_select:
            need = args.min_select - len(frame_out_paths)
            if not client:
                print("  [warn] OPENAI_API_KEY not set; cannot fill by LLM. Skipping video.")
                continue

            # 生成候选帧
            # 均匀采样 K_CANDIDATES，过滤掉与已有 GT 时间点太接近的候选（close-sec 内）
            def gen_uniform_times(duration: float, k: int) -> List[float]:
                if duration <= 0: return []
                pad = min(1.0, duration * 0.02)
                usable = max(0.0, duration - 2 * pad)
                if usable <= 0: return [duration / 2.0] * k
                return [pad + (i + 1) * usable / (k + 1) for i in range(k)]

            cand_times = gen_uniform_times(duration, args.k_candidates)
            used = set(gt_times)

            def is_close(t: float) -> bool:
                return any(abs(t - u) <= args.close_sec for u in used)

            cand_dir = pathlib.Path(args.out_candidates_dir) / qkey
            ensure_dir(str(cand_dir))
            cand_infos: List[Tuple[int, float, str]] = []
            for idx, t in enumerate(cand_times):
                if is_close(t):
                    continue
                out_jpg = str(cand_dir / f"cand_{idx:02d}.jpg")
                try:
                    extract_candidate_frame(video_path, t, out_jpg, max_wh=args.candidate_max_wh, q=args.candidate_jpeg_q)
                    cand_infos.append((len(cand_infos), t, out_jpg))  # 重新编号连续
                except Exception as e:
                    print(f"  [cand fail] t={t:.2f}s: {e}")

            if len(cand_infos) < need:
                print(f"  [skip] too few candidate frames ({len(cand_infos)}) to fill {need}.")
                continue

            # 收集答案选项
            choices = []
            for i in range(5):
                k = f"{args.f_choices_prefix}{i}"
                if k in it and it[k]:
                    choices.append(it[k])

            messages = build_messages_for_fill(
                question=q,
                answer_choices=choices,
                answer=answer,
                candidate_infos=cand_infos,
                need_n=need,
            )
            sel = call_gpt4o_select_indices(client, args.model, args.temperature, messages, need)
            if len(sel) != need:
                print(f"  [skip] LLM did not return enough indices (need {need}, got {sel}).")
                continue

            # 按顺序补帧，编号接在已有帧后面
            for j, idx in enumerate(sel, start=len(frame_out_paths) + 1):
                if idx < 0 or idx >= len(cand_infos):
                    continue
                t = cand_infos[idx][1]
                out_png = str(keyframe_dir / f"keyframe_{j}.png")
                try:
                    extract_frame_at(video_path, t, out_png)
                    frame_out_paths.append(out_png)
                except Exception as e:
                    print(f"  [fill extract fail] idx={idx}, t={t:.2f}s: {e}")

            need_llm_total += 1
            print(f"  filled {need} with LLM, total frames={len(frame_out_paths)}")

        # 最终至少要有 1 张（如果 reasoning 全失败又没补帧，就跳过）
        if len(frame_out_paths) == 0:
            print("  [skip] no frames extracted.")
            continue

        # 写入记录（保留原字段，只新增 frames）
        rec = dict(it)
        rec["frames"] = frame_out_paths
        all_results.append(rec)
        print(f"  DONE {vid}: saved {len(frame_out_paths)} frames")

    ensure_dir(pathlib.Path(args.output).parent.as_posix())
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nALL DONE. Saved to {args.output}.")
    print(f"Videos processed: {len(all_results)} | used LLM fill for: {need_llm_total} | GT frames saved: {gt_total_saved}")

if __name__ == "__main__":
    main()
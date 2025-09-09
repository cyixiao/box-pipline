import os, json, pathlib, base64, argparse
from typing import List, Dict, Any, Optional
from openai import OpenAI

def parse_args():
    ap = argparse.ArgumentParser(description="Generate short frame-level labels for keyframes using an LLM.")
    ap.add_argument("--in-meta", required=True, help="输入元数据 JSON（包含 frames 等字段）")
    ap.add_argument("--output", required=True, help="输出 JSON（写入 frame_labels 字段）")
    ap.add_argument("--field.frames", dest="f_frames", required=True, help="字段名：frames 列表")
    ap.add_argument("--field.question", dest="f_question", required=True, help="字段名：问题文本")
    ap.add_argument("--field.answer", dest="f_answer", required=True, help="字段名：正确答案")
    ap.add_argument("--field.reasoning", dest="f_reasoning", default="", help="字段名：推理/解析；可留空表示该数据集没有该字段")
    ap.add_argument("--field.choices-prefix", dest="f_choices_prefix", default="answer_choice_", help="选项字段前缀（默认：answer_choice_）")
    ap.add_argument("--model", default="gpt-4o", help="LLM 模型（默认：gpt-4o）")
    ap.add_argument("--temperature", type=float, default=0.0, help="LLM 温度（默认：0.0）")
    return ap.parse_args()

# ================== 工具函数 ==================
def ensure_parent(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

def image_to_data_url(img_path: str) -> str:
    with open(img_path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    ext = pathlib.Path(img_path).suffix.lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
    return f"data:{mime};base64,{b64}"

def build_label_prompt(question: str,
                       answer: Optional[str],
                       reasoning: Optional[str],
                       answer_choices: List[str]) -> Dict[str, Any]:

    lines = [
        "You are given ONE video keyframe image plus a QA context.",
        "Goal: output ONE short, canonical label (1–3 lowercase English nouns/noun-phrases) naming the SINGLE most decisive, visible object/part in THIS frame that helps answer the question.",
        "Rules:",
        "- The label must be a concrete, visible object/part in THIS image.",
        "- Lowercase only, ASCII only, no quotes or punctuation, <= 3 words.",
        "- Avoid verbs, attributes, abstract words, or long phrases with prepositions/possessives.",
        "",
        f"Question: {question.strip() if question else ''}",
    ]
    if answer_choices:
        lines.append("Choices:")
        for i, ch in enumerate(answer_choices):
            lines.append(f"- {chr(65+i)}. {ch}")
    if answer:
        lines.append(f"Correct answer: {answer}")
    if reasoning:
        lines.append("Reasoning (for context):")
        lines.append(reasoning.strip())

    lines += [
        "",
        'Return ONLY JSON with this schema: {"label":"<short noun phrase>"}',
    ]
    return {"type": "text", "text": "\n".join(lines)}

def call_gpt4o_label(client: OpenAI, model: str, temperature: float,
                     img_path: str,
                     question: str,
                     answer: Optional[str],
                     reasoning: Optional[str],
                     answer_choices: List[str]) -> Optional[str]:
    content = [
        build_label_prompt(question, answer, reasoning, answer_choices),
        {"type": "image_url", "image_url": {"url": image_to_data_url(img_path)}}
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": content}],
        )
        txt = resp.choices[0].message.content
        data = json.loads(txt)
        label = data.get("label", "")
        if isinstance(label, str) and label.strip():
            lbl = label.strip().lower()
            # 简单清洗：只保留较短标签
            if len(lbl.split()) <= 4 and len(lbl) <= 40:
                return lbl
    except Exception as e:
        # 可按需打印 e 进行调试
        pass
    return None

# ================== 主流程 ==================
def main():
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. `export OPENAI_API_KEY=...`")
    client = OpenAI(api_key=api_key)

    if not pathlib.Path(args.in_meta).exists():
        raise FileNotFoundError(f"missing: {args.in_meta}")
    items: List[Dict[str, Any]] = json.load(open(args.in_meta, "r"))

    updated = []
    for rec in items:
        vid = rec.get("video_id")
        q = rec.get(args.f_question, "")
        ans = rec.get(args.f_answer)
        if args.f_reasoning:
            reasoning = rec.get(args.f_reasoning, "")
        else:
            reasoning = ""
        frames: List[str] = rec.get(args.f_frames, []) or []

        # 选项
        choices: List[str] = []
        for i in range(5):
            k = f"{args.f_choices_prefix}{i}"
            if k in rec and rec[k]:
                choices.append(str(rec[k]))

        if not frames:
            updated.append(rec)
            print(f"[skip] {vid} has no frames")
            continue

        labels: List[str] = []
        print(f"\n>>> {vid}: {len(frames)} frames")
        for idx, fp in enumerate(frames, 1):
            if not pathlib.Path(fp).exists():
                labels.append("scene")  # 兜底
                print(f"  frame_{idx}: missing -> label=scene")
                continue
            label = call_gpt4o_label(client, args.model, args.temperature, fp, q, ans, reasoning, choices)
            if not label:
                label = "scene"  # 兜底
            labels.append(label)
            print(f"  frame_{idx}: label={label}")

        # 写回
        out_rec = dict(rec)
        out_rec["frame_labels"] = labels  # 与 frames 顺序一一对应
        updated.append(out_rec)

    ensure_parent(args.output)
    with open(args.output, "w") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    print(f"\nDONE. Saved to {args.output}. {len(updated)} samples updated.")

if __name__ == "__main__":
    main()
import argparse, subprocess, sys, pathlib
from typing import Dict, Any, List
import yaml

THIS_DIR = pathlib.Path(__file__).resolve().parent

SCRIPTS = {
    "download": THIS_DIR / "download_videos.py",
    "keyframes": THIS_DIR / "keyframes_gpt.py",
    "labels": THIS_DIR / "add_labels.py",
    "bbox": THIS_DIR / "bbox_sam.py",
    "latents": THIS_DIR / "make_latents.py",
}

STAGE_ORDER = ["download", "keyframes", "labels", "bbox", "latents"]


def load_cfg(path: str) -> Dict[str, Any]:
    p = pathlib.Path(path)
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def derive_paths_from_root(root: pathlib.Path) -> Dict[str, str]:
    """
    Derive all file/dir paths from a single dataset root.

    Layout (example: root=/.../minerva, name='minerva'):
      <root>/
        name.json
        name_keyframes.json
        name_label.json
        name_bbox.json
        frames/
          candidates/
          keyframes/
        latent/
          bbox/
          enlarge/
        videos/
    """
    root = root.resolve()
    name = root.name

    paths = {
        "root": str(root),
        "name": name,
        # files
        "in_meta": str(root / f"{name}.json"),
        "keyframes_json": str(root / f"{name}_keyframes.json"),
        "labels_json": str(root / f"{name}_label.json"),
        "bbox_json": str(root / f"{name}_bbox.json"),
        "latents_json": str(root / f"{name}_latent.json"),
        # dirs
        "frames_keyframes": str(root / "frames" / "keyframes"),
        "frames_candidates": str(root / "frames" / "candidates"),
        "latent_enlarge": str(root / "latent" / "enlarge"),
        "latent_bbox": str(root / "latent" / "bbox"),
        "videos_dir": str(root / "videos"),
    }
    return paths


def build_cmd(stage: str, cfg: Dict[str, Any], paths: Dict[str, str]) -> List[str]:
    fields = cfg.get("fields", {})
    s = cfg.get(stage, {})  # only for params/overrides
    params = s.get("params", {})

    if stage == "download":
        cmd = [sys.executable, str(SCRIPTS[stage]),
               "--in-meta", paths["in_meta"],
               "--out-dir", paths["videos_dir"],
               "--field.video_id", str(fields["video_id"]) ]
        url = s.get("overrides", {}).get("url")
        if url:
            cmd += ["--field.url", str(url)]
        return cmd

    if stage == "keyframes":
        cmd = [sys.executable, str(SCRIPTS[stage]),
               "--in-meta", paths["in_meta"],
               "--video-dirs", paths["videos_dir"],
               "--out-keyframes-dir", paths["frames_keyframes"],
               "--out-candidates-dir", paths["frames_candidates"],
               "--output", paths["keyframes_json"],
               "--field.video_id", str(fields["video_id"]),
               "--field.question_id", str(fields["question_id"]),
               "--field.question", str(fields["question"]),
               "--field.answer", str(fields["answer"]),
               "--field.reasoning", str(fields["reasoning"]),
               "--field.choices-prefix", str(fields["choices_prefix"]),
               "--min-select", str(params.get("min_select", 4)),
               "--k-candidates", str(params.get("k_candidates", 100)),
               "--candidate-max-wh", str(params.get("candidate_max_wh", 512)),
               "--candidate-jpeg-q", str(params.get("candidate_jpeg_q", 4)),
               "--close-sec", str(params.get("close_sec", 0.5)),
               "--model", str(params.get("model", "gpt-4o")),
               "--temperature", str(params.get("temperature", 0.0)),
               ]
        return cmd

    if stage == "labels":
        cmd = [sys.executable, str(SCRIPTS[stage]),
               "--in-meta", paths["keyframes_json"],
               "--output", paths["labels_json"],
               "--field.frames", str(fields["frames"]),
               "--field.question", str(fields["question"]),
               "--field.answer", str(fields["answer"]),
               "--field.reasoning", str(fields["reasoning"]),
               "--field.choices-prefix", str(fields["choices_prefix"]),
               "--model", str(params.get("model", "gpt-4o")),
               "--temperature", str(params.get("temperature", 0.0)),
               ]
        return cmd

    if stage == "bbox":
        cmd = [sys.executable, str(SCRIPTS[stage]),
               "--in-meta", paths["labels_json"],
               "--output", paths["bbox_json"],
               "--field.frames", str(fields["frames"]),
               "--field.frame_labels", str(fields["frame_labels"]),
               "--field.bboxes-out", str(fields["bboxes"]),
               "--model-id", str(params.get("model_id", "IDEA-Research/grounding-dino-base")),
               "--box-threshold", str(params.get("box_threshold", 0.30)),
               "--text-threshold", str(params.get("text_threshold", 0.25)),
               ]
        return cmd

    if stage == "latents":
        cmd = [sys.executable, str(SCRIPTS[stage]),
               "--in-keyframes", paths["bbox_json"],
               "--in-meta", paths["in_meta"],
               "--output", paths["latents_json"],
               "--out-enlarge-dir", paths["latent_enlarge"],
               "--out-bbox-dir", paths["latent_bbox"],
               "--field.video_id", str(fields["video_id"]),
               "--field.question_id", str(fields["question_id"]),
               "--field.frames", str(fields["frames"]),
               "--field.bboxes", str(fields["bboxes"]),
               "--field.enlarge-out", str(fields["enlarge_out"]),
               "--field.bbox-out", str(fields["bbox_out"]),
               "--rel-root-enlarge", str(params.get("rel_root_enlarge", "/latent/enlarge")),
               "--rel-root-bbox", str(params.get("rel_root_bbox", "/latent/bbox")),
               "--filename-template", str(params.get("filename_template", "keyframe_{i}.png")),
               "--draw-width", str(params.get("draw_width", 3)),
               "--padding-pix", str(params.get("padding_pix", 0)),
               "--resize-short", str(params.get("resize_short", 0)),
               ]
        return cmd

    raise ValueError(f"Unknown stage: {stage}")


def run(cmd: List[str], dry_run: bool) -> int:
    print("\n$", " ".join(map(str, cmd)))
    if dry_run:
        return 0
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Run video pipeline from a single dataset root.")
    ap.add_argument("--config", required=True, help="YAML config containing 'fields' and stage 'params'")
    ap.add_argument("--root", required=True, help="Dataset root directory (e.g., /.../minerva)")
    ap.add_argument("--only", choices=STAGE_ORDER)
    ap.add_argument("--from", dest="from_stage", choices=STAGE_ORDER)
    ap.add_argument("--to", dest="to_stage", choices=STAGE_ORDER)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    paths = derive_paths_from_root(pathlib.Path(args.root))

    # Show resolved layout
    print("[root-mode] using paths (dataset name = %s):" % paths["name"])
    for k in ["in_meta", "keyframes_json", "labels_json", "bbox_json", "latents_json",
              "frames_keyframes", "frames_candidates", "latent_enlarge", "latent_bbox", "videos_dir"]:
        print(f"  - {k}: {paths[k]}")

    if args.only:
        stages = [args.only]
    else:
        stages = STAGE_ORDER[:]
        if args.from_stage:
            stages = stages[stages.index(args.from_stage):]
        if args.to_stage:
            stages = stages[:stages.index(args.to_stage) + 1]

    for st in stages:
        script = SCRIPTS.get(st)
        if not script or not script.exists():
            print(f"[error] script for stage '{st}' not found at {script}", file=sys.stderr)
            sys.exit(2)
        cmd = build_cmd(st, cfg, paths)
        code = run(cmd, args.dry_run)
        if code != 0:
            print(f"[error] stage '{st}' failed with exit code {code}", file=sys.stderr)
            sys.exit(code)

    print("\nAll stages finished successfully.")


if __name__ == "__main__":
    main()

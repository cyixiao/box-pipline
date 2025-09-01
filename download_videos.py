import json, subprocess, pathlib, random

NEPTUNE_JSON = "/home/cyixiao/Project/videollm/pipline/datasets/neptune_full.json"
OUT_DIR = "/home/cyixiao/Project/videollm/pipline/videos/raw"
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

with open(NEPTUNE_JSON, "r") as f:
    data = json.load(f)

filtered = [item for item in data if item["key"].endswith("/1")]

unique = {}
for item in filtered:
    vid = item["video_id"]
    if vid not in unique:
        unique[vid] = item

video_ids = list(unique.keys())

sample_ids = random.sample(video_ids, 100)

sampled_meta = [unique[vid] for vid in sample_ids]

with open("/home/cyixiao/Project/videollm/pipline/sampled_meta.json", "w") as f:
    json.dump(sampled_meta, f, indent=2)

for vid in sample_ids:
    url = f"https://www.youtube.com/watch?v={vid}"
    out_tpl = f"{OUT_DIR}/{vid}.%(ext)s"
    cmd = [
        "yt-dlp",
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        "-o", out_tpl,
        "--ignore-errors",
        "--retries", "10",
        "--fragment-retries", "10",
        url,
    ]
    print("Downloading:", url)
    subprocess.run(cmd, check=False)
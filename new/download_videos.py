import json, subprocess, pathlib, os, argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-meta", required=True, help="输入元数据 JSON")
    ap.add_argument("--out-dir", required=True, help="视频输出目录")
    ap.add_argument("--field.video_id", dest="field_video_id", required=True, help="字段名：视频ID")
    ap.add_argument("--field.url", dest="field_url", help="（可选）字段名：视频URL；若为空则按 YouTube 规则拼接")
    return ap.parse_args()

args = parse_args()

NEPTUNE_JSON = args.in_meta
OUT_DIR = args.out_dir
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

with open(NEPTUNE_JSON, "r") as f:
    data = json.load(f)

filtered = data

unique = {}
for item in filtered:
    vid = item[args.field_video_id]
    if vid not in unique:
        unique[vid] = item

video_ids = list(unique.keys())
print("Total unique video_ids:", len(video_ids))

if len(video_ids) == 0:
    raise SystemExit("No video_ids found after filtering. Check the input JSON format.")

succeeded, failed = set(), set()

cookies_file = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
cookies_from_browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "").strip()
if cookies_file and not pathlib.Path(cookies_file).exists():
    print(f"[warn] YTDLP_COOKIES_FILE set but file not found: {cookies_file}. Ignoring.")
    cookies_file = ""
if cookies_from_browser:
    print(f"[info] Using --cookies-from-browser {cookies_from_browser}")
elif cookies_file:
    print(f"[info] Using --cookies {cookies_file}")
else:
    print("[info] No cookies provided. If you hit 'Sign in to confirm you’re not a bot', set YTDLP_COOKIES_FILE or YTDLP_COOKIES_FROM_BROWSER.")

extractor_args = os.environ.get("YTDLP_EXTRACTOR_ARGS", "youtube:player_client=web")
user_agent = os.environ.get(
    "YTDLP_UA",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
)
print(f"[info] Using extractor args: {extractor_args}")
print(f"[info] Using UA: {user_agent[:40]}...")

for vid in video_ids:
    if args.field_url and args.field_url in unique[vid]:
        url = unique[vid][args.field_url]
    else:
        url = f"https://www.youtube.com/watch?v={vid}"
    out_tpl = f"{OUT_DIR}/{vid}.%(ext)s"
    # Skip if final output already exists
    final_path = pathlib.Path(OUT_DIR) / f"{vid}.mp4"
    if final_path.exists() and final_path.stat().st_size > 0:
        print(f"[skip] {final_path} already exists. Skipping download.")
        succeeded.add(vid)
        continue
    cmd = [
        "yt-dlp",
        "-f", "bv*+ba/b",
        "--merge-output-format", "mp4",
        "-o", out_tpl,
        "--ignore-errors",
        "--retries", "10",
        "--fragment-retries", "10",
        "--sleep-requests", "1",
        "--min-sleep-interval", "1",
        "--max-sleep-interval", "3",
        "--extractor-args", extractor_args,
        "--user-agent", user_agent,
    ]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    elif cookies_file:
        cmd += ["--cookies", cookies_file]
    cmd.append(url)
    print("Downloading:", url)
    subprocess.run(cmd, check=False)
    if final_path.exists() and final_path.stat().st_size > 0:
        print(f"[ok] {vid} -> {final_path}")
        succeeded.add(vid)
    else:
        print(f"[fail] {vid} download failed; will purge its items from {NEPTUNE_JSON}")
        failed.add(vid)

# Purge failed video items from the ORIGINAL input JSON and print summary
if failed:
    before = len(data)
    failed_str = {str(v) for v in failed}
    filtered_out = [item for item in data if str(item.get(args.field_video_id)) not in failed_str]
    pathlib.Path(NEPTUNE_JSON).write_text(json.dumps(filtered_out, indent=2, ensure_ascii=False))
    print(f"[purge] Removed {before - len(filtered_out)} items for {len(failed)} failed video(s). New dataset size: {len(filtered_out)}")
else:
    print("[purge] No failed videos; input JSON left unchanged.")
print(f"[summary] Success videos: {len(succeeded)}; Failed videos: {len(failed)}")

import json, subprocess, pathlib, os

NEPTUNE_JSON = "/home/cyixiao/Project/videollm/pipline/datasets/train/minerva.json"
OUT_DIR = "/home/cyixiao/Project/videollm/pipline/videos/minerva"
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

with open(NEPTUNE_JSON, "r") as f:
    data = json.load(f)

filtered = data

unique = {}
for item in filtered:
    vid = item["video_id"]
    if vid not in unique:
        unique[vid] = item

video_ids = list(unique.keys())
print("Total unique video_ids:", len(video_ids))

if len(video_ids) == 0:
    raise SystemExit("No video_ids found after filtering. Check the input JSON format.")

# Use all unique video_ids without sampling
sampled_meta = list(unique.values())

with open("/home/cyixiao/Project/videollm/pipline/sampled_meta.json", "w") as f:
    json.dump(sampled_meta, f, indent=2)

# Optional cookies: set one of these env vars on the server
#   YTDLP_COOKIES_FILE=/path/to/cookies.txt   (exported from your desktop browser)
#   YTDLP_COOKIES_FROM_BROWSER=chrome|firefox|edge|brave (only works if that browser profile exists on THIS machine)
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

# Force web client and set a desktop User-Agent to reduce bot checks.
# You can override these via env vars YTDLP_EXTRACTOR_ARGS and YTDLP_UA.
extractor_args = os.environ.get("YTDLP_EXTRACTOR_ARGS", "youtube:player_client=web")
user_agent = os.environ.get(
    "YTDLP_UA",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
)
print(f"[info] Using extractor args: {extractor_args}")
print(f"[info] Using UA: {user_agent[:40]}...")

for vid in video_ids:
    url = f"https://www.youtube.com/watch?v={vid}"
    out_tpl = f"{OUT_DIR}/{vid}.%(ext)s"
    # Skip if final output already exists (merged to mp4)
    final_path = pathlib.Path(OUT_DIR) / f"{vid}.mp4"
    if final_path.exists() and final_path.stat().st_size > 0:
        print(f"[skip] {final_path} already exists. Skipping download.")
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
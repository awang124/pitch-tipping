import json
import os
import subprocess
import re

JSON_DIR = "pitchers_json"
BASE_OUTPUT = "pitcher_clips"

os.makedirs(BASE_OUTPUT, exist_ok=True)

def safe(text):
    if not text:
        return "Unknown"
    return re.sub(r'[^A-Za-z0-9_.-]', '_', text)

for json_file in os.listdir(JSON_DIR):
    if not json_file.endswith(".json"):
        continue

    with open(os.path.join(JSON_DIR, json_file)) as f:
        data = json.load(f)

    plays = data["data"]["search"]["plays"]

    for play in plays:
        for media in play.get("mediaPlayback", []):

            info = media.get("playInfo", {})
            players = info.get("players", {})

            pitcher_name = safe(players.get("pitcher", {}).get("name", "Unknown_Pitcher"))
            batter_name = safe(players.get("batter", {}).get("name", "Unknown_Batter"))
            pitch_type = safe(info.get("pitchType", "UnknownPitch"))
            pitch_speed = safe(info.get("pitchSpeed", "0mph"))
            date = safe(media.get("date", "UnknownDate"))

            pitch_folder = os.path.join(BASE_OUTPUT, pitcher_name, pitch_type)
            os.makedirs(pitch_folder, exist_ok=True)

            filename = f"{pitch_type}_{pitch_speed}_{batter_name}_{date}.mp4"
            output_path = os.path.join(pitch_folder, filename)

            if os.path.exists(output_path):
                continue

            for feed in media.get("feeds", []):
                for pb in feed.get("playbacks", []):
                    url = pb.get("url", "")

                    if "fastball-clips.mlb.com" not in url:
                        continue

                    print("Downloading:", filename)

                    # 🔥 Use yt-dlp for ALL formats (m3u8 or mp4)
                    subprocess.run([
                        "yt-dlp",
                        "-f", "best",
                        "--no-part",
                        "--add-header", "Referer: https://www.mlb.com/",
                        "-o", output_path,
                        url
                    ])

                    break
                else:
                    continue
                break

print("All downloads complete.")

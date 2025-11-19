from ultralytics import YOLO
import csv, os
import json

model = YOLO('yolov8n.pt')

# I used .track() so the model can follow the same objects across the whole video instead of treating each frame separately. 
# This makes the results more stable and easier to understand for a continuous scene.
# I used ByteTrack because it’s the default, reliable tracker for YOLO videos and works well without extra setup. 
# It keeps object labels steady across frames in a consistent way, which is all I need for a short simple scene like this.
results = model.track(source="desk_scene.mov", show=True, save=True, tracker="bytetrack.yaml")

# I wrote the loop this way because iterating through frames and then each box keeps the
# data aligned with how YOLO outputs predictions. Pulling class, confidence, and box
# coordinates directly from each detection makes it straightforward to store everything cleanly.
with open("detections.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame_number", "object_class", "confidence_score", "x1", "y1", "x2", "y2"])

    for frame_id, r in enumerate(results):
        boxes = r.boxes 

        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            writer.writerow([frame_id, cls, conf] + xyxy)

# I formatted the JSON-to-CSV conversion this way because each item already contains a timestamp and text, 
# so looping through the list keeps the structure simple and clean.
# Writing only these two fields makes the transcript easy to read and consistent with how the data 
# is organized in the original file.
with open("infantlabtask.json", "r") as f:
    data = json.load(f)

segments = data["transcription"]["segments"]

with open("transcript.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["start", "end", "text"])

    for seg in segments:
        writer.writerow([seg["start"], seg["end"], seg["text"]])


W, H = 720, 1280

os.makedirs("cvat_yolo", exist_ok=True)

with open("detections.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame = int(row["frame_number"])
        x1, y1, x2, y2 = map(float, [row["x1"], row["y1"], row["x2"], row["y2"]])

        x_c = ((x1 + x2) / 2) / W
        y_c = ((y1 + y2) / 2) / H
        w   = (x2 - x1) / W
        h   = (y2 - y1) / H

        with open(f"cvat_yolo/{frame:06}.txt", "a") as out:
            out.write(f'{row["object_class"]} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f} {row["confidence_score"]}\n')

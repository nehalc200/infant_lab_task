This repository contains my implementation of the take-home task for the UCSD Infant Wearable Lab application. It includes:
- YOLOv8 tracking and annotated video
- Whisper transcript with timestamps
- Detection and transcript CSV exports
- Example corrected annotations using CVAT (extra credit)
- Full runnable code and outputs

Setup (Tested on Python 3.10 Conda + macOS):
- Run this script
  - git clone https://github.com/nehalc200/infant_lab_task.git
  - cd infant_lab_task
  - pip install ultralytics opencv-python pandas
  - python yolov8infantlab.py
- This will:
  - Run YOLOv8 with ByteTrack on desk_scene.mov
  - Export detections to detections.csv
  - Save the annotated video in runs/detect/
  - Convert Whisper JSON (infantlabtask.json) into transcript.csv

Files:

yolov8infantlab.py      – Main script (YOLO tracking + transcript conversion)

desk_scene.mov          – Input 1-minute desk scene video

infantlabtask.m4a       – Audio narration clip

infantlabtask.json      – Whisper transcription export

transcript.csv          – Timestamped transcript (start, end, text)

detections.csv          – YOLOv8 detection results (frame_number, class, confidence, bounding box)

annotation1.png         – CVAT correction screenshot (misclassified object fixed)

annotation2.png         – Another sample corrected frame

annotation3.png         – Additional corrected annotation example

runs/                   – YOLO-generated annotated video output folder

yolov8n.pt              – YOLOv8 model weights (included for reproducibility)


Choices:
- Used model.track() with ByteTrack to maintain consistent object identity across frames, making predictions easier to interpret for continuous video.
- Performed manual corrections in CVAT rather than re-import detections, as this was the most efficient way to address key errors without overengineering the pipeline.
- Kept CSV formatting minimal and structured to support direct future use in analysis or model refinement.

Thank you for reviewing my submission! Please feel free to contact me with any questions.

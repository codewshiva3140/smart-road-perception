# Lane, Vehicle and Drivable Area Detection for Smart Roads (Minor Project)

This project takes a road video recorded from inside a vehicle and produces an annotated video that shows three things in every frame:

- **Vehicles** (cars, buses, trucks, motorcycles, bicycles) with boxes and labels
- **Drivable area** (the road surface), tinted green
- **Lane / road boundary lines**, drawn in yellow

No model is trained in this project. It combines two pretrained deep learning models with a classical computer vision method.

## Methods used

| Task | Method | Trained on |
|---|---|---|
| Vehicle detection | YOLOv8m (Ultralytics) | COCO dataset |
| Drivable area | SegFormer-B2 semantic segmentation, "road" class | Cityscapes dataset |
| Lane lines | OpenCV: top-hat filter, Canny edges, Hough transform, line fitting, smoothing | No training needed |

## Requirements

- Python 3.9 or newer
- An NVIDIA GPU is recommended (it also runs on CPU, but slowly)
- Libraries: see `requirements.txt`

## How to run

### Option 1: Google Colab (the setup used for the results below)

1. Open a new notebook at colab.research.google.com.
2. Choose **Runtime > Change runtime type > T4 GPU**.
3. Upload `pipeline.py` and your video (named `road.mp4`) using the file panel.
4. Run these cells:

```
!pip install -q ultralytics transformers opencv-python
```
```
!python pipeline.py --input road.mp4 --output result.mp4 --width 1280 --skip 2
```
```
from google.colab import files
files.download("result.mp4")
```

### Option 2: Local computer

1. Install PyTorch with GPU support from pytorch.org (choose your CUDA version).
2. Install the other libraries: `pip install -r requirements.txt`
3. Run: `python pipeline.py --input road.mp4 --output result.mp4 --width 1280 --skip 2`

The first run downloads the model weights, so it needs an internet connection.

## Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--input` | required | Input video file |
| `--output` | `result.mp4` | Output video file |
| `--width` | 1280 | Resize each frame to this width before processing (0 keeps the original size) |
| `--skip` | 1 | Run the vehicle and road models on every K-th frame and reuse the results in between |
| `--conf` | 0.35 | Minimum confidence for a vehicle detection |
| `--imgsz` | 640 | Input size for YOLO (larger finds smaller, far-away vehicles) |
| `--yolo` | `yolov8m.pt` | YOLO weights (use `yolov8n.pt` for a faster, lighter model) |
| `--seg` | SegFormer-B2 Cityscapes | Segmentation model name on Hugging Face |
| `--no-vehicles` | off | Turn off vehicle detection |
| `--no-drivable` | off | Turn off drivable area segmentation |
| `--no-lanes` | off | Turn off lane detection |

The three `--no-...` options are useful for measuring how much each module costs in speed.

## Results (Colab T4 GPU, 1280 px wide, `--skip 2`, 715-frame video)

| Configuration | FPS | Time per frame (ms) |
|---|---|---|
| All three modules | 5.2 | 192 |
| Without vehicle detection | 6.2 | 161 |
| Without lane detection | 6.4 | 156 |
| Without drivable area | 6.6 | 152 |

Each number is from a single run, so treat them as approximate. No single module is the main bottleneck. The pipeline works on recorded video but is not real-time (real-time would need roughly 25 to 30 FPS).

## Known limitations

- **Auto-rickshaws are labeled "truck".** COCO has no auto-rickshaw class, and they are common on Indian roads.
- **Lane lines follow road boundaries more than the dashed centre line** on our test video. Faded paint and tree shadows make classical lane detection difficult.
- **Straight-line model.** Lanes are fitted as straight lines, so sharp curves are not followed well.
- **Speed.** About 5 FPS on a Colab T4 GPU.
- **Cityscapes road model** was trained on European city roads, so its accuracy on Indian roads is not measured here.
- Only one test video has been used so far, and no accuracy metrics (mAP, mIoU) were computed.

## Files

```
pipeline.py            main script
requirements.txt       Python libraries
README.md              this file
CODE_WALKTHROUGH.md    plain-language explanation of the code
road.mp4               input test video
result.mp4             annotated output video
```

## Planned work for the major project

- Fine-tune the models on Indian road data, including an auto-rickshaw class
- Replace the classical lane detector with a deep learning lane model
- Evaluate with proper metrics: mAP for vehicles, mIoU for the drivable area, accuracy for lanes
- Add vehicle tracking and counting for smart-road applications
- Improve speed with lighter models and lower resolution
"# smart-road-perception" 

import argparse
import time

import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO

# COCO class ids: 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
VEHICLE_CLASSES = [1, 2, 3, 5, 7]
# Cityscapes class id for "road"
ROAD_ID = 0


class LaneDetector:
    """Paint-based lane detection restricted to the road region."""

    def __init__(self, alpha=0.85, max_miss=10):
        self.alpha = alpha          # smoothing (higher = steadier lines)
        self.max_miss = max_miss    # frames to keep a line after it disappears
        self.left = None
        self.right = None
        self.miss_l = 0
        self.miss_r = 0
        self.kernel = None

    def _paint_mask(self, frame, road, boxes):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k = max(15, (w // 40) | 1)  # odd kernel, a bit wider than a lane stripe
        if self.kernel is None or self.kernel.shape[0] != k:
            self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # white top-hat keeps thin bright stripes and removes large bright areas
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.kernel)
        paint = ((tophat > 35) & (gray > 130)).astype(np.uint8) * 255
        if road is not None:
            road_d = cv2.dilate(road * 255, np.ones((9, 9), np.uint8))
            paint = cv2.bitwise_and(paint, road_d)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(paint, (x1, y1), (x2, y2), 0, -1)
        paint[: int(0.5 * h)] = 0  # nothing above the rough horizon
        return paint

    @staticmethod
    def _fit(segs, h):
        """Length-weighted fit of x = m*y + b. Returns (line, support) or None."""
        if not segs:
            return None
        xs, ys, ws = [], [], []
        for x1, y1, x2, y2, length in segs:
            xs += [x1, x2]
            ys += [y1, y2]
            ws += [length, length]
        ys = np.array(ys, float)
        xs = np.array(xs, float)
        if np.ptp(ys) < 5:
            return None
        m, b = np.polyfit(ys, xs, 1, w=np.array(ws, float))
        y1, y2 = h, int(0.62 * h)
        line = np.array([m * y1 + b, y1, m * y2 + b, y2], dtype=np.float32)
        return line, float(sum(ws))

    def _update(self, old, miss, new):
        if new is not None:
            line = new[0]
            old = line if old is None else self.alpha * old + (1 - self.alpha) * line
            return old, 0
        miss += 1
        if miss > self.max_miss:
            return None, miss
        return old, miss

    def detect(self, frame, road=None, boxes=()):
        h, w = frame.shape[:2]
        paint = self._paint_mask(frame, road, boxes)
        edges = cv2.Canny(paint, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                                minLineLength=w // 25, maxLineGap=w // 12)
        left_s, right_s = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                dx, dy = int(x2) - int(x1), int(y2) - int(y1)
                if dx == 0:
                    continue
                slope = dy / dx
                if abs(slope) < 0.3 or abs(slope) > 5:  # not lane-like
                    continue
                seg = (int(x1), int(y1), int(x2), int(y2), float(np.hypot(dx, dy)))
                (left_s if slope < 0 else right_s).append(seg)

        left = self._fit(left_s, h)
        right = self._fit(right_s, h)

        # left and right lines must not cross; if they do, keep the stronger one
        if left is not None and right is not None:
            lb, lt = left[0][0], left[0][2]
            rb, rt = right[0][0], right[0][2]
            if lb >= rb or lt > rt:
                if left[1] >= right[1]:
                    right = None
                else:
                    left = None

        self.left, self.miss_l = self._update(self.left, self.miss_l, left)
        self.right, self.miss_r = self._update(self.right, self.miss_r, right)
        return self.left, self.right


def draw_lanes(frame, left, right):
    for line in (left, right):
        if line is not None:
            x1, y1, x2, y2 = line.astype(int)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 5)
    return frame


def resize_to_width(frame, width):
    h, w = frame.shape[:2]
    if width and w > width:
        scale = width / w
        frame = cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def segment_road(seg, processor, frame, device):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, return_tensors="pt").to(device)
    if device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()
    with torch.no_grad():
        logits = seg(**inputs).logits
    logits = torch.nn.functional.interpolate(
        logits.float(), size=(h, w), mode="bilinear", align_corners=False)
    return (logits.argmax(1)[0] == ROAD_ID).cpu().numpy().astype(np.uint8)


def detect_vehicles(yolo, frame, conf, imgsz, device):
    res = yolo.predict(frame, classes=VEHICLE_CLASSES, conf=conf, imgsz=imgsz,
                       verbose=False, device=0 if device == "cuda" else "cpu")[0]
    dets = []
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        dets.append((x1, y1, x2, y2, f"{res.names[int(box.cls)]} {float(box.conf):.2f}"))
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="result.mp4")
    ap.add_argument("--yolo", default="yolov8m.pt")
    ap.add_argument("--seg", default="nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO input size (960 finds smaller far-away cars)")
    ap.add_argument("--width", type=int, default=1280, help="resize frames to this width (0 = keep original)")
    ap.add_argument("--skip", type=int, default=1, help="run YOLO + SegFormer every K-th frame")
    ap.add_argument("--no-lanes", action="store_true")
    ap.add_argument("--no-drivable", action="store_true")
    ap.add_argument("--no-vehicles", action="store_true")
    args = ap.parse_args()
    args.skip = max(1, args.skip)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yolo = YOLO(args.yolo)
    processor = SegformerImageProcessor.from_pretrained(args.seg)
    seg = SegformerForSemanticSegmentation.from_pretrained(args.seg).to(device).eval()
    if device == "cuda":
        seg = seg.half()
    lanes = LaneDetector()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Video has no readable frames")
    frame = resize_to_width(frame, args.width)
    h, w = frame.shape[:2]
    print(f"Processing at {w}x{h}")
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    road, dets = None, []
    n, t0 = 0, time.time()
    while ok:
        vis = frame.copy()
        heavy = (n % args.skip == 0)

        # 1) Drivable area (road class from SegFormer)
        if not args.no_drivable:
            if heavy or road is None:
                road = segment_road(seg, processor, frame, device)
            green = np.zeros_like(vis)
            green[road == 1] = (0, 200, 0)
            vis = cv2.addWeighted(vis, 1.0, green, 0.4, 0)

        # 2) Vehicles (detected before lanes so lanes can ignore them)
        if not args.no_vehicles and heavy:
            dets = detect_vehicles(yolo, frame, args.conf, args.imgsz, device)

        # 3) Lanes
        if not args.no_lanes:
            road_for_lanes = None if args.no_drivable else road
            left, right = lanes.detect(frame, road_for_lanes, [d[:4] for d in dets])
            vis = draw_lanes(vis, left, right)

        # 4) Draw vehicles on top
        if not args.no_vehicles:
            for x1, y1, x2, y2, label in dets:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(vis, label, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        n += 1
        fps = n / (time.time() - t0)
        cv2.putText(vis, f"FPS: {fps:.1f}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(vis)
        if n % 30 == 0:
            print(f"{n}/{total} frames, {fps:.1f} FPS")

        ok, frame = cap.read()
        if ok:
            frame = resize_to_width(frame, args.width)

    cap.release()
    out.release()
    print(f"Done. Saved {args.output} ({n} frames, avg {n / (time.time() - t0):.1f} FPS)")


if __name__ == "__main__":
    main()

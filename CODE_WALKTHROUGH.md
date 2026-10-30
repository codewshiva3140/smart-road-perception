# Code Walkthrough: pipeline.py

This file explains what the code does, in plain language, so you can describe it in a viva.

## The big picture

The script reads a video one frame at a time. For every frame it does three jobs, then draws the results and saves the frame to a new video.

```
video frame
   |
   +--> resize to 1280 px wide
   |
   +--> 1. Drivable area : SegFormer labels every pixel, "road" pixels become green
   |
   +--> 2. Vehicles      : YOLOv8 finds vehicles and gives boxes + labels
   |
   +--> 3. Lanes         : OpenCV finds bright painted stripes on the road, fits two lines
   |
   +--> draw everything on the frame, write it to result.mp4
```

Nothing is trained in this script. The models are downloaded already trained.

## Imports and constants

- `cv2` (OpenCV) reads and writes video and does the image processing for lanes.
- `torch` runs the neural networks on the GPU.
- `ultralytics.YOLO` is the vehicle detector.
- `SegformerForSemanticSegmentation` is the road segmentation model from Hugging Face.
- `VEHICLE_CLASSES = [1, 2, 3, 5, 7]` are the COCO class numbers for bicycle, car, motorcycle, bus and truck. Only these are kept.
- `ROAD_ID = 0` is the number of the "road" class in the Cityscapes dataset.

## Function: segment_road (drivable area)

1. Convert the frame from BGR (OpenCV's order) to RGB (what the model expects).
2. The SegFormer processor prepares the image (resizing and normalising it).
3. The model outputs, for every pixel, a score for each of 19 classes (road, sidewalk, car, building, and so on).
4. Those scores are stretched back up to the frame size.
5. `argmax` picks the highest-scoring class for each pixel. Pixels where the class is "road" become 1, all others 0. This is the **road mask**.

On the GPU the model runs in half precision (`.half()`), which is faster and uses less memory.

## Function: detect_vehicles

Runs YOLOv8 on the frame, keeps only the vehicle classes, and drops anything below the confidence threshold (default 0.35). It returns each vehicle's box corners and a label like `car 0.91`.

## Class: LaneDetector (lane lines)

This is classical computer vision, with no neural network. It has these steps.

**1. Find bright paint (`_paint_mask`)**
- Convert to grayscale.
- Apply a **white top-hat filter**. This keeps thin bright shapes (like painted stripes) and removes large bright areas (like the sky or a white car).
- Keep only pixels that are both "locally bright" (top-hat above 35) and bright overall (gray above 130).
- Keep only pixels inside the road mask (slightly enlarged), so trees and buildings are ignored.
- Black out the vehicle boxes so cars are not mistaken for lane lines.
- Ignore the top half of the image, where the sky and the horizon are.

**2. Find line segments (`detect`)**
- **Canny** finds edges in the paint mask.
- The **Hough transform** (`HoughLinesP`) turns those edges into straight line segments.
- Segments that are almost horizontal or almost vertical are discarded, because lane lines in a forward-facing camera slope diagonally.
- Segments sloping one way are treated as the **left** line, and the other way as the **right** line.

**3. Fit one line per side (`_fit`)**
All the segments on one side are combined into a single straight line using a weighted fit, where longer segments count more.

**4. Sanity check**
If the left and right lines cross each other, which cannot happen on a real road, the weaker one is dropped.

**5. Smoothing (`_update`)**
The new line is blended with the previous one (85% old, 15% new), so lines don't jump between frames. If a line is not found, the old one is kept for up to 10 frames and then removed.

## Function: resize_to_width

Shrinks large frames (our video is probably 4K) to 1280 px wide before processing. This makes everything much faster, with little loss in the results.

## Function: main (puts it together)

1. Read the command-line options.
2. Choose the GPU if one is available, and load the models.
3. Open the video and read the first frame to find the output size.
4. Loop over every frame:
   - On every K-th frame (`--skip`) run the **road model** and the **vehicle model**. On the frames in between, reuse the last results. This saves a lot of time because the scene changes little between neighbouring frames.
   - Run the **lane detector** on every frame (it needs to be smooth).
   - Draw the green road tint, the yellow lanes, and the red vehicle boxes with labels.
   - Print the running FPS on the frame and write it to the output video.
5. Print the average FPS at the end.

## Terms you may be asked about

- **Semantic segmentation**: labelling every pixel of an image with a class (road, car, sky...).
- **Object detection**: finding objects and drawing a box around each one.
- **YOLO**: a fast object detector that looks at the whole image once. "v8m" means version 8, medium size.
- **SegFormer**: a transformer-based segmentation model.
- **COCO / Cityscapes**: public datasets the models were trained on. COCO has everyday objects. Cityscapes has street scenes from European cities.
- **Pretrained model**: a model already trained by someone else, used here without further training.
- **Canny edge detector**: finds places where brightness changes sharply.
- **Hough transform**: finds straight lines from edge points.
- **FPS**: frames processed per second.

## What to say about the limits

- Auto-rickshaws are labelled "truck" because the COCO dataset has no auto-rickshaw class.
- The lane detector finds road boundaries more than the dashed centre line on this video, because the markings are faint and partly hidden by tree shadows.
- Lines are straight, so curves are not followed well.
- The road model was trained on European roads, so its accuracy on Indian roads has not been measured.
- All of these are the reasons the major project plans fine-tuning and a deep learning lane model.

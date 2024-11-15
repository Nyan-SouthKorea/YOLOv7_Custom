# Custom YOLOv7 Object Detection with OpenCV

This repository provides a custom object detection implementation using YOLOv7 and OpenCV. The `Custom_YOLOv7` class loads a YOLOv7 model and allows for real-time object detection on webcam video input, with bounding boxes and class labels displayed on the screen.

## Requirements

- Python 3.7 or higher
- OpenCV
- PyTorch

## Installation

1. Clone this repository: git clone https://github.com/your-repository-url cd your-repository-url


2. Install the required packages:

```python

3. Download the `best.pt` model and place it in the root directory of the project.

## Usage

### Running Object Detection with YOLOv7

Run the following code to start real-time object detection using a webcam:

```python
import cv2

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # Set frame width
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set frame height

# Define class labels
class_list = ['person', 'dog', 'cat']

# Initialize the YOLOv7 model
model = Custom_YOLOv7('best.pt', class_list)

while True:
 ret, img = cap.read()
 if not ret:
     print('Error accessing webcam')
     break
 
 # Run predictions and draw results
 dic_list = model.predict(img)
 cv2.imshow('Object Detection', model.draw(img, dic_list))
 
 # Press 'q' to exit
 if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cv2.destroyAllWindows()

```
### Code Overview

- **`Custom_YOLOv7`**: Main class that initializes the YOLOv7 model and provides `predict`, `nms_dic_list`, and `draw` methods.
  - **`__init__`**: Loads the YOLOv7 model and sets the class list.
  - **`predict`**: Preprocesses the input image, runs object detection, and returns bounding box results with confidence scores.
  - **`nms_dic_list`**: Applies Non-Maximum Suppression (NMS) to remove duplicate bounding boxes.
  - **`draw`**: Draws bounding boxes and labels on the image.
- **`smart_resize`**: Resizes an image while maintaining its aspect ratio.

## Key Functions

- **Non-Maximum Suppression (NMS)**: Filters overlapping bounding boxes based on Intersection over Union (IoU).
- **FPS Display**: Displays frames per second (FPS) on the output video.

## Example

The following code snippet shows how to use the `Custom_YOLOv7` class for object detection.

```python
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
class_list = ['person', 'dog', 'cat']

model = Custom_YOLOv7('best.pt', class_list)
while True:
    ret, img = cap.read()
    if not ret:
        print('Error accessing webcam')
        break
    dic_list = model.predict(img)
    cv2.imshow('Detection', model.draw(img, dic_list))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
```

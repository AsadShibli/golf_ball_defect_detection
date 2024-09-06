# Golf Ball Defect Detection 
This repository contains the code and resources for detecting defects on golf balls using a custom YOLOv8 model. The dataset was captured, manually annotated in Roboflow, and used to train and run inference on the YOLOv8 model. This project leverages Google Colab, Roboflow, and Supervision libraries for annotation and detection.

## Demo
![2](https://github.com/user-attachments/assets/45f22a61-8ce2-4d0e-a770-c433f9157e53)

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)

## Installation
To get started, you will need to install the necessary libraries for running the detection model and annotations.
```bash
!pip install -q roboflow
!pip install ultralytics==8.0.196
!pip install supervision
```
Ensure that the API key is added to your environment variables or replaced in the script directly.

## Training
We use YOLOv8 for detecting defects on golf balls. The training was performed using the YOLOv8x model and a custom dataset downloaded from Roboflow.
```python
import os
HOME = os.getcwd()

# Create a folder for the dataset
!mkdir {HOME}/datasets
%cd {HOME}/datasets

# Download the dataset from Roboflow
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("sm-group").project("golf-ball-cover-defect-data")
version = project.version(1)
dataset = version.download("yolov8")

%cd {HOME}

# Train the model using YOLOv8
!yolo task=detect mode=train model=yolov8x.pt data={dataset.location}/data.yaml epochs=50 imgsz=600 plots=True
```
The dataset contains annotated images of golf balls with identified defects. The training process includes 50 epochs and an image size of 600x600 pixels.

## Inference

For inference, a custom YOLOv8 model is used to detect defects in new golf ball images.
```python
from ultralytics import YOLO
import supervision as sv
import cv2

# Load the trained model from Google Drive
model = YOLO('/content/drive/MyDrive/client project/defect/weights/best (6).pt')
bounding_box = sv.BoxAnnotator()

# Load an image
image_path = "/content/image (2).png"
frame = cv2.imread(image_path)

# Run inference
results = model(frame)

# Parse the results
detections = sv.Detections.from_ultralytics(results[0])

# Annotate results on the frame
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(detections['class_name'], detections.confidence)
]

label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
annotated_frame = bounding_box.annotate(scene=frame.copy(), detections=detections)
annotated_frame = label_annotator.annotate(scene=annotated_frame.copy(), detections=detections, labels=labels)

# Display the results
sv.plot_image(annotated_frame)
```
This script loads a custom pre-trained model, runs inference on an input image, and then visualizes the results with bounding boxes and labels.

## Results

After running the inference, the model outputs the predicted bounding boxes and labels on the input images. The results can be visualized using the `supervision` library for clear and annotated detections.





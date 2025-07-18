from ultralytics import YOLO
import os
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
path = os.path.join('.','config.yaml')

# Use the model
results = model.train(data=path, epochs=100)  # train the model

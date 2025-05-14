import gradio as gr
import torch
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from PIL import Image
import requests
import io
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path  # Fixed from "pathly" to "pathlib"
import json

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)

# Download model if it doesn't exist
model_path = "models/yolov11x_best.pt"
if not os.path.exists(model_path):
    url = "https://github.com/moured/YOLOv11-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov11x_best.pt"
    print(f"Downloading model from {url}...")
    r = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(r.content)
    print(f"Model downloaded to {model_path}")

# Load the model
model = YOLO(model_path)
print("Model loaded successfully!")

# Define classes (from DocLayNet dataset)
CLASSES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", 
           "Picture", "Section-header", "Table", "Text", "Title"]

# Define visual elements we want to extract
VISUAL_ELEMENTS = ["Picture", "Caption", "Table", "Formula"]

# Define colors for visualization
COLORS = sv.ColorPalette.default()

# Set up the annotator
box_annotator = sv.BoxAnnotator(color=COLORS)

def predict_layout(image):
    if image is None:
        return None, None, None
    
    # Convert to numpy array if it's not already
    if isinstance(image, np.ndarray):
        img = image
    else:
        img = np.array(image)
    
    # Get image dimensions
    img_height, img_width = img.shape[:2]
    
    # Run inference
    results = model(img)[0]
    
    # Format detections
    detections = sv.Detections.from_ultralytics(results)
    
    # Get class names
    class_ids = detections.class_id
    labels = [f"{CLASSES[class_id]} {confidence:.2f}" 
              for class_id, confidence in zip(class_ids, detections.confidence)]
    
    # Get annotated frame
    annotated_image = box_annotator.annotate(
        scene=img.copy(), 
        detections=detections,
        labels=labels
    )
    
    # Extract bounding boxes for all visual elements
    boxes_data = []
    for i, (class_id, xyxy, confidence) in enumerate(zip(detections.class_id, detections.xyxy, detections.confidence)):
        class_name = CLASSES[class_id]
        
        # Include all visual elements (Pictures, Captions, Tables, Formulas)
        # You can add or remove classes based on what you consider "visual elements"
        if class_name in VISUAL_ELEMENTS:
            x1, y1, x2, y2 = map(int, xyxy)
            width = x2 - x1
            height = y2 - y1
            
            boxes_data.append({
                "class": class_name,
                "confidence": float(confidence),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "width": int(width),
                "height": int(height)
            })
    
    # Create DataFrame for display
    if boxes_data:
        df = pd.DataFrame(boxes_data)
        df = df[["class", "confidence", "x1", "y1", "x2", "y2", "width", "height"]]
    else:
        df = pd.DataFrame(columns=["class", "confidence", "x1", "y1", "x2", "y2", "width", "height"])
    
    # Convert to JSON for download
    json_data = json.dumps(boxes_data, indent=2)
    
    return annotated_image, df, json_data

# Function to download JSON
def download_json(json_data):
    if not json_data:
        return None
    return json_data

# Set up the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# YOLOv11x Document Layout Analysis for Visual Elements")
    gr.Markdown("Upload a document image to extract visual elements including diagrams, tables, formulas, and captions.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Document")
            analyze_btn = gr.Button("Analyze Layout", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Detected Layout")
    
    with gr.Row():
        with gr.Column():
            output_table = gr.DataFrame(label="Visual Elements Bounding Boxes")
            json_output = gr.JSON(label="JSON Output")
            download_btn = gr.Button("Download JSON")
            json_file = gr.File(label="Download JSON File", interactive=False)
    
    analyze_btn.click(
        fn=predict_layout,
        inputs=input_image,
        outputs=[output_image, output_table, json_output]
    )
    
    download_btn.click(
        fn=download_json,
        inputs=[json_output],
        outputs=[json_file]
    )
    
    gr.Markdown("## Detected Visual Elements")
    gr.Markdown("""
    This application detects and extracts coordinates for the following visual elements:
    
    - **Pictures**: Diagrams, photos, illustrations, flowcharts, etc.
    - **Tables**: Structured data presented in rows and columns
    - **Formulas**: Mathematical equations and expressions
    - **Captions**: Text describing pictures or tables
    
    For each element, the system returns:
    - Element type (class)
    - Confidence score (0-1)
    - Coordinates (x1, y1, x2, y2)
    - Width and height in pixels
    """)
    
    gr.Markdown("## About")
    gr.Markdown("""
    This demo uses YOLOv11x for document layout analysis, with a focus on extracting visual elements.
    Model from [moured/YOLOv11-Document-Layout-Analysis](https://github.com/moured/YOLOv11-Document-Layout-Analysis)
    """)
    
    # Add example images
    gr.Examples(
        examples=[
            "https://raw.githubusercontent.com/moured/YOLOv11-Document-Layout-Analysis/main/assets/sample1.png",
            "https://raw.githubusercontent.com/moured/YOLOv11-Document-Layout-Analysis/main/assets/sample2.png",
        ],
        inputs=input_image
    )

demo.launch()
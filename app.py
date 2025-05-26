import os
os.environ["GRADIO_TEMP_DIR"] = "./tmp"

import sys
import torch
import torchvision
import gradio as gr
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from visualization import visualize_bbox

# Create necessary directories
os.makedirs("tmp", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Define class mapping
id_to_names = {
    0: 'title', 
    1: 'plain text',
    2: 'abandon', 
    3: 'figure', 
    4: 'figure_caption', 
    5: 'table', 
    6: 'table_caption', 
    7: 'table_footnote', 
    8: 'isolate_formula', 
    9: 'formula_caption'
}

# Visual elements for extraction (can be customized)
VISUAL_ELEMENTS = ['figure', 'table', 'figure_caption', 'table_caption', 'isolate_formula']

def load_model():
    """Load the DocLayout-YOLO model from Hugging Face"""
    try:
        # Download model weights if they don't exist
        model_dir = snapshot_download(
            'juliozhao/DocLayout-YOLO-DocStructBench', 
            local_dir='./models/DocLayout-YOLO-DocStructBench'
        )
        
        # Select device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Import and load the model
        from doclayout_yolo import YOLOv10
        model = YOLOv10(os.path.join(
            os.path.dirname(__file__), 
            "models", 
            "DocLayout-YOLO-DocStructBench", 
            "doclayout_yolo_docstructbench_imgsz1024.pt"
        ))
        
        return model, device
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, 'cpu'

def recognize_image(input_img, conf_threshold, iou_threshold):
    """Process input image and detect document elements"""
    if input_img is None:
        return None, None
    
    try:
        # Load model (global model if already loaded)
        global model, device
        
        # Run prediction
        det_res = model.predict(
            input_img,
            imgsz=1024,
            conf=conf_threshold,
            device=device,
        )[0]
        
        # Extract detection results
        boxes = det_res.__dict__['boxes'].xyxy
        classes = det_res.__dict__['boxes'].cls
        scores = det_res.__dict__['boxes'].conf
        
        # Apply non-maximum suppression
        indices = torchvision.ops.nms(
            boxes=torch.Tensor(boxes), 
            scores=torch.Tensor(scores),
            iou_threshold=iou_threshold
        )
        
        boxes, scores, classes = boxes[indices], scores[indices], classes[indices]
        
        # Handle single detection case
        if len(boxes.shape) == 1:
            boxes = np.expand_dims(boxes, 0)
            scores = np.expand_dims(scores, 0)
            classes = np.expand_dims(classes, 0)
            
        # Visualize results
        vis_result = visualize_bbox(input_img, boxes, classes, scores, id_to_names)
        
        # Create DataFrame for extraction
        elements_data = []
        for i, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
            class_name = id_to_names[int(cls_id)]
            
            # Only extract visual elements if specified
            if not VISUAL_ELEMENTS or class_name in VISUAL_ELEMENTS:
                x1, y1, x2, y2 = map(int, box)
                width = x2 - x1
                height = y2 - y1
                
                elements_data.append({
                    "class": class_name,
                    "confidence": float(score),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": width,
                    "height": height
                })
        
        # Convert to DataFrame for display
        import pandas as pd
        if elements_data:
            df = pd.DataFrame(elements_data)
            df = df[["class", "confidence", "x1", "y1", "x2", "y2", "width", "height"]]
        else:
            df = pd.DataFrame(columns=["class", "confidence", "x1", "y1", "x2", "y2", "width", "height"])
        
        return vis_result, df
        
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def gradio_reset():
    """Reset the UI"""
    return gr.update(value=None), gr.update(value=None), gr.update(value=None)

# Create basic HTML header
header_html = """
<div style="text-align: center; max-width: 900px; margin: 0 auto;">
    <div>
        <h1 style="font-weight: 900; margin-bottom: 7px;">
            Document Layout Analysis
        </h1>
        <p style="margin-top: 7px; font-size: 94%;">
            Detect and extract structured elements from document images using DocLayout-YOLO
        </p>
    </div>
</div>
"""

# Main execution
if __name__ == "__main__":
    # Load model
    model, device = load_model()
    
    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.HTML(header_html)
        
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Upload Document Image", interactive=True)
                
                with gr.Row():
                    clear_btn = gr.Button(value="Clear")
                    predict_btn = gr.Button(value="Detect Elements", interactive=True, variant="primary")
                
                with gr.Row():
                    conf_threshold = gr.Slider(
                        label="Confidence Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.25,
                    )
                    
                    iou_threshold = gr.Slider(
                        label="NMS IOU Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.45,
                    )
            
            with gr.Column():
                output_img = gr.Image(label="Detection Result", interactive=False)
                output_table = gr.DataFrame(label="Detected Visual Elements")
        
        with gr.Row():
            gr.Markdown("""
            ## Detected Elements
            This application detects and extracts the following elements from document images:
            
            - **Title**: Document and section titles
            - **Plain Text**: Regular paragraph text
            - **Figure**: Images, charts, diagrams, etc.
            - **Figure Caption**: Text describing figures
            - **Table**: Tabular data structures
            - **Table Caption**: Text describing tables
            - **Table Footnote**: Notes below tables
            - **Formula**: Mathematical equations
            - **Formula Caption**: Text describing formulas
            
            For each element, the system returns coordinates and confidence scores.
            """)
        
        # Connect events
        clear_btn.click(gradio_reset, inputs=None, outputs=[input_img, output_img, output_table])
        predict_btn.click(
            recognize_image, 
            inputs=[input_img, conf_threshold, iou_threshold], 
            outputs=[output_img, output_table]
        )
        
    # Launch the interface
port = int(os.environ.get("PORT", 8080))

demo.launch(share=True, server_name="0.0.0.0", server_port=port)

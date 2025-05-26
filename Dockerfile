FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ultralytics

# Create necessary directories
RUN mkdir -p tmp models

# Pre-download the model during build time
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('juliozhao/DocLayout-YOLO-DocStructBench', local_dir='./models/DocLayout-YOLO-DocStructBench', local_dir_use_symlinks=False)"

# Copy application code
COPY app.py visualization.py doclayout_yolo.py ./

# Environment variable for Gradio
ENV GRADIO_TEMP_DIR=/app/tmp
ENV PYTHONUNBUFFERED=1

# Set the port that Cloud Run will use
ENV PORT=8080

# Add specific GPU environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Command to run the application
CMD ["python", "app.py"]

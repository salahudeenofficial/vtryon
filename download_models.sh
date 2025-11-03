#!/bin/bash
set -e

# Download models using Python script for proper directory structure
# This ensures all config files and model structure are downloaded correctly
python3 download_models.py

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/diffusers
pip install -r requirements.txt
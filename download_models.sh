#!/bin/bash
set -e

# Download models using Python script for proper directory structure
# This ensures all config files and model structure are downloaded correctly
python3 download_models.py

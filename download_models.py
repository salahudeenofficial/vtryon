#!/usr/bin/env python3
"""
Download models using huggingface_hub for proper directory structure
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    # Download Qwen-Image-Edit-2509 to ./Qwen-Image-Edit-2509 directory
    model_dir = Path("./Qwen-Image-Edit-2509")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Downloading Qwen-Image-Edit-2509 model (complete repository)...")
    # Download the full repository structure - all files
    snapshot_download(
        repo_id="Qwen/Qwen-Image-Edit-2509",
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        repo_type="model",
    )
    
    # Download LoRA weights to models directory
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("[INFO] Downloading Qwen-Image-Lightning LoRA...")
    from huggingface_hub import hf_hub_download
    lora_path = hf_hub_download(
        repo_id="lightx2v/Qwen-Image-Lightning",
        filename="Qwen-Image-Lightning-4steps-V2.0.safetensors",
        local_dir=str(models_dir),
    )
    
    print(f"[OK] Model downloaded to {model_dir}")
    print(f"[OK] LoRA downloaded to {lora_path}")

if __name__ == "__main__":
    main()


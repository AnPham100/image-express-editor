#!/usr/bin/env python3
"""
Model Size Checker for Image Express Editor

This script provides commands to check the memory usage and sizes of all AI models used in the app.
Can be run independently or imported into other scripts.

Usage:
    python check_model_sizes.py
    python check_model_sizes.py --detailed
    python check_model_sizes.py --models-only
"""

import os
import sys
import argparse
from pathlib import Path

def get_model_info():
    """Get information about all models used in the app."""
    
    models_info = {
        'rembg': {
            'purpose': 'Background Removal',
            'typical_size': '~167 MB',
            'model_path': '~/.u2net/',
            'enabled_by_default': True,
            'description': 'U²-Net model for background removal'
        },
        'retinanet': {
            'purpose': 'Object Detection',
            'typical_size': '~145 MB',
            'model_path': 'Downloaded via torchvision',
            'enabled_by_default': True,
            'description': 'RetinaNet ResNet50 FPN V2 for general object detection'
        },
        'sam2': {
            'purpose': 'Segmentation',
            'typical_size': '~856 MB',
            'model_path': '~/.cache/sam2/sam2.1_hiera_large.pt',
            'enabled_by_default': True,
            'description': 'SAM2 Hiera Large model for precise segmentation'
        },
        'grounding_dino': {
            'purpose': 'Text-guided Object Detection',
            'typical_size': '~1.7 GB',
            'model_path': '~/.cache/huggingface/hub/models--IDEA-Research--grounding-dino-base',
            'enabled_by_default': True,
            'description': 'Grounding DINO for flexible text-based object detection'
        },
        'sdxl_controlnet': {
            'purpose': 'Image-to-Image Generation',
            'typical_size': '~12 GB',
            'model_path': '~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0',
            'enabled_by_default': False,
            'description': 'SDXL with ControlNet for guided image generation'
        },
        'sdxl_txt2img': {
            'purpose': 'Text-to-Image Generation',
            'typical_size': '~12 GB',
            'model_path': '~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0',
            'enabled_by_default': False,
            'description': 'SDXL for text-to-image generation'
        },
        'sdxl_inpaint': {
            'purpose': 'AI Inpainting/Outpainting',
            'typical_size': '~12 GB',
            'model_path': '~/.cache/huggingface/hub/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1',
            'enabled_by_default': False,
            'description': 'SDXL Inpainting for AI-powered inpainting and outpainting'
        }
    }
    
    return models_info

def check_model_files():
    """Check which model files actually exist on disk."""
    models_info = get_model_info()
    existing_models = {}
    
    for model_name, info in models_info.items():
        model_path = os.path.expanduser(info['model_path'])
        
        exists = False
        actual_size = 0
        
        if os.path.exists(model_path):
            if os.path.isfile(model_path):
                exists = True
                actual_size = os.path.getsize(model_path)
            elif os.path.isdir(model_path):
                exists = True
                # Calculate directory size
                actual_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
        
        existing_models[model_name] = {
            **info,
            'exists': exists,
            'actual_size_bytes': actual_size,
            'actual_size_formatted': format_size(actual_size) if actual_size > 0 else 'N/A'
        }
    
    return existing_models

def format_size(size_bytes):
    """Format bytes into human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def print_summary():
    """Print a summary of all models."""
    print("🧠 AI MODEL SUMMARY - Image Express Editor")
    print("=" * 60)
    
    models = check_model_files()
    
    total_size = 0
    loaded_count = 0
    available_count = 0
    
    for model_name, info in models.items():
        status = "✅ Available" if info['exists'] else "❌ Not Downloaded"
        enabled = "🟢 Enabled" if info['enabled_by_default'] else "🔴 Disabled (Dev Mode)"
        
        print(f"\n📋 {model_name.upper()}")
        print(f"   Purpose: {info['purpose']}")
        print(f"   Status: {status}")
        print(f"   Default: {enabled}")
        print(f"   Expected: {info['typical_size']}")
        
        if info['exists']:
            print(f"   Actual: {info['actual_size_formatted']}")
            total_size += info['actual_size_bytes']
            available_count += 1
            if info['enabled_by_default']:
                loaded_count += 1
        
        print(f"   Location: {info['model_path']}")
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print(f"   Total Models: {len(models)}")
    print(f"   Available: {available_count}")
    print(f"   Enabled by Default: {loaded_count}")
    print(f"   Total Size on Disk: {format_size(total_size)}")
    
    # Development mode note
    print(f"\n💡 NOTE: In DEVELOPMENT_MODE, heavy SDXL models are disabled by default")
    print(f"   to speed up startup. Enable them by setting DEVELOPMENT_MODE = False")

def print_detailed():
    """Print detailed information about each model."""
    print("🔍 DETAILED MODEL ANALYSIS")
    print("=" * 80)
    
    models = check_model_files()
    
    for model_name, info in models.items():
        print(f"\n🏷️  {model_name.upper()}")
        print(f"{'─' * 50}")
        print(f"Purpose:        {info['purpose']}")
        print(f"Description:    {info['description']}")
        print(f"Expected Size:  {info['typical_size']}")
        print(f"File Path:      {info['model_path']}")
        
        if info['exists']:
            print(f"Status:         ✅ Downloaded and Available")
            print(f"Actual Size:    {info['actual_size_formatted']}")
            print(f"Size (bytes):   {info['actual_size_bytes']:,}")
        else:
            print(f"Status:         ❌ Not Downloaded")
            print(f"Actual Size:    N/A")
        
        print(f"Default State:  {'🟢 Enabled' if info['enabled_by_default'] else '🔴 Disabled (Dev Mode)'}")

def print_models_only():
    """Print just the model names and basic info."""
    models = get_model_info()
    
    print("📋 MODELS USED IN IMAGE EXPRESS EDITOR")
    print("-" * 40)
    
    for model_name, info in models.items():
        enabled_indicator = "✓" if info['enabled_by_default'] else "✗"
        print(f"{enabled_indicator} {model_name:<20} | {info['purpose']:<30} | {info['typical_size']}")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check AI model sizes for Image Express Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_model_sizes.py                    # Show summary
  python check_model_sizes.py --detailed         # Show detailed analysis  
  python check_model_sizes.py --models-only      # Show just model list
        """
    )
    
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed information about each model')
    parser.add_argument('--models-only', action='store_true',
                       help='Show just the model names and basic info')
    
    args = parser.parse_args()
    
    if args.detailed:
        print_detailed()
    elif args.models_only:
        print_models_only()
    else:
        print_summary()

if __name__ == "__main__":
    main()
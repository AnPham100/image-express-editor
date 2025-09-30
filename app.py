#!/usr/bin/env python3
"""
All-in-One Image Express Editor

A Streamlit app for comprehensive image processing including:
- Object detection (humans, cars, general objects)
- Background removal
- Shadow effects
- Realistic and cartoon image-to-image generation
- Advanced post-processing options

Based on the car image processing app but expanded for general image editing.
"""

import streamlit as st
import io
import os
import zipfile
from typing import List, Optional, Dict, Any, Tuple
import json
import time
from datetime import datetime, timedelta
import logging

# ML and image processing imports
from PIL import Image, ImageOps, ImageDraw, ImageFilter, ImageEnhance
import torch
from scipy import ndimage
import cv2
import numpy as np

# Deep learning imports
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
)
from rembg import new_session, remove
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_V2_Weights,
    retinanet_resnet50_fpn_v2,
)
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
import easyocr  # For license plate text recognition
import pytesseract  # Alternative OCR engine

# System monitoring imports (with fallbacks)
try:
    import psutil  # For system monitoring
except ImportError:
    psutil = None
    
try:
    import shutil  # For disk usage
except ImportError:
    shutil = None
    
import platform  # For system information

# -----------------------------------------------------------------------------
# AI Model Configuration - Enable/Disable Models

# Background Removal Models
ENABLE_REMBG = True                    # rembg for quick background removal

# Object Detection Models  
ENABLE_RETINANET = True                # RetinaNet for general object detection
ENABLE_GROUNDING_DINO = True           # Grounding DINO for flexible text-based detection

# Segmentation Models
ENABLE_SAM2 = True                     # SAM2 for precise segmentation

# Generative AI Models
ENABLE_SDXL_CONTROLNET = True          # SDXL ControlNet for image-to-image generation
ENABLE_SDXL_TXT2IMG = True             # SDXL for text-to-image generation  
ENABLE_SDXL_INPAINT = True             # SDXL Inpainting for AI-powered inpainting/outpainting

# Development flags
DEVELOPMENT_MODE = True
if DEVELOPMENT_MODE:
    # - Disable heavy models for faster development
    ENABLE_SDXL_CONTROLNET = False
    # ENABLE_SDXL_TXT2IMG = False
    ENABLE_SDXL_INPAINT = False
    # ENABLE_SAM2 = False

# -----------------------------------------------------------------------------
# Configuration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UI Settings
DEFAULT_LOG_DISPLAY_LIMIT = 10  # Number of logs to show by default

# Style presets
REALISTIC_PROMPT = "professional photograph, high quality, studio lighting, sharp details, commercial photography"
CARTOON_PROMPT = "cartoon style, animated, colorful, stylized, illustration, digital art"

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, ugly, bad lighting, watermark, text, logo"

# Model paths
SAM2_CHECKPOINT = os.path.expanduser("~/.cache/sam2/sam2.1_hiera_large.pt")
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# -----------------------------------------------------------------------------
# UI Helper Functions

def create_detection_method_ui(help_text: str = "Method for detecting objects in the image") -> str:
    """Create standardized detection method selectbox."""
    return st.selectbox(
        "Detection Method",
        ["retinanet", "grounding_dino"],
        format_func=lambda x: {
            "retinanet": "🔍 RetinaNet (General Objects)",
            "grounding_dino": "🎯 Grounding DINO (Text Query)"
        }[x],
        help=help_text
    )

def create_text_query_ui(default_query: str = "person. human. car. vehicle. object.", help_text: str = "Describe what objects to find (separate with periods)") -> str:
    """Create standardized text query input for Grounding DINO."""
    return st.text_input(
        "Search Query",
        default_query,
        help=help_text
    )

# -----------------------------------------------------------------------------
# Device Setup and Model Loading

@st.cache_resource
def setup_device():
    """Setup computing device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        device_info = f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        device_info = "🍎 Using Apple Silicon MPS acceleration"
    else:
        device = torch.device("cpu")
        device_info = "⚠️ Using CPU - processing will be slow"
    
    return device, device_info

@st.cache_resource
def load_models():
    """Load all AI models for image processing."""
    device, device_info = setup_device()
    models = {}

    # Create a status container to show loading progress
    status_container = st.empty()
    with status_container.container():
        with st.spinner("Loading AI models... This may take a few minutes on first run."):
            total_models = sum([ENABLE_REMBG, ENABLE_RETINANET, ENABLE_SAM2, ENABLE_GROUNDING_DINO, 
                            ENABLE_SDXL_CONTROLNET, ENABLE_SDXL_TXT2IMG, ENABLE_SDXL_INPAINT])
            current_model = 0
            progress_bar = st.progress(0)
            
            # ___ rembg for background removal ___
            if ENABLE_REMBG:
                st.write("📦 Loading rembg for background removal...")
                try:
                    # Configure REMBG to use ~/.cache/rembg directory
                    import os
                    rembg_cache_dir = os.path.expanduser("~/.cache/rembg")
                    os.makedirs(rembg_cache_dir, exist_ok=True)
                    
                    # Set REMBG cache directory via environment variable
                    os.environ['U2NET_HOME'] = rembg_cache_dir
                    
                    rembg_session = new_session()
                    models['rembg'] = rembg_session
                    current_model += 1
                    progress_bar.progress(current_model / total_models)
                except Exception as e:
                    st.error(f"Failed to load rembg: {e}")
                    models['rembg'] = None
            else:
                st.write("⏭️ Skipping rembg (disabled)")
                models['rembg'] = None

            # ___ RetinaNet for general object detection ___
            if ENABLE_RETINANET:
                st.write("📦 Loading RetinaNet for object detection...")
                try:
                    retinanet_transforms = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
                    retinanet_model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
                    retinanet_model.to(device)
                    retinanet_model.eval()
                    retinanet_categories = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
                    
                    models['retinanet'] = {
                        'model': retinanet_model,
                        'transforms': retinanet_transforms,
                        'categories': retinanet_categories,
                        'device': device
                    }
                    current_model += 1
                    progress_bar.progress(current_model / total_models)
                except Exception as e:
                    st.error(f"Failed to load RetinaNet: {e}")
                    models['retinanet'] = None
            else:
                st.write("⏭️ Skipping RetinaNet (disabled)")
                models['retinanet'] = None

            # ___ SAM2 for precise segmentation ___
            if ENABLE_SAM2:
                st.write("📦 Loading SAM2 for segmentation...")
                try:
                    sam2_dir = os.path.dirname(SAM2_CHECKPOINT)
                    os.makedirs(sam2_dir, exist_ok=True)
                    
                    if not os.path.exists(SAM2_CHECKPOINT):
                        st.info("📥 Downloading SAM2 checkpoint (first time only)...")
                        import urllib.request
                        checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
                        urllib.request.urlretrieve(checkpoint_url, SAM2_CHECKPOINT)
                        st.success("✅ SAM2 checkpoint downloaded")
                    
                    sam2_predictor = SAM2ImagePredictor(
                        build_sam2(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
                    )
                    models['sam2'] = sam2_predictor
                    current_model += 1
                    progress_bar.progress(current_model / total_models)
                except Exception as e:
                    st.error(f"Failed to load SAM2: {e}")
                    models['sam2'] = None
            else:
                st.write("⏭️ Skipping SAM2 (disabled)")
                models['sam2'] = None

            # ___ Grounding DINO for flexible object detection ___
            if ENABLE_GROUNDING_DINO:
                st.write("📦 Loading Grounding DINO...")
                try:
                    grounding_dino_model_id = "IDEA-Research/grounding-dino-base"
                    grounding_dino_processor = AutoProcessor.from_pretrained(grounding_dino_model_id)
                    grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                        grounding_dino_model_id, use_safetensors=True
                    ).to(device)
                    
                    models['grounding_dino'] = {
                        'model': grounding_dino_model,
                        'processor': grounding_dino_processor,
                        'device': device
                    }
                    current_model += 1
                    progress_bar.progress(current_model / total_models)
                except Exception as e:
                    st.error(f"Failed to load Grounding DINO: {e}")
                    models['grounding_dino'] = None
            else:
                st.write("⏭️ Skipping Grounding DINO (disabled)")
                models['grounding_dino'] = None

            # ___ Stable Diffusion XL for image-to-image generation ___
            if ENABLE_SDXL_CONTROLNET:
                st.write("📦 Loading SDXL ControlNet...")
                try:
                    dtype = torch.float16 if device.type == "cuda" else torch.float32
                    
                    eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler", use_safetensors=True
                    )
                    controlnet = ControlNetModel.from_pretrained(
                        "xinsir/controlnet-canny-sdxl-1.0", torch_dtype=dtype, use_safetensors=True
                    )
                    vae = AutoencoderKL.from_pretrained(
                        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype, use_safetensors=True
                    )
                    sdxl_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        controlnet=controlnet,
                        vae=vae,
                        torch_dtype=dtype,
                        scheduler=eulera_scheduler,
                        use_safetensors=True,
                    ).to(device)
                    
                    models['sdxl'] = sdxl_pipeline
                    current_model += 1
                    progress_bar.progress(current_model / total_models)
                except Exception as e:
                    st.error(f"Failed to load SDXL ControlNet: {e}")
                    models['sdxl'] = None
            else:
                st.write("⏭️ Skipping SDXL ControlNet (disabled)")
                models['sdxl'] = None

            # ___ Stable Diffusion XL for text-to-image generation ___
            if ENABLE_SDXL_TXT2IMG:
                st.write("📦 Loading SDXL Text-to-Image...")
                try:
                    # Reuse the same components for efficiency if ControlNet version loaded
                    if models.get('sdxl') is not None:
                        # Create text-to-image pipeline from the existing ControlNet pipeline
                        txt2img_pipeline = StableDiffusionXLPipeline(
                            vae=models['sdxl'].vae,
                            text_encoder=models['sdxl'].text_encoder,
                            text_encoder_2=models['sdxl'].text_encoder_2,
                            tokenizer=models['sdxl'].tokenizer,
                            tokenizer_2=models['sdxl'].tokenizer_2,
                            unet=models['sdxl'].unet,
                            scheduler=models['sdxl'].scheduler,
                        ).to(device)
                        models['sdxl_txt2img'] = txt2img_pipeline
                    else:
                        # Load standalone text-to-image pipeline
                        dtype = torch.float16 if device.type == "cuda" else torch.float32
                        txt2img_pipeline = StableDiffusionXLPipeline.from_pretrained(
                            "stabilityai/stable-diffusion-xl-base-1.0",
                            torch_dtype=dtype,
                            use_safetensors=True,
                        ).to(device)
                        models['sdxl_txt2img'] = txt2img_pipeline
                    current_model += 1
                    progress_bar.progress(current_model / total_models)
                except Exception as e:
                    st.error(f"Failed to load SDXL Text-to-Image: {e}")
                    models['sdxl_txt2img'] = None
            else:
                st.write("⏭️ Skipping SDXL Text-to-Image (disabled)")
                models['sdxl_txt2img'] = None

            # ___ SDXL Inpainting Pipeline for AI-powered inpainting ___
            if ENABLE_SDXL_INPAINT:
                st.write("📦 Loading SDXL Inpainting...")
                try:
                    from diffusers import StableDiffusionXLInpaintPipeline
                    dtype = torch.float16 if device.type == "cuda" else torch.float32
                    
                    sdxl_inpaint_pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                        torch_dtype=dtype,
                        use_safetensors=True,
                    ).to(device)
                    
                    models['sdxl_inpaint'] = sdxl_inpaint_pipeline
                    current_model += 1
                    progress_bar.progress(current_model / total_models)
                except Exception as e:
                    st.error(f"Failed to load SDXL Inpainting: {e}")
                    models['sdxl_inpaint'] = None
            else:
                st.write("⏭️ Skipping SDXL Inpainting (disabled)")
                models['sdxl_inpaint'] = None
        
        # Show summary of loaded models
        loaded_models = [name for name, model in models.items() if model is not None]
        disabled_models = [name for name, model in models.items() if model is None]

        if loaded_models:
            st.success(f"✅ Successfully loaded {len(loaded_models)} models: {', '.join(loaded_models)}")
        if disabled_models:
            st.info(f"⏭️ Disabled models: {', '.join(disabled_models)}")
        if DEVELOPMENT_MODE:
            st.warning("🚧 Running in DEVELOPMENT_MODE - Heavy AI models are disabled for faster startup")

    # Clear status container
    status_container.empty()

    # Store models in session state for later use
    st.session_state['models'] = models
    st.session_state['device_info'] = device_info

    return models, device_info

# Helper functions for size calculations
def format_size(size_bytes: int) -> str:
    """Format bytes into human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def get_model_parameters(model) -> int:
    """Count parameters in a PyTorch model."""
    if hasattr(model, 'parameters'):
        return sum(p.numel() for p in model.parameters())
    return 0

def get_object_size(obj) -> int:
    """Get approximate memory size of an object."""
    try:
        import sys
        return sys.getsizeof(obj)
    except:
        return 0

def get_model_disk_size(model_name: str) -> Dict[str, int]:
    """Get disk storage size for model files."""
    import os
    import glob
    from pathlib import Path
    
    disk_info = {'total_size': 0, 'files': [], 'estimated_size': 0}
    
    # Common model cache directories
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/torch/hub"),
        os.path.expanduser("~/.cache/sam2"),
        # os.path.expanduser("~/.u2net"), # Default rembg location
        os.path.expanduser("~/.cache/rembg"),
        os.path.expanduser("~/.cache/diffusers"),
        os.path.expanduser("~/.cache/transformers"),
    ]
    
    # Model-specific search patterns for files and Hugging Face folders
    search_patterns = {
        'rembg': {
            'files': ['*u2net*', '*rembg*'],
            'hf_folders': []
        },
        'retinanet': {
            'files': ['*retinanet_resnet50_fpn_v2*'],
            'hf_folders': []
        },
        'sam2': {
            'files': ['*sam2*'],
            'hf_folders': []
        },
        'grounding_dino': {
            'files': ['*grounding*', '*dino*'],
            'hf_folders': ['models--IDEA-Research--grounding-dino-base']
        },
        'sdxl': {
            'files': ['*stable-diffusion-xl*', '*controlnet*canny*', '*sdxl*vae*'],
            'hf_folders': [
                'models--stabilityai--stable-diffusion-xl-base-1.0',
                'models--xinsir--controlnet-canny-sdxl-1.0',
                'models--madebyollin--sdxl-vae-fp16-fix'
            ]
        },
        'sdxl_txt2img': {
            'files': ['*stable-diffusion-xl*'],
            'hf_folders': ['models--stabilityai--stable-diffusion-xl-base-1.0']
        },
        'sdxl_inpaint': {
            'files': ['*stable-diffusion*inpaint*'],
            'hf_folders': ['models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1']
        }
    }
    
    # Estimated sizes (in MB) when files can't be found
    estimated_sizes = {
        'rembg': 167,  # U2Net ONNX model
        'retinanet': 145,  # RetinaNet ResNet50 FPN
        'sam2': 856,  # SAM2 Hiera Large
        'grounding_dino': 1700,  # Grounding DINO
        'sdxl': 12000,  # SDXL ControlNet (UNet + VAE + Text Encoders + ControlNet)
        'sdxl_txt2img': 12000,  # SDXL Text2Image
        'sdxl_inpaint': 12000   # SDXL Inpainting
    }
    
    disk_info['estimated_size'] = estimated_sizes.get(model_name, 1000) * 1024 * 1024  # Convert to bytes
    
    model_patterns = search_patterns.get(model_name, {'files': [f'*{model_name}*'], 'hf_folders': []})
    
    try:
        total_size = 0
        found_files = []
        seen_files = set()  # Track unique file paths to avoid duplicates
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                
                # Search for direct file patterns
                for pattern in model_patterns['files']:
                    search_path = os.path.join(cache_dir, '**', pattern)
                    for file_path in glob.glob(search_path, recursive=True):
                        if os.path.isfile(file_path):
                            abs_path = os.path.abspath(file_path)
                            if abs_path not in seen_files:
                                try:
                                    file_size = os.path.getsize(abs_path)
                                    total_size += file_size
                                    found_files.append({
                                        'path': abs_path,
                                        'size': file_size,
                                        'name': os.path.basename(abs_path)
                                    })
                                    seen_files.add(abs_path)
                                except (OSError, IOError):
                                    continue
                
                # Search for Hugging Face model folders (only in HF hub cache)
                if cache_dir.endswith('/huggingface/hub'):
                    for hf_folder in model_patterns['hf_folders']:
                        hf_path = os.path.join(cache_dir, hf_folder)
                        if os.path.isdir(hf_path):
                            # Recursively calculate folder size
                            folder_size = 0
                            folder_files = []
                            
                            for root, dirs, files in os.walk(hf_path):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    abs_path = os.path.abspath(file_path)
                                    if abs_path not in seen_files:
                                        try:
                                            file_size = os.path.getsize(abs_path)
                                            folder_size += file_size
                                            folder_files.append({
                                                'path': abs_path,
                                                'size': file_size,
                                                'name': f"{hf_folder}/{os.path.relpath(abs_path, hf_path)}"
                                            })
                                            seen_files.add(abs_path)
                                        except (OSError, IOError):
                                            continue
                            
                            total_size += folder_size
                            found_files.extend(folder_files)
        
        # Special handling for SAM2 checkpoint
        if model_name == 'sam2':
            sam2_checkpoint = os.path.expanduser("~/.cache/sam2/sam2.1_hiera_large.pt")
            if os.path.exists(sam2_checkpoint):
                try:
                    abs_checkpoint = os.path.abspath(sam2_checkpoint)
                    if abs_checkpoint not in seen_files:
                        file_size = os.path.getsize(abs_checkpoint)
                        total_size += file_size
                        found_files.append({
                            'path': abs_checkpoint,
                            'size': file_size,
                            'name': os.path.basename(abs_checkpoint)
                        })
                        seen_files.add(abs_checkpoint)
                except (OSError, IOError):
                    pass
        
        disk_info['total_size'] = total_size
        disk_info['files'] = found_files
        
    except Exception as e:
        logger.warning(f"Error calculating disk size for {model_name}: {e}")
    
    return disk_info

def calculate_model_memory_size(obj) -> int:
    """Calculate actual memory usage for PyTorch models and complex objects."""
    total_size = 0
    
    try:
        import sys
        
        # For PyTorch models, calculate parameter memory
        if hasattr(obj, 'parameters'):
            for param in obj.parameters():
                if param.data is not None:
                    # Calculate memory: num_elements * bytes_per_element
                    param_size = param.data.numel() * param.data.element_size()
                    total_size += param_size
                    
                    # Add gradient memory if exists
                    if param.grad is not None:
                        grad_size = param.grad.numel() * param.grad.element_size()
                        total_size += grad_size
        
        # For dictionaries (like model collections), calculate recursively
        elif isinstance(obj, dict):
            for key, value in obj.items():
                if hasattr(value, 'parameters'):  # It's a PyTorch model
                    total_size += calculate_model_memory_size(value)
                else:
                    total_size += sys.getsizeof(value)
        
        # For SDXL pipelines and other objects with components - check components first
        elif hasattr(obj, 'unet') or hasattr(obj, 'vae') or hasattr(obj, 'text_encoder'):
            # This is likely an SDXL or similar pipeline
            if hasattr(obj, 'unet') and hasattr(obj.unet, 'parameters'):
                total_size += calculate_model_memory_size(obj.unet)
            if hasattr(obj, 'vae') and hasattr(obj.vae, 'parameters'):
                total_size += calculate_model_memory_size(obj.vae)  
            if hasattr(obj, 'text_encoder') and hasattr(obj.text_encoder, 'parameters'):
                total_size += calculate_model_memory_size(obj.text_encoder)
            if hasattr(obj, 'text_encoder_2') and hasattr(obj.text_encoder_2, 'parameters'):
                total_size += calculate_model_memory_size(obj.text_encoder_2)
            if hasattr(obj, 'controlnet') and hasattr(obj.controlnet, 'parameters'):
                total_size += calculate_model_memory_size(obj.controlnet)
                
        # For objects with state_dict (fallback for PyTorch models)
        elif hasattr(obj, 'state_dict'):
            try:
                state_dict = obj.state_dict()
                for param_tensor in state_dict.values():
                    if hasattr(param_tensor, 'numel') and hasattr(param_tensor, 'element_size'):
                        param_size = param_tensor.numel() * param_tensor.element_size()
                        total_size += param_size
            except:
                # If state_dict fails, use sys.getsizeof as fallback
                total_size = sys.getsizeof(obj)
        
        # For objects with model attribute (like SAM2 predictor)
        elif hasattr(obj, 'model') and hasattr(obj.model, 'parameters'):
            total_size += calculate_model_memory_size(obj.model)
            
        # For REMBG sessions (ONNX Runtime) - estimate size from model bytes
        elif hasattr(obj, 'inner_session') and hasattr(obj.inner_session, '_model_bytes'):
            try:
                # REMBG uses ONNX Runtime, get model size from _model_bytes
                if obj.inner_session._model_bytes:
                    total_size = len(obj.inner_session._model_bytes)
                    logger.info(f"REMBG ONNX model size from _model_bytes: {total_size:,} bytes")
                else:
                    # Fallback: estimate typical U2Net model size (~170MB)
                    total_size = 170 * 1024 * 1024  # ~170MB typical for U2Net
                    logger.info(f"REMBG ONNX model size estimated: {total_size:,} bytes")
            except:
                # Final fallback for REMBG
                total_size = 170 * 1024 * 1024  # ~170MB
                logger.info(f"REMBG ONNX model size fallback: {total_size:,} bytes")
        
        # Fallback to sys.getsizeof for simple objects
        else:
            total_size = sys.getsizeof(obj)
            
    except Exception as e:
        logger.warning(f"Error calculating model memory size: {e}")
        total_size = 0
    
    return total_size

def get_model_memory_usage() -> Dict[str, Dict]:
    """
    Calculate memory usage for all loaded models.
    
    Returns:
        Dictionary with model names and their memory information
    """
    models = st.session_state.get('models', {})

    # If session state is empty (e.g., after page refresh), get models from cache
    if not models:
        logger.info("Session state models empty, attempting to restore from cache")
        try:
            models, device_info = load_models()
            logger.info("Restored models from cache to session state")
        except Exception as e:
            logger.error(f"Failed to restore models from cache: {e}")
            models = {}
    
    # Debug logging
    logger.info(f"get_model_memory_usage called with {len(models)} models from session state")
    logger.info(f"Available keys in session state: {list(st.session_state.keys())}")
    for model_name, model_data in models.items():
        status = "loaded" if model_data is not None else "disabled"
        logger.info(f"  - {model_name}: {status}")
    
    model_sizes = {}
    
    # Check each model
    for model_name, model_data in models.items():
        if model_data is None:
            # Still get disk size information for disabled models
            disk_info = get_model_disk_size(model_name)
            disk_size_mb = disk_info['total_size'] / (1024 * 1024) if disk_info['total_size'] > 0 else 0
            estimated_size_mb = disk_info['estimated_size'] / (1024 * 1024)
            
            model_sizes[model_name] = {
                'status': 'disabled',
                'memory_mb': 0,
                'memory_formatted': '0 B',
                'parameters': 0,
                'parameters_formatted': 'N/A',
                'description': 'Model not loaded',
                'components': {},
                'disk_size': disk_info['total_size'],
                'disk_size_mb': round(disk_size_mb, 1),
                'disk_size_formatted': format_size(disk_info['total_size']),
                'estimated_size_mb': round(estimated_size_mb, 1),
                'estimated_size_formatted': format_size(disk_info['estimated_size']),
                'disk_files': disk_info['files']
            }
            continue
        
        total_params = 0
        total_memory = 0
        model_info = {'status': 'loaded', 'components': {}}
        
        try:
            if model_name == 'rembg':
                # rembg session (ONNX Runtime model)
                total_memory = calculate_model_memory_size(model_data)
                # REMBG typically uses U2Net with ~44M parameters
                total_params = 44_000_000  # Approximate parameter count for U2Net
                model_info['description'] = 'Background removal model (ONNX)'
                model_info['components']['model'] = 'U2Net ONNX (~44M parameters)'
                
            elif model_name == 'retinanet':
                # RetinaNet dictionary with model, transforms, etc.
                if isinstance(model_data, dict) and 'model' in model_data:
                    total_params = get_model_parameters(model_data['model'])
                    total_memory = calculate_model_memory_size(model_data['model'])  # Calculate just the model part
                    # Add approximate size for transforms and other components
                    total_memory += get_object_size(model_data.get('transforms', {}))
                    total_memory += get_object_size(model_data.get('categories', []))
                    model_info['description'] = 'Object detection model'
                    model_info['components']['model'] = f"{total_params:,} parameters"
                    
            elif model_name == 'sam2':
                # SAM2 predictor
                if hasattr(model_data, 'model'):
                    total_params = get_model_parameters(model_data.model)
                    total_memory = calculate_model_memory_size(model_data)
                    model_info['description'] = 'Segmentation model'
                    model_info['components']['model'] = f"{total_params:,} parameters"
                    
            elif model_name == 'grounding_dino':
                # Grounding DINO dictionary with model and processor
                if isinstance(model_data, dict):
                    if 'model' in model_data:
                        model_params = get_model_parameters(model_data['model'])
                        total_params += model_params
                        total_memory += calculate_model_memory_size(model_data['model'])
                        model_info['components']['model'] = f"{model_params:,} parameters"
                    if 'processor' in model_data:
                        proc_size = get_object_size(model_data['processor'])
                        total_memory += proc_size
                        model_info['components']['processor'] = format_size(proc_size)
                    model_info['description'] = 'Text-guided object detection'
                    
            elif model_name in ['sdxl', 'sdxl_txt2img', 'sdxl_inpaint']:
                # SDXL pipelines - these are large multi-component models
                # Use the new memory calculation which handles complex pipelines
                total_memory = calculate_model_memory_size(model_data)
                
                # Still calculate individual component parameters for display
                if hasattr(model_data, 'unet'):
                    unet_params = get_model_parameters(model_data.unet)
                    total_params += unet_params
                    model_info['components']['unet'] = f"{unet_params:,} parameters"
                    
                if hasattr(model_data, 'vae'):
                    vae_params = get_model_parameters(model_data.vae)
                    total_params += vae_params
                    model_info['components']['vae'] = f"{vae_params:,} parameters"
                    
                if hasattr(model_data, 'text_encoder'):
                    te_params = get_model_parameters(model_data.text_encoder)
                    total_params += te_params
                    model_info['components']['text_encoder'] = f"{te_params:,} parameters"
                    
                if hasattr(model_data, 'text_encoder_2'):
                    te2_params = get_model_parameters(model_data.text_encoder_2)
                    total_params += te2_params
                    model_info['components']['text_encoder_2'] = f"{te2_params:,} parameters"
                    
                model_info['description'] = f'Stable Diffusion XL {model_name.replace("sdxl_", "").replace("sdxl", "controlnet")}'
        
        except Exception as e:
            model_info['description'] = f'Error calculating size: {str(e)}'
            logger.error(f"Error calculating size for {model_name}: {e}")
            
        # Convert memory to MB for easier reading
        memory_mb = total_memory / (1024 * 1024) if total_memory > 0 else 0
        
        # Get disk size information
        disk_info = get_model_disk_size(model_name)
        disk_size_mb = disk_info['total_size'] / (1024 * 1024) if disk_info['total_size'] > 0 else 0
        estimated_size_mb = disk_info['estimated_size'] / (1024 * 1024)
        
        logger.info(f"Model {model_name}: RAM {total_memory:,} bytes = {memory_mb:.1f} MB, Disk {disk_info['total_size']:,} bytes = {disk_size_mb:.1f} MB, {total_params:,} parameters")
        
        model_sizes[model_name] = {
            'status': model_info['status'],
            'memory_mb': round(memory_mb, 1),
            'memory_formatted': format_size(total_memory),
            'parameters': total_params,
            'parameters_formatted': f"{total_params:,}" if total_params > 0 else "N/A",
            'description': model_info['description'],
            'components': model_info.get('components', {}),
            'disk_size': disk_info['total_size'],
            'disk_size_mb': round(disk_size_mb, 1),
            'disk_size_formatted': format_size(disk_info['total_size']),
            'estimated_size_mb': round(estimated_size_mb, 1),
            'estimated_size_formatted': format_size(disk_info['estimated_size']),
            'disk_files': disk_info['files']
        }
    
    return model_sizes

def get_system_info():
    """Get comprehensive system information including CPU, memory, and disk usage."""
    try:
        # Check if required modules are available
        if psutil is None:
            return {
                'error': 'psutil module not available. Install with: pip install psutil',
                'cpu': {}, 'memory': {}, 'swap': {}, 'disk': {}, 'system': {}, 'gpu': {}
            }
        
        # CPU Information
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
        
        # Memory Information  
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk Information
        if shutil:
            disk_usage = shutil.disk_usage("/")
            disk_total = disk_usage.total
            disk_used = disk_usage.used
            disk_free = disk_usage.free
        else:
            disk_total = disk_used = disk_free = 0
        
        # System Information
        system_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
        
        # GPU Information (if available)
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                'cuda_available': True,
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count(),
                'current_gpu': torch.cuda.current_device(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Unknown'
            }
            
            # GPU Memory info
            if torch.cuda.device_count() > 0:
                gpu_memory = torch.cuda.get_device_properties(0)
                gpu_info['total_memory'] = gpu_memory.total_memory
                gpu_info['memory_allocated'] = torch.cuda.memory_allocated(0)
                gpu_info['memory_reserved'] = torch.cuda.memory_reserved(0)
        else:
            gpu_info = {'cuda_available': False}
            
        return {
            'cpu': {
                'physical_cores': cpu_count,
                'logical_cores': cpu_count_logical,
                'current_frequency': cpu_freq.current if cpu_freq else 0,
                'min_frequency': cpu_freq.min if cpu_freq else 0,
                'max_frequency': cpu_freq.max if cpu_freq else 0,
                'usage_percent': cpu_percent
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percentage': memory.percent,
                'free': memory.free
            },
            'swap': {
                'total': swap.total,
                'used': swap.used,
                'free': swap.free,
                'percentage': swap.percent
            },
            'disk': {
                'total': disk_total,
                'used': disk_used,
                'free': disk_free,
                'percentage': (disk_used / disk_total * 100) if disk_total > 0 else 0
            },
            'system': system_info,
            'gpu': gpu_info
        }
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {
            'error': f"Failed to get system information: {str(e)}",
            'cpu': {}, 'memory': {}, 'swap': {}, 'disk': {}, 'system': {}, 'gpu': {}
        }

def display_system_status_tab():
    """Display comprehensive system status information in a tab."""
    st.subheader("🖥️ System Status")

    system_info = get_system_info()
    model_sizes = get_model_memory_usage()

    # ----- System Overview -----

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📟 CPU Usage", 
            value=f"{system_info['cpu'].get('usage_percent', 0):.1f}%",
            delta=f"{system_info['cpu'].get('logical_cores', 0)} Cores"
        )

    with col2:
        memory_used = system_info['memory'].get('used', 0)
        memory_total = system_info['memory'].get('total', 1)
        memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
        st.metric(
            label="💾 Memory Usage (App)", 
            value=f"{memory_percent:.1f}%",
            delta=f"{format_size(memory_used)} / {format_size(memory_total)}"
        )
        
    with col3:
        disk_percent = system_info['disk'].get('percentage', 0)
        disk_used = system_info['disk'].get('used', 0)
        disk_total = system_info['disk'].get('total', 1)
        st.metric(
            label="💿 Disk Usage", 
            value=f"{disk_percent:.1f}%",
            delta=f"{format_size(disk_used)} / {format_size(disk_total)}"
        )
        
    with col4:
        # AI Models loaded count
        loaded_models = sum(1 for info in model_sizes.values() if info['status'] == 'loaded')
        total_models = len(model_sizes)
        st.metric(
            label="🤖 AI Models", 
            value=f"{loaded_models}/{total_models}",
            delta="Loaded/Total"
        )

    # ----- Detailed System Information -----

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("🖥️ Hardware Details", expanded=False):
            st.write("System Information:")
            sys_info = system_info['system']
            st.write(f"• OS: {sys_info.get('platform', 'Unknown')} {sys_info.get('platform_release', '')}")
            st.write(f"• Architecture: {sys_info.get('architecture', 'Unknown')}")
            st.write(f"• Processor: {sys_info.get('processor', 'Unknown')}")
            st.write(f"• Python: {sys_info.get('python_version', 'Unknown')}")
            
            st.write("\nCPU Details:")
            cpu_info = system_info['cpu']
            st.write(f"• Physical Cores: {cpu_info.get('physical_cores', 'Unknown')}")
            st.write(f"• Logical Cores: {cpu_info.get('logical_cores', 'Unknown')}")
            
            if cpu_info.get('current_frequency', 0) > 0:
                st.write(f"• Current Frequency: {cpu_info['current_frequency']:.0f} MHz")
                st.write(f"• Max Frequency: {cpu_info.get('max_frequency', 0):.0f} MHz")
                
            # GPU Information
            st.write("\nGPU Information:")
            gpu_info = system_info['gpu']
            if gpu_info.get('cuda_available', False):
                st.write(f"• CUDA Available: ✅ Yes (v{gpu_info.get('cuda_version', 'Unknown')})")
                st.write(f"• GPU Name: {gpu_info.get('gpu_name', 'Unknown')}")
                st.write(f"• GPU Count: {gpu_info.get('gpu_count', 0)}")

                if 'total_memory' in gpu_info:
                    total_gpu_mem = gpu_info['total_memory']
                    allocated_gpu_mem = gpu_info.get('memory_allocated', 0)
                    reserved_gpu_mem = gpu_info.get('memory_reserved', 0)

                    st.write(f"• GPU Memory Total: {format_size(total_gpu_mem)}")
                    st.write(f"• GPU Memory Allocated: {format_size(allocated_gpu_mem)}")
                    st.write(f"• GPU Memory Reserved: {format_size(reserved_gpu_mem)}")
            else:
                st.write("• CUDA Available: ❌ No")
                
            # ML Compute Device Information
            st.write("\nML Compute Device:")
            ml_device_info = st.session_state.get('device_info', 'Not available')
            st.write(f"• {ml_device_info}")
            
            # Additional device details based on the type
            if torch.cuda.is_available():
                st.write("• Device Type: CUDA GPU (NVIDIA)")
                if torch.cuda.device_count() > 0:
                    device_name = torch.cuda.get_device_name(0)
                    st.write(f"• Active Device: {device_name}")
                    # Show compute capability if available
                    try:
                        props = torch.cuda.get_device_properties(0)
                        st.write(f"• Compute Capability: {props.major}.{props.minor}")
                        st.write(f"• Multiprocessor Count: {props.multi_processor_count}")
                    except:
                        pass
            elif torch.backends.mps.is_available():
                st.write("• Device Type: Apple Silicon MPS")
                st.write("• Optimized for: Apple M1/M2/M3 chips")
            else:
                st.write("• Device Type: CPU")
                st.write("• Note: GPU acceleration not available")

    with col2:
        with st.expander("📊 Memory & Storage", expanded=False):
            # Memory details
            st.write("System Memory:")
            memory_info = system_info['memory']
            st.write(f"• Total: {format_size(memory_info.get('total', 0))}")
            st.write(f"• Used: {format_size(memory_info.get('used', 0))} ({memory_info.get('percentage', 0):.1f}%)")
            st.write(f"• Available: {format_size(memory_info.get('available', 0))}")
            st.write(f"• Free: {format_size(memory_info.get('free', 0))}")
            
            # Swap details
            swap_info = system_info['swap']
            if swap_info.get('total', 0) > 0:
                st.write("\nSwap Memory:")
                st.write(f"• Total: {format_size(swap_info['total'])}")
                st.write(f"• Used: {format_size(swap_info.get('used', 0))} ({swap_info.get('percentage', 0):.1f}%)")
                st.write(f"• Free: {format_size(swap_info.get('free', 0))}")
            
            # Disk details
            st.write("\nDisk Storage:")
            disk_info = system_info['disk']
            st.write(f"• Total: {format_size(disk_info.get('total', 0))}")
            st.write(f"• Used: {format_size(disk_info.get('used', 0))} ({disk_info.get('percentage', 0):.1f}%)")
            st.write(f"• Free: {format_size(disk_info.get('free', 0))}")

    st.markdown("---")
    
    # ----- AI Models Status -----

    st.subheader("🤖 AI Models Status")

    # Model summary metrics
    total_memory_mb = sum(info['memory_mb'] for info in model_sizes.values() if info['status'] == 'loaded')
    total_parameters = sum(info['parameters'] for info in model_sizes.values() if info['status'] == 'loaded')
    loaded_models = sum(1 for info in model_sizes.values() if info['status'] == 'loaded')  # Recalculate here
    total_models = len(model_sizes)
    
    # Calculate total disk usage (actual + estimated for missing files)
    total_disk_mb = 0
    for info in model_sizes.values():
        if info.get('disk_size', 0) > 0:
            total_disk_mb += info.get('disk_size_mb', 0)
        else:
            total_disk_mb += info.get('estimated_size_mb', 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Models Loaded", f"{loaded_models}/{total_models}")
    with col2:
        st.metric("Memory Usage (Loaded)", format_size(total_memory_mb * 1024 * 1024))
    with col3:
        st.metric("Disk Usage (All Models)", format_size(total_disk_mb * 1024 * 1024))
    with col4:
        st.metric("Total Parameters", f"{total_parameters:,}")

    # Display individual model information
    for model_name, info in model_sizes.items():
        status_icon = "✅" if info['status'] == 'loaded' else "❌"
        with st.expander(f"{status_icon} **{model_name.upper()}** - {info['description']}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"Status: {info['status']}")
                if info['status'] == 'loaded':
                    st.write(f"Memory Usage: {format_size(info['memory_mb'] * 1024 * 1024)}")
                    st.write(f"Parameters: {info['parameters']:,}")
                else:
                    st.write(f"Reason: {info.get('description', 'Model not loaded')}")
                
                # Always show disk size information
                if info.get('disk_size', 0) > 0:
                    st.write(f"Disk Size (Actual): {info['disk_size_formatted']}")
                    if len(info.get('disk_files', [])) > 0:
                        st.write(f"Files Found: {len(info['disk_files'])} files")
                else:
                    st.write(f"Disk Size (Estimated): {info['estimated_size_formatted']}")
                    st.write(f"Files Found: Not cached locally")

            with col2:
                if info.get('components'):
                    st.write("Components:")
                    for comp_name, comp_info in info['components'].items():
                        st.write(f"• {comp_name}: {comp_info}")
                        
                # Show disk file details if available
                if info.get('disk_files') and len(info['disk_files']) > 0:
                    st.write("Largest Files:")
                    # Sort files by size and show top 3
                    sorted_files = sorted(info['disk_files'], key=lambda x: x['size'], reverse=True)[:3]
                    for file_info in sorted_files:
                        file_size = format_size(file_info['size'])
                        file_name = file_info['name'][:70] + '...' if len(file_info['name']) > 70 else file_info['name']
                        st.write(f"• {file_name}: {file_size}")

def display_processing_history_tab():
    """Display processing history in a tab."""
    st.subheader("📋 Processing History")

    # Processing logs
    if st.session_state.processing_logs:
        total_logs = len(st.session_state.processing_logs)
        
        # Initialize log display mode in session state
        if 'show_all_logs' not in st.session_state:
            st.session_state.show_all_logs = False
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)

        # Calculate statistics
        successful_ops = sum(1 for log in st.session_state.processing_logs if log.get('status') == 'success')
        failed_ops = total_logs - successful_ops
        # total_time = sum(log.get('processed_duration_seconds', 0) for log in st.session_state.processing_logs)
        total_time = sum(parse_duration_to_seconds(log.get('processed_duration', 0)) for log in st.session_state.processing_logs)
        avg_time = total_time / total_logs if total_logs > 0 else 0

        with col1:
            st.metric("Total Operations", total_logs)
        with col2:
            st.metric("Successful", successful_ops, delta=f"{(successful_ops/total_logs*100):.1f}%" if total_logs > 0 else "0%")
        with col3:
            st.metric("Failed", failed_ops)
        with col4:
            # st.metric("Avg Time", f"{avg_time:.3f}s")
            st.metric("Avg Time", format_duration(avg_time).split('.')[0])

        st.markdown("---")
        
        # Button to toggle between recent and all logs
        if total_logs > DEFAULT_LOG_DISPLAY_LIMIT:
            col1, col2 = st.columns([3, 1])
            with col2:
                if not st.session_state.show_all_logs:
                    if st.button("📋 Load All Logs", key="load_all_logs", help=f"Load all {total_logs} operation logs"):
                        st.session_state.show_all_logs = True
                        st.rerun()
                else:
                    if st.button("📋 Show Recent Only", key="show_recent_logs", help=f"Show only recent {DEFAULT_LOG_DISPLAY_LIMIT} logs"):
                        st.session_state.show_all_logs = False
                        st.rerun()
        
        # Display logs based on current mode
        if st.session_state.show_all_logs or total_logs <= DEFAULT_LOG_DISPLAY_LIMIT:
            # Show all logs
            st.write(f"Showing all {total_logs} operation logs:")
            display_processing_logs(st.session_state.processing_logs)
        else:
            # Show recent logs only
            display_count = min(DEFAULT_LOG_DISPLAY_LIMIT, total_logs)
            recent_logs = st.session_state.processing_logs[-display_count:]
            st.write(f"Showing recent {display_count} of {total_logs} operation logs:")
            display_processing_logs(recent_logs)
    else:
        st.info("No processing history available yet. Start processing some images to see the history here!")

# -----------------------------------------------------------------------------
# Message Processing Functions

def create_pipeline_message(
    source_image_id: str,
    operation_chain: List[str],
    operation_params: Dict[str, Dict] = None
) -> Dict[str, Any]:
    """
    Create initial standardized message for multi-step processing pipeline.
    
    Args:
        source_image_id: Gallery ID of the source image (already saved in gallery)
        operation_chain: List of operations to execute in sequence
        operation_params: Dictionary mapping operation names to their parameters
        
    Returns:
        Standardized message dictionary
    """
    message = {
        'status': 'init',
        'status_msg': '',
        'common': {
            'data': [{
                'type': 'image',
                'source': f'gallery:{source_image_id}' if source_image_id else None,
                'timestamp': datetime.now().isoformat()
            }],
            'operation_chain': operation_chain if operation_chain else [],
            'current_step': 0,
            'start_time': datetime.now().isoformat()
        }
    }

    # Initialize operation sections with parameters
    operation_params = operation_params or {}
    for operation in operation_chain:
        message[operation] = {
            'input_params': operation_params.get(operation, {})
        }
    
    return message

def update_message_with_result(
    message: Dict[str, Any],
    operation: str,
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update message with operation result.

    Args:
        message: Current message
        operation: Operation name
        result: Result from operation

    Returns:
        Updated message
    """   
    image_id = result.get('image_id', None)
    is_reprocessable_image = result.get('is_reprocessable_image', False)

    message['status'] = result.get('status', 'ERROR')
    message['status_msg'] = result.get('status_msg', '')

    # ----- Update operation section ----- 
    message[operation]['output'] = {
        'status': result.get('status', 'ERROR'),
        'status_msg': result.get('status_msg', ''),
        'metadata': result.get('metadata', {}),
        'image_id': image_id,
        'processed_duration': format_duration(result.get('processed_duration_seconds', 0)),
        'timestamp': datetime.now().isoformat(),
    }

    # ----- Update common section -----

    # Update common data if this is a re-processable image
    if is_reprocessable_image and image_id:
        message['common']['data'].append({
            'type': 'image',
            'source': f'gallery:{image_id}',
            'timestamp': datetime.now().isoformat()
        })

    # Update current step
    if 'operation_chain' in message.get('common', {}):
        operation_chain = message['common']['operation_chain']
        if operation in operation_chain:
            message['common']['current_step'] = operation_chain.index(operation) + 1

    # Update total time
    if 'start_time' in message.get('common', {}):
        # Calculate total time from ISO start_time
        start_dt = datetime.fromisoformat(message['common']['start_time'])
        total_seconds = (datetime.now() - start_dt).total_seconds()
        message['common']['total_time'] = format_duration(total_seconds)

    return message

def get_latest_image_from_message(message: Dict[str, Any]) -> Optional[Image.Image]:
    """
    Extract the latest image from message common data.
    
    Args:
        message: Standardized message dictionary
        
    Returns:
        PIL Image or None if no image found
    """
    if not message.get('common', {}).get('data'):
        return None
    
    # Get the latest data entry
    latest_data = message['common']['data'][-1]
    
    if latest_data.get('type') != 'image':
        return None
    
    source = latest_data.get('source', '')
    if source.startswith('gallery:'):
        gallery_id = source.replace('gallery:', '')
        if 'gallery' in st.session_state and gallery_id in st.session_state.gallery:
            return st.session_state.gallery[gallery_id]['image']
    
    return None

# -----------------------------------------------------------------------------
# Detection Functions

def perform_object_detection(image: Image.Image, method: str, models: Dict, text_query: str = "") -> Dict[str, List]:
    """Perform object detection using specified method."""
    if method == "retinanet":
        return detect_objects_retinanet(image, models)
    else:
        if not text_query:
            text_query = "person. human. car. vehicle. object."
        return detect_objects_grounding_dino(image, text_query, models)

def detect_objects_retinanet(image: Image.Image, models: Dict, score_threshold: float = 0.5) -> Dict[str, List]:
    """Detect objects using RetinaNet."""
    if 'retinanet' not in models or models['retinanet'] is None:
        raise RuntimeError("RetinaNet model not available")
    
    model_info = models['retinanet']
    image_tensor = model_info['transforms'](image).unsqueeze(0).to(model_info['device'])
    
    with torch.no_grad():
        preds = model_info['model'](image_tensor)
    
    boxes = preds[0]["boxes"].cpu().numpy()
    scores = preds[0]["scores"].cpu().numpy()
    labels = preds[0]["labels"].cpu().numpy()
    
    mask = scores > score_threshold
    boxes = boxes[mask].astype(int).tolist()
    labels = [model_info['categories'][label] for label in labels[mask]]
    scores = scores[mask].tolist()
    
    return {"boxes": boxes, "labels": labels, "scores": scores}

def detect_objects_grounding_dino(
    image: Image.Image, 
    text_query: str, 
    models: Dict, 
    box_threshold: float = 0.35,
    text_threshold: float = 0.25
) -> Dict[str, List]:
    """Detect objects using Grounding DINO with text queries."""
    if models.get('grounding_dino') is None:
        raise RuntimeError("Grounding DINO model not available")
    
    model_info = models['grounding_dino']
    inputs = model_info['processor'](images=image, text=text_query, return_tensors="pt")
    inputs = inputs.to(model_info['device'])
    
    with torch.no_grad():
        outputs = model_info['model'](**inputs)
    
    try:
        outputs = model_info['processor'].post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )
        boxes = outputs[0]["boxes"].cpu().numpy().astype(int).tolist()
        labels = outputs[0]["labels"]
        scores = outputs[0]["scores"].cpu().numpy().tolist()
    except TypeError:
        # Fallback for newer transformers versions
        outputs = model_info['processor'].post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image.size[::-1]],
        )
        boxes = outputs[0]["boxes"].cpu().numpy()
        labels = outputs[0]["labels"]
        scores = outputs[0]["scores"].cpu().numpy()
        # Filter by threshold
        valid_indices = scores >= box_threshold
        boxes = boxes[valid_indices].astype(int).tolist()
        labels = [labels[i] for i in range(len(labels)) if valid_indices[i]]
        scores = scores[valid_indices].tolist()

    return {"boxes": boxes, "labels": labels, "scores": scores}

def segment_object(image: Image.Image, box: List[int], models: Dict) -> np.ndarray:
    """Segment object using SAM2."""
    if models.get('sam2') is None:
        raise RuntimeError("SAM2 model not available")
    
    predictor = models['sam2']
    device = next(predictor.model.parameters()).device
    
    if device.type == "mps":
        with torch.no_grad():
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(box=box)
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(box=box)
    
    # Get best mask and fill holes
    sorted_indices = np.argsort(scores)[::-1]
    mask = masks[sorted_indices[0]].astype(bool)
    mask = ndimage.binary_fill_holes(mask)
    
    return mask

# -----------------------------------------------------------------------------
# License Plate Detection and OCR Functions

def detect_license_plates(image: Image.Image, models: Dict, confidence_threshold: float = 0.3) -> Dict[str, List]:
    """
    Detect license plates in an image using object detection.
    
    Args:
        image: Input PIL Image
        models: Dictionary containing loaded models
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Dictionary with license plate detection results
    """
    # Try different detection methods for license plates
    license_plates = {"boxes": [], "scores": [], "labels": []}
    
    # Method 1: Use Grounding DINO for license plate detection
    if models.get('grounding_dino') is not None:
        try:
            plate_detections = detect_objects_grounding_dino(
                image, 
                "license plate. number plate. car plate. vehicle registration plate.", 
                models,
                box_threshold=confidence_threshold
            )
            
            license_plates["boxes"].extend(plate_detections["boxes"])
            license_plates["scores"].extend(plate_detections["scores"])
            license_plates["labels"].extend(["license_plate"] * len(plate_detections["boxes"]))
            
        except Exception as e:
            logger.warning(f"Grounding DINO license plate detection failed: {e}")
    
    # Method 2: Use RetinaNet and filter for car-related objects (fallback)
    if not license_plates["boxes"] and models.get('retinanet') is not None:
        try:
            # Detect cars first, then look for license plates within car regions
            car_detections = detect_objects_retinanet(image, models, score_threshold=0.5)
            
            # For now, we'll use a simple heuristic to find potential license plate regions
            # This could be enhanced with a dedicated license plate detection model
            for i, box in enumerate(car_detections["boxes"]):
                if car_detections["labels"][i].lower() in ["car", "truck", "bus", "motorcycle"]:
                    # Look for rectangular regions within the car that might be license plates
                    x1, y1, x2, y2 = box
                    car_width = x2 - x1
                    car_height = y2 - y1
                    
                    # Estimate license plate location (typically bottom center of vehicle)
                    plate_width = min(car_width * 0.3, 200)  # ~30% of car width, max 200px
                    plate_height = min(car_height * 0.1, 60)  # ~10% of car height, max 60px
                    
                    plate_x1 = x1 + (car_width - plate_width) // 2
                    plate_y1 = y2 - car_height * 0.2  # 20% from bottom
                    plate_x2 = plate_x1 + plate_width
                    plate_y2 = plate_y1 + plate_height
                    
                    # Add estimated license plate region
                    license_plates["boxes"].append([int(plate_x1), int(plate_y1), int(plate_x2), int(plate_y2)])
                    license_plates["scores"].append(car_detections["scores"][i] * 0.7)  # Lower confidence for estimate
                    license_plates["labels"].append("license_plate_estimate")
                    
        except Exception as e:
            logger.warning(f"Car-based license plate estimation failed: {e}")
    
    logger.info(f"Detected {len(license_plates['boxes'])} potential license plates")
    
    # Keep all detections for debugging - don't filter here
    # OCR-based selection will happen in visualization
    logger.info(f"Keeping all {len(license_plates['boxes'])} detections for OCR analysis and debugging")
    return license_plates

def read_license_plate_text(image: Image.Image, bbox: List[int], ocr_engine: str = "easyocr") -> Dict[str, Any]:
    """
    Read text from a license plate region using OCR.
    
    Args:
        image: Input PIL Image
        bbox: Bounding box [x1, y1, x2, y2] of license plate
        ocr_engine: OCR engine to use ("easyocr" or "tesseract")
        
    Returns:
        Dictionary with OCR results
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop license plate region
    plate_crop = image.crop((x1, y1, x2, y2))
    
    # Enhance image for better OCR
    plate_enhanced = enhance_image_for_ocr(plate_crop)
    
    ocr_result = {
        "text": "",
        "confidence": 0.0,
        "engine": ocr_engine,
        "bbox": bbox,
        "enhanced_image": plate_enhanced
    }
    
    try:
        if ocr_engine == "easyocr":
            try:
                import easyocr
                # Initialize EasyOCR reader for license plates
                reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
                
                # Convert PIL to numpy array for EasyOCR
                img_array = np.array(plate_enhanced)
                
                # Read text with focus on alphanumeric characters
                results = reader.readtext(img_array, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
                
                if results:
                    # Get the result with highest confidence
                    best_result = max(results, key=lambda x: x[2])  # x[2] is confidence
                    ocr_result["text"] = best_result[1].strip().upper()
                    ocr_result["confidence"] = best_result[2]
                    
            except ImportError:
                logger.warning("EasyOCR not available, falling back to Tesseract")
                ocr_result = read_license_plate_text(image, bbox, "tesseract")
                
        elif ocr_engine == "tesseract":
            try:
                import pytesseract
                
                # Convert PIL to format suitable for Tesseract
                img_array = np.array(plate_enhanced)
                
                # Configure Tesseract for license plates
                config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                
                # Extract text
                text = pytesseract.image_to_string(img_array, config=config).strip().upper()
                
                # Get confidence (if available)
                try:
                    data = pytesseract.image_to_data(img_array, config=config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
                    
                    ocr_result["text"] = text
                    ocr_result["confidence"] = avg_confidence
                    
                except Exception as e:
                    ocr_result["text"] = text
                    ocr_result["confidence"] = 0.5  # Default confidence
                    logger.warning(f"Could not get Tesseract confidence: {e}")
                    
            except ImportError:
                logger.error("Neither EasyOCR nor Tesseract available for license plate OCR")
                ocr_result["text"] = "OCR_NOT_AVAILABLE"
                ocr_result["confidence"] = 0.0
                
    except Exception as e:
        logger.error(f"License plate OCR failed: {e}")
        ocr_result["text"] = "OCR_FAILED"
        ocr_result["confidence"] = 0.0
    
    # Clean up the text (remove common OCR artifacts)
    if ocr_result["text"]:
        ocr_result["text"] = clean_license_plate_text(ocr_result["text"])
    
    return ocr_result

def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Enhance license plate image for better OCR accuracy.
    
    Args:
        image: Cropped license plate image
        
    Returns:
        Enhanced image optimized for OCR
    """
    # Convert to grayscale for processing
    if image.mode != 'L':
        gray = image.convert('L')
    else:
        gray = image.copy()
    
    # Convert to numpy array
    img_array = np.array(gray)
    
    # Resize if too small (OCR works better on larger images)
    height, width = img_array.shape
    if width < 200 or height < 50:
        scale_factor = max(200 / width, 50 / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Apply histogram equalization
    img_array = cv2.equalizeHist(img_array)
    
    # Apply Gaussian blur and sharpen
    blurred = cv2.GaussianBlur(img_array, (0, 0), 1.0)
    sharpened = cv2.addWeighted(img_array, 1.5, blurred, -0.5, 0)
    
    # Apply morphological operations to clean up text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to PIL Image
    enhanced = Image.fromarray(cleaned)
    
    # Convert back to RGB for consistency
    enhanced = enhanced.convert('RGB')
    
    return enhanced

def clean_license_plate_text(text: str) -> str:
    """
    Clean OCR text by removing common artifacts and standardizing format.
    
    Args:
        text: Raw OCR text
        
    Returns:
        Cleaned license plate text
    """
    if not text:
        return ""
    
    # Remove common OCR artifacts
    cleaned = text.upper().strip()
    
    # Remove spaces within the text (license plates typically don't have internal spaces)
    cleaned = cleaned.replace(" ", "")
    
    # Remove non-alphanumeric characters except hyphens
    import re
    cleaned = re.sub(r'[^A-Z0-9-]', '', cleaned)
    
    # Handle common OCR mistakes
    replacements = {
        'O': '0',  # O often mistaken for 0
        'I': '1',  # I often mistaken for 1
        'S': '5',  # S sometimes mistaken for 5
        'Z': '2',  # Z sometimes mistaken for 2
    }
    
    # Apply replacements cautiously (only if the context suggests it's a number)
    # This is a simple heuristic - more sophisticated logic could be added
    for old, new in replacements.items():
        if len(cleaned) > 0 and cleaned.count(old) == 1:
            # Only replace if surrounded by numbers or at specific positions
            pos = cleaned.find(old)
            if pos > 0 and cleaned[pos-1].isdigit() or pos < len(cleaned)-1 and cleaned[pos+1].isdigit():
                cleaned = cleaned.replace(old, new, 1)
    
    return cleaned

def visualize_license_plates(image: Image.Image, detections: Dict[str, List], ocr_results: List[Dict] = None) -> Image.Image:
    """
    Visualize detected license plates with OCR results.
    Shows only the detection with the highest OCR confidence for cleanest visualization.
    
    Args:
        image: Original image
        detections: License plate detection results
        ocr_results: Optional OCR results for each detection
        
    Returns:
        Image with single best license plate visualization
    """
    vis_image = image.copy()
    draw = ImageDraw.Draw(vis_image)
    
    # If no detections, return original image
    if not detections["boxes"]:
        return vis_image
    
    # Find the detection with highest DETECTION confidence for the box
    # And separately find the best OCR results for labeling
    box_idx = 0  # Use highest detection confidence for box coordinates
    best_ocr_text = ""
    best_ocr_conf = 0.0
    longest_text = ""
    longest_ocr_conf = 0.0
    
    # Find highest detection confidence for reliable box coordinates
    if detections["scores"]:
        box_idx = detections["scores"].index(max(detections["scores"]))
        logger.info(f"Using detection index {box_idx} for box coordinates (Det conf: {detections['scores'][box_idx]:.3f})")
    
    # Find best OCR results from all detections for labeling
    all_ocr_results = []  # Store all valid OCR results for analysis
    
    if ocr_results and len(ocr_results) > 0:
        for i, ocr_data in enumerate(ocr_results):
            if i < len(detections["boxes"]) and isinstance(ocr_data, dict) and "confidence" in ocr_data:
                ocr_conf = ocr_data["confidence"]
                ocr_text = ocr_data.get("text", "")
                
                # Only consider valid OCR results
                if ocr_text and ocr_text not in ["OCR_FAILED", "OCR_NOT_AVAILABLE"]:
                    all_ocr_results.append({
                        'text': ocr_text,
                        'confidence': ocr_conf,
                        'length': len(ocr_text),
                        'index': i
                    })
                    
                    # Track highest OCR confidence
                    if ocr_conf > best_ocr_conf:
                        best_ocr_conf = ocr_conf
                        best_ocr_text = ocr_text
                        logger.info(f"Best OCR at index {i}: '{ocr_text}' with confidence {ocr_conf:.4f}")
                    
                    # Track longest text
                    if len(ocr_text) > len(longest_text):
                        longest_text = ocr_text
                        longest_ocr_conf = ocr_conf
                        logger.info(f"Longest text at index {i}: '{ocr_text}' (length: {len(ocr_text)}, conf: {ocr_conf:.4f})")
    
    # Find alternative backup: prioritize longest text that's different from primary
    backup_text = ""
    backup_conf = 0.0
    
    if all_ocr_results and len(all_ocr_results) > 1:
        # Sort by length descending, then by confidence descending
        sorted_by_length = sorted(all_ocr_results, key=lambda x: (x['length'], x['confidence']), reverse=True)
        
        # First, try to find the longest text that's different from primary
        for result in sorted_by_length:
            if result['text'] != best_ocr_text:
                backup_text = result['text']
                backup_conf = result['confidence']
                logger.info(f"Selected backup: '{backup_text}' (length: {len(backup_text)}, conf: {backup_conf:.4f})")
                break  # Take the first (longest and highest confidence) alternative
    
    logger.info(f"Final: Box from detection {box_idx}, Best OCR: '{best_ocr_text}' ({best_ocr_conf:.4f}), Longest: '{longest_text}' ({longest_ocr_conf:.4f})")
    
    # Draw the detection box using the most reliable detection coordinates
    if box_idx < len(detections["boxes"]):
        box = detections["boxes"][box_idx]
        score = detections["scores"][box_idx]
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box with color based on detection confidence
        color = "lime" if score > 0.5 else "yellow" if score > 0.3 else "orange"
        width = 3 if score > 0.5 else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        # Create primary label with best OCR text (regardless of which detection it came from)
        primary_label = f"License Plate (Det: {score:.2f})"
        backup_label = None
        
        if best_ocr_text:
            primary_label = f"{best_ocr_text} ({best_ocr_conf:.3f})"
        
        # Create backup label using the smart alternative logic
        if backup_text and backup_text != best_ocr_text:
            backup_label = f"{backup_text} ({backup_conf:.3f})"
        
        # Calculate dynamic label dimensions based on detection box size
        box_height = y2 - y1
        box_width = x2 - x1

        # Label box height = 70% of detection box height, minimum 20px, maximum 500px
        label_height = max(20, min(500, int(box_height * 0.7)))
        
        # Calculate dynamic font size based on label height (roughly 60% of label height)
        font_size = max(12, min(120, int(label_height * 0.6)))
        
        # Try to load a font with dynamic size, fallback to default if not available
        try:
            from PIL import ImageFont
            try:
                # Try to load a TrueType font with the calculated size
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except (OSError, IOError):
                logger.warning("Custom font not found, falling back to default.")
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
        except ImportError:
            font = None
        
        # Calculate label position (above the detection box)
        total_labels = 1 if not backup_label else 2
        single_label_height = max(20, min(500, int(box_height * 0.7)))
        total_label_height = single_label_height * total_labels + (2 if backup_label else 0)  # 2px gap between labels
        
        primary_label_y = y1 - total_label_height - 2  # Start above the detection box
        backup_label_y = primary_label_y + single_label_height + 2 if backup_label else 0
        
        # Draw primary label background with dynamic height
        try:
            # Try to get text size with the selected font
            if font:
                bbox = draw.textbbox((x1, primary_label_y), primary_label, font=font)
                primary_width = bbox[2] - bbox[0] + 8  # Add 8px padding
            else:
                bbox = draw.textbbox((x1, primary_label_y), primary_label)
                primary_width = bbox[2] - bbox[0] + 8  # Add 8px padding
            
            # Ensure label doesn't exceed detection box width too much
            primary_width = min(primary_width, box_width * 1.5)
            
            # Draw primary background rectangle
            draw.rectangle([x1, primary_label_y, x1 + primary_width, primary_label_y + single_label_height], 
                         fill="black", outline=color, width=2)
            
            # Draw backup label if exists
            if backup_label:
                if font:
                    backup_bbox = draw.textbbox((x1, backup_label_y), backup_label, font=font)
                    backup_width = backup_bbox[2] - backup_bbox[0] + 8
                else:
                    backup_bbox = draw.textbbox((x1, backup_label_y), backup_label)
                    backup_width = backup_bbox[2] - backup_bbox[0] + 8
                
                backup_width = min(backup_width, box_width * 1.5)
                
                # Draw backup background rectangle with different color
                draw.rectangle([x1, backup_label_y, x1 + backup_width, backup_label_y + single_label_height], 
                             fill="darkgray", outline="orange", width=2)
                
        except:
            # Fallback for older PIL versions
            logger.warning("Falling back to default rectangle drawing.")
            fallback_width = min(int(box_width * 1.2), 250)
            draw.rectangle([x1, primary_label_y, x1 + fallback_width, primary_label_y + single_label_height], 
                         fill="black", outline=color, width=2)
            
            if backup_label:
                draw.rectangle([x1, backup_label_y, x1 + fallback_width, backup_label_y + single_label_height], 
                             fill="darkgray", outline="orange", width=2)
        
        # Draw the text labels with dynamic positioning
        primary_text_y = primary_label_y + (single_label_height - font_size) // 2  # Center text vertically
        if font:
            draw.text((x1 + 4, primary_text_y), primary_label, fill="white", font=font)  # 4px left padding
        else:
            draw.text((x1 + 4, primary_text_y), primary_label, fill="white")  # 4px left padding
        
        # Draw backup label text if exists
        if backup_label:
            backup_text_y = backup_label_y + (single_label_height - font_size) // 2
            if font:
                draw.text((x1 + 4, backup_text_y), backup_label, fill="white", font=font)
            else:
                draw.text((x1 + 4, backup_text_y), backup_label, fill="white")
    
    return vis_image

# -----------------------------------------------------------------------------
# Image Processing Functions

def remove_background(image: Image.Image, models: Dict) -> Image.Image:
    """Remove background using rembg."""
    if models.get('rembg') is None:
        raise RuntimeError("rembg model not available")
    
    session = models['rembg']
    return remove(image, session=session)

def smooth_background_removal(
    image: Image.Image, 
    models: Dict,
    smooth_edges: bool = True,
    feather_amount: int = 10,
    edge_refinement: bool = True
) -> Image.Image:
    """
    Remove background with smooth edge processing.
    
    Args:
        image: Input image
        models: Model dictionary containing rembg
        smooth_edges: Apply edge smoothing
        feather_amount: Amount of edge feathering (1-50)
        edge_refinement: Apply edge refinement algorithms
    """
    # First, do standard background removal
    bg_removed = remove_background(image, models)
    
    if not smooth_edges:
        return bg_removed
    
    # Convert to RGBA if not already
    if bg_removed.mode != 'RGBA':
        bg_removed = bg_removed.convert('RGBA')
    
    # Get the alpha channel for processing
    alpha = np.array(bg_removed.split()[-1])
    
    if edge_refinement:
        # Apply morphological operations to clean up the mask
        kernel_size = max(3, feather_amount)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Close small gaps and holes
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        alpha = cv2.medianBlur(alpha, 3)
    
    if feather_amount > 0:
        # Apply Gaussian blur for smooth edges
        # Full range allowed for maximum quality (may take longer for high values)
        blur_kernel = feather_amount * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (blur_kernel, blur_kernel), 0)
    
    # Create result image with smoothed alpha
    result = np.array(bg_removed)
    result[:, :, 3] = alpha
    
    return Image.fromarray(result)

def add_drop_shadow(
    image: Image.Image, 
    shadow_offset: Tuple[int, int] = (5, 5),
    shadow_blur: int = 10,
    shadow_opacity: float = 0.5,
    shadow_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """Add drop shadow effect to image."""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Create shadow layer
    shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    
    # Get alpha channel for shadow shape
    alpha = image.split()[-1]
    
    # Create shadow with offset
    shadow_img = Image.new('RGBA', image.size, (0, 0, 0, 0))
    shadow_colored = Image.new('RGBA', image.size, shadow_color + (int(255 * shadow_opacity),))
    shadow_img.paste(shadow_colored, shadow_offset, alpha)
    
    # Blur the shadow
    if shadow_blur > 0:
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
    
    # Composite shadow with original image
    result = Image.new('RGBA', image.size, (0, 0, 0, 0))
    result = Image.alpha_composite(result, shadow_img)
    result = Image.alpha_composite(result, image)
    
    return result

def add_smart_shadow_effect(cutout_image: Image.Image) -> Image.Image:
    """Add realistic drop shadow to cutout image using smart algorithm."""
    output = np.array(cutout_image)

    if output.shape[2] != 4:
        # If no alpha channel, create one
        alpha = np.ones((output.shape[0], output.shape[1], 1), dtype=np.uint8) * 255
        output = np.concatenate([output, alpha], axis=2)
    
    mask = output[:, :, 3]
    mask = (mask > 128).astype(np.uint8) * 255
    
    # Create shadow by rolling the mask
    roll_amount = int(max(cutout_image.size[0], cutout_image.size[1]) * 0.01)
    rolled_mask = np.roll(mask, roll_amount, axis=0)
    shadow_strip = rolled_mask.copy()
    shadow_strip[mask > 0] = 0
    
    # Remove small artifacts
    contours, _ = cv2.findContours(shadow_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 0.005 * shadow_strip.size:
            cv2.drawContours(shadow_strip, [contour], -1, 0, cv2.FILLED)
    
    # Dilate and create convex hull
    kernel_size = int(max(cutout_image.size[0], cutout_image.size[1]) * 0.003)
    if kernel_size > 0:
        dilated_strip = cv2.dilate(shadow_strip, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    else:
        dilated_strip = shadow_strip
    
    contours, _ = cv2.findContours(dilated_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    strip_hull = np.zeros_like(dilated_strip)
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.drawContours(strip_hull, [hull], -1, 255, cv2.FILLED)
    
    # Create shadow
    shadow = np.zeros_like(output)
    shadow[strip_hull > 0, :3] = 96  # Gray shadow
    shadow[strip_hull > 0, 3] = 255
    
    # Blur the shadow
    sigma = int(max(cutout_image.size[0], cutout_image.size[1]) * 0.1)
    if sigma % 2 == 0:
        sigma = sigma + 1
    if sigma > 0:
        blurred_shadow = cv2.GaussianBlur(shadow, (sigma, sigma), 0)
    else:
        blurred_shadow = shadow
    
    # Compose final image
    final_image = np.copy(output)
    alpha_car = final_image[:, :, 3] / 255.0
    alpha_shadow = 1.0 - alpha_car
    
    for c in range(3):
        final_image[:, :, c] = (alpha_car * final_image[:, :, c] + alpha_shadow * blurred_shadow[:, :, c])
    
    final_image[:, :, 3] = np.maximum(final_image[:, :, 3], blurred_shadow[:, :, 3])
    
    return Image.fromarray(final_image)

def extract_canny_edges(image: Image.Image, sigma: float = 0.33) -> np.ndarray:
    """Extract Canny edges from image."""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Automatic threshold detection
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(gray, lower, upper)
    
    return edges

def prepare_control_image(
    image: Image.Image, 
    control_method: str = "canny_edges",
    control_method_params: Dict = None,
    target_size: Tuple[int, int] = (1024, 1024),
    models: Optional[Dict] = None
) -> Image.Image:
    """
    Prepare control image for SDXL generation.
    
    Args:
        image: Input image
        control_method: Control method to use:
            - "canny_edges": Use Canny edge detection
            - "mask_background_removal": Generate mask by removing background
            - "mask_foreground_objects": Generate mask from detected foreground objects  
            - "mask_largest_object": Generate mask for the largest detected object
            - "control_img_upload": Use control image from provided image
        control_method_params: Parameters for the selected control method:
        target_size: Target image dimensions
        models: Model dictionary for automatic mask generation
    """
    control_method_params = control_method_params or {}
    mask = control_method_params.get('pre_computed_mask')
    
    # Generate control image based on method
    if control_method == "canny_edges":
        # Use Canny edge detection
        sigma = control_method_params.get('sigma', 0.33)
        edges = extract_canny_edges(image, sigma)
        
        # Apply mask to edges if provided
        if mask is not None:
            edges[~mask] = 0
        
        control_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    elif control_method == "control_img_upload":
        # Handle manual control image upload
        control_img_source = control_method_params.get('control_img_source')
        if control_img_source is not None:
            try:
                # Resize manual mask to match source image size
                manual_mask_resized = control_img_source.resize(image.size, Image.Resampling.LANCZOS)
                # Convert to grayscale
                mask_gray = manual_mask_resized.convert('L')
                mask_array = np.array(mask_gray)
                
                # Detect if this is likely a Canny edge map (white edges on black background)
                # Check if most pixels are dark and there are bright edge-like structures
                dark_ratio = np.sum(mask_array < 50) / mask_array.size
                bright_ratio = np.sum(mask_array > 200) / mask_array.size
                
                if dark_ratio > 0.7 and bright_ratio < 0.1:
                    # This looks like a Canny edge map - use it directly as control image
                    print("Detected Canny edge map in manual mask - using as edge control instead of mask")
                    control_img = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2RGB)
                    control_image = Image.fromarray(control_img)
                    # Skip the normal mask processing and return early
                    return _resize_control_image(control_image, target_size)
                else:
                    # Normal mask: Bright areas = True (apply style), Dark areas = False (no change)
                    mask = mask_array > 128
                    # Create mask-based control image
                    control_img = np.zeros((*image.size[::-1], 3), dtype=np.uint8)  # Black background
                    control_img[mask] = 255  # White regions where mask is True
            except Exception as e:
                print(f"Manual mask processing failed: {e}")
                # Fallback: create a full white mask (apply style everywhere)
                control_img = np.full((*image.size[::-1], 3), 255, dtype=np.uint8)
        else:
            # No control image provided, create a full white mask
            print("Warning: No control image provided, using full image mask")
            control_img = np.full((*image.size[::-1], 3), 255, dtype=np.uint8)
    
    elif control_method.startswith("mask_"):
        # Generate mask if not provided
        if mask is None and models is not None:
            
            if control_method == "mask_background_removal":
                # Use rembg to create foreground mask
                if models.get('rembg') is not None:
                    try:
                        bg_removed = remove_background(image, models)
                        if bg_removed.mode == 'RGBA':
                            # Extract alpha channel as mask
                            alpha = np.array(bg_removed.split()[-1])
                            mask = alpha > 128  # Convert to boolean mask
                    except Exception as e:
                        print(f"Auto mask generation failed: {e}")
            
            elif control_method in ["mask_foreground_objects", "mask_largest_object"]:
                # Use object detection + SAM2 for segmentation
                try:
                    # Detect objects using RetinaNet (prefer general objects)
                    if models.get('retinanet') is not None:
                        detections = detect_objects_retinanet(image, models, score_threshold=0.3)
                        
                        if detections["boxes"]:
                            masks = []
                            
                            if control_method == "mask_largest_object":
                                # Find largest bounding box
                                largest_idx = 0
                                largest_area = 0
                                for i, box in enumerate(detections["boxes"]):
                                    area = (box[2] - box[0]) * (box[3] - box[1])
                                    if area > largest_area:
                                        largest_area = area
                                        largest_idx = i
                                
                                # Segment only the largest object
                                if models.get('sam2') is not None:
                                    obj_mask = segment_object(image, detections["boxes"][largest_idx], models)
                                    masks.append(obj_mask)
                            
                            else:  # mask_foreground_objects
                                # Segment all detected objects
                                if models.get('sam2') is not None:
                                    for box in detections["boxes"][:5]:  # Limit to top 5 objects
                                        obj_mask = segment_object(image, box, models)
                                        masks.append(obj_mask)
                            
                            # Combine all masks
                            if masks:
                                mask = np.logical_or.reduce(masks)
                
                except Exception as e:
                    print(f"Auto mask generation failed: {e}")
        
        # Create mask-based control image
        if mask is not None:
            # Convert boolean mask to proper mask format for ControlNet
            # Create a black background with white regions where the mask is True
            control_img = np.zeros((*image.size[::-1], 3), dtype=np.uint8)  # Black background
            control_img[mask] = 255  # White regions where mask is True
        else:
            # Fallback: create a full white mask (apply style everywhere)
            print("Warning: No mask generated, using full image mask")
            control_img = np.full((*image.size[::-1], 3), 255, dtype=np.uint8)
    
    else:
        raise ValueError(f"Unknown control method: {control_method}")
    
    control_image = Image.fromarray(control_img)
    
    # Resize to optimal size using helper function
    return _resize_control_image(control_image, target_size)

def _resize_control_image(control_image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Helper function to resize control image to target dimensions."""
    # Get original dimensions
    orig_width, orig_height = control_image.size
    
    # Calculate optimal dimensions maintaining aspect ratio
    # Scale to fit within target area
    target_area = target_size[0] * target_size[1]
    current_area = orig_width * orig_height
    
    if current_area > 0:
        ratio = np.sqrt(target_area / current_area)
        new_width = int(orig_width * ratio)
        new_height = int(orig_height * ratio)
        
        # Ensure dimensions are divisible by 8 (required for diffusion models)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        # Make sure we don't exceed target dimensions
        if new_width > target_size[0] or new_height > target_size[1]:
            scale_w = target_size[0] / new_width
            scale_h = target_size[1] / new_height
            scale = min(scale_w, scale_h)
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)
            # Re-align to 8 pixels
            new_width = (new_width // 8) * 8
            new_height = (new_height // 8) * 8
    else:
        new_width, new_height = target_size
    
    # Resize to optimal size
    return control_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def generate_image_from_text(
    prompt: str,
    models: Dict,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    style: str = "realistic",
    width: int = 1024,
    height: int = 1024,
    inference_steps: int = 20,
    guidance_scale: float = 7.5
) -> Dict[str, Any]:
    """Generate image from text prompt using SDXL."""
    if models.get('sdxl_txt2img') is None:
        raise RuntimeError("SDXL text-to-image model not available")
    
    pipeline = models['sdxl_txt2img']
    
    # Combine style with prompt
    if style == "cartoon":
        full_prompt = f"{CARTOON_PROMPT}, {prompt}"
    else:
        full_prompt = f"{REALISTIC_PROMPT}, {prompt}"
    
    # Collect all parameters for debugging
    debug_params = {
        'style': style,
        'guidance_scale': guidance_scale,
        'inference_steps': inference_steps,
        'image_size': f"{width}x{height}",
        'prompt': prompt,
        'full_prompt': full_prompt,
        'negative_prompt': negative_prompt,
    }
    
    with torch.no_grad():
        result = pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
        )
    
    return {
        'image': result.images[0],
        'debug_params': debug_params
    }

def generate_image_controlled(
    control_image: Image.Image,
    prompt: str,
    models: Dict,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    style: str = "realistic",
    controlnet_scale: float = 0.6,
    inference_steps: int = 12
) -> Dict[str, Any]:
    """Generate image using ControlNet + SDXL and return result with debug info."""
    if models.get('sdxl') is None:
        raise RuntimeError("SDXL model not available")
    
    pipeline = models['sdxl']
    
    # Combine style with prompt
    if style == "cartoon":
        full_prompt = f"{CARTOON_PROMPT}, {prompt}"
    else:
        full_prompt = f"{REALISTIC_PROMPT}, {prompt}"
    
    # Collect all parameters for debugging
    debug_params = {
        'style': style,
        'controlnet_scale': controlnet_scale,
        'inference_steps': inference_steps,        
        'control_image_size': control_image.size,
        'prompt': prompt,
        'full_prompt': full_prompt,
        'negative_prompt': negative_prompt,
    }
    
    with torch.no_grad():
        result = pipeline(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            controlnet_conditioning_scale=controlnet_scale,
            width=control_image.width,
            height=control_image.height,
            num_inference_steps=inference_steps,
            num_images_per_prompt=1,
        )
    
    return {
        'image': result.images[0],
        'debug_params': debug_params
    }

def generate_ai_inpainting(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    models: Dict,
    negative_prompt: str = "blurry, low quality, distorted, deformed, artifacts",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    strength: float = 1.0
) -> Dict[str, Any]:
    """
    Perform AI-powered inpainting using Stable Diffusion XL Inpainting.
    
    Args:
        image: Input image to inpaint
        mask: Binary mask (white = inpaint area, black = keep original)
        prompt: Text prompt describing what to generate in masked areas
        models: Dictionary containing loaded models
        negative_prompt: What to avoid in generation
        num_inference_steps: Number of denoising steps
        guidance_scale: How closely to follow the prompt
        strength: How much to change the masked area (0.0 = no change, 1.0 = full generation)
    
    Returns:
        Dictionary with inpainted image and debug parameters
    """
    if models.get('sdxl_inpaint') is None:
        raise RuntimeError("SDXL Inpainting model not available")
    
    pipeline = models['sdxl_inpaint']
    
    # Ensure image and mask are the same size
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    
    # Convert mask to RGB if needed (some pipelines expect RGB masks)
    if mask.mode != 'RGB':
        mask = mask.convert('RGB')
    
    # Ensure image is RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Collect parameters for debugging
    debug_params = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'strength': strength,
        'image_size': image.size,
        'mask_size': mask.size,
    }
    
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            width=image.width,
            height=image.height,
        )
    
    return {
        'image': result.images[0],
        'debug_params': debug_params
    }

def generate_ai_outpainting(
    image: Image.Image,
    prompt: str,
    models: Dict,
    extend_top: int = 0,
    extend_bottom: int = 0,
    extend_left: int = 0,
    extend_right: int = 0,
    outpaint_method: str = "inpaint",
    negative_prompt: str = "blurry, low quality, distorted, deformed, artifacts",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    strength: float = 1.0
) -> Dict[str, Any]:
    """
    Perform AI-powered outpainting by extending image canvas and generating content in new areas.
    
    Args:
        image: Input image to outpaint
        prompt: Text prompt describing what to generate in extended areas
        models: Dictionary containing loaded models
        extend_top: Pixels to extend upward
        extend_bottom: Pixels to extend downward
        extend_left: Pixels to extend leftward
        extend_right: Pixels to extend rightward
        outpaint_method: Method for outpainting ('inpaint' or 'controlnet')
        negative_prompt: What to avoid in generation
        num_inference_steps: Number of denoising steps
        guidance_scale: How closely to follow the prompt
        strength: How much to change the extended areas (0.0 = no change, 1.0 = full generation)
    
    Returns:
        Dictionary with outpainted image, extended canvas, mask, and debug parameters
    """
    if models.get('sdxl_inpaint') is None:
        raise RuntimeError("SDXL Inpainting model not available")
    
    # Validate outpaint method
    if outpaint_method not in ['inpaint', 'controlnet']:
        logger.warning(f"Unknown outpaint method '{outpaint_method}', defaulting to 'inpaint'")
        outpaint_method = 'inpaint'
    
    # Check if controlnet method is requested but not available
    if outpaint_method == 'controlnet':
        logger.info("ControlNet outpainting method requested - using inpaint method (controlnet coming soon)")
        outpaint_method = 'inpaint'  # Fallback for now
    
    # Validate that we have extensions to perform outpainting
    if extend_top <= 0 and extend_bottom <= 0 and extend_left <= 0 and extend_right <= 0:
        logger.warning("All extension values are 0 or negative - no outpainting will occur!")
        logger.info(f"Extensions: top={extend_top}, bottom={extend_bottom}, left={extend_left}, right={extend_right}")
        return {
            'image': image.copy(),  # Return original image unchanged
            'extended_canvas': image.copy(),
            'mask': Image.new('RGB', image.size, (0, 0, 0)),  # All black mask (no generation)
            'debug_params': {
                'error': 'No valid extensions specified',
                'extensions': {'top': extend_top, 'bottom': extend_bottom, 'left': extend_left, 'right': extend_right}
            }
        }
    
    # Ensure image is RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Calculate new canvas dimensions
    original_width, original_height = image.size
    new_width = original_width + extend_left + extend_right
    new_height = original_height + extend_top + extend_bottom
    
    # # Create extended canvas with black background (will be inpainted)
    # extended_canvas = Image.new('RGB', (new_width, new_height), (0, 0, 0))
    # Create extended canvas with WHITE background (HuggingFace recommended approach)
    extended_canvas = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    # Paste original image onto extended canvas
    paste_x = extend_left
    paste_y = extend_top
    extended_canvas.paste(image, (paste_x, paste_y))
    
    # # Create mask for outpainting areas (white = generate, black = keep original)
    # mask = Image.new('RGB', (new_width, new_height), (255, 255, 255))  # Start with all white
    # Using grayscale L mode as required by SDXL inpainting pipeline
    mask = Image.new('L', (new_width, new_height), 255)  # Start with all white (generate)

    # Create black region where original image is placed (keep original content)
    # mask_black_region = Image.new('RGB', image.size, (0, 0, 0))
    mask_black_region = Image.new('L', image.size, 0)  # Black = keep original
    mask.paste(mask_black_region, (paste_x, paste_y))
    
    # Debug: Verify mask is created correctly
    # mask_gray = mask.convert('L')
    # mask_array_debug = np.array(mask_gray)
    mask_array_debug = np.array(mask)
    white_pixels = np.sum(mask_array_debug == 255)
    black_pixels = np.sum(mask_array_debug == 0)
    total_pixels = mask_array_debug.size

    logger.info(f"Outpainting mask created: {new_width}x{new_height}")
    logger.info(f"Original image pasted at: ({paste_x}, {paste_y})")
    logger.info(f"Mask pixels - White (generate): {white_pixels}, Black (keep): {black_pixels}, Total: {total_pixels}")
    logger.info(f"Mask coverage - Generate: {white_pixels/total_pixels*100:.1f}%, Keep: {black_pixels/total_pixels*100:.1f}%")

    # Validation: Ensure we have areas to generate (white pixels)
    if white_pixels == 0:
        logger.warning("No white pixels in mask - nothing will be generated!")
        return {
            'image': extended_canvas,  # Return original canvas if no generation areas
            'extended_canvas': extended_canvas,
            'mask': mask,
            'debug_params': debug_params,
            'error': 'No generation areas in mask'
        }
    
    # Optional: Add feathering/blending at the borders for smoother transitions
    if extend_top > 0 or extend_bottom > 0 or extend_left > 0 or extend_right > 0:
        # Convert mask to array for feathering
        # mask_array = np.array(mask.convert('L'))
        mask_array = np.array(mask)
        
        # Apply slight gaussian blur to soften mask edges for better blending
        # feather_size = min(10, min(extend_top, extend_bottom, extend_left, extend_right) // 4) if any([extend_top, extend_bottom, extend_left, extend_right]) else 0
        # Only consider non-zero extensions for feather size calculation
        non_zero_extensions = [ext for ext in [extend_top, extend_bottom, extend_left, extend_right] if ext > 0]
        feather_size = min(10, min(non_zero_extensions) // 4) if non_zero_extensions else 0
        if feather_size > 1:
            mask_array = cv2.GaussianBlur(mask_array, (feather_size*2+1, feather_size*2+1), 0)
        
        # # Convert back to RGB mask
        # mask = Image.fromarray(cv2.cvtColor(mask_array, cv2.COLOR_GRAY2RGB))
        # Convert back to grayscale mask (L mode)
        mask = Image.fromarray(mask_array, mode='L')

        # Debug: Print mask statistics for troubleshooting
        mask_stats = {
            'mask_size': mask.size,
            'feather_size': feather_size,
            'non_zero_extensions': non_zero_extensions,
            # 'mask_unique_values': np.unique(np.array(mask.convert('L')))[:10]  # First 10 unique values
            'mask_unique_values': np.unique(np.array(mask))[:10]  # First 10 unique values
        }
        logger.info(f"Outpainting mask debug: {mask_stats}")
    
    # Collect parameters for debugging
    debug_params = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'strength': strength,
        'original_size': image.size,
        'extended_size': (new_width, new_height),
        'extensions': {
            'top': extend_top,
            'bottom': extend_bottom,
            'left': extend_left,
            'right': extend_right
        },
        'paste_position': (paste_x, paste_y)
    }
    
    # Generate outpainted content using inpainting pipeline
    pipeline = models['sdxl_inpaint']
    
    # Final debug check before pipeline call
    logger.info(f"Final outpainting setup:")
    logger.info(f"  - Extended canvas size: {extended_canvas.size}")
    logger.info(f"  - Mask size: {mask.size}")
    logger.info(f"  - Mask mode: {mask.mode}")
    logger.info(f"  - Pipeline params: steps={num_inference_steps}, guidance={guidance_scale}, strength={strength}")

    with torch.no_grad():
        # Generate the outpainted areas using SDXL inpainting
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=extended_canvas,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            width=new_width,
            height=new_height,
        )
        
        # Get the generated result
        generated_image = result.images[0]
        
        # CRITICAL: Preserve original image by pasting it back onto the generated result
        # This ensures the original image area is exactly preserved, only extended areas are generated
        final_result = generated_image.copy()
        final_result.paste(image, (paste_x, paste_y))
        
        logger.info("✅ Outpainting completed - original image preserved, extended areas generated")
    
    return {
        # 'image': result.images[0],
        'image': final_result,  # Final result with original preserved
        'extended_canvas': extended_canvas,  # Canvas before generation
        'mask': mask,  # Mask used for generation
        'debug_params': debug_params,
        'generated_raw': generated_image  # Raw generation result before preservation
    }

def rotate_image(
    image: Image.Image,
    angle: float,
    expand: bool = True,
    fill_color: tuple = (255, 255, 255, 0),
    background_type: str = "transparent"
) -> Image.Image:
    """
    Rotate image by specified angle.
    
    Args:
        image: Input image to rotate
        angle: Rotation angle in degrees (positive = clockwise)
        expand: Whether to expand canvas to fit rotated image
        fill_color: Background fill color for transparent areas (RGBA)
        background_type: "transparent" or "solid_color"
    
    Returns:
        Rotated PIL Image
    """
    # Handle zero angle case
    if angle == 0:
        return image.copy()
    
    # Ensure image has alpha channel for proper rotation handling
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # PIL's rotate method rotates counter-clockwise by default
    # Since we want positive angles to be clockwise (as expected by users),
    # we need to negate the angle
    pil_angle = -angle
    
    # Rotate image with specified fill color
    rotated = image.rotate(pil_angle, expand=expand, fillcolor=fill_color)
    
    # If user wants solid background and we have transparency, composite properly
    if background_type == "solid_color" and fill_color[3] > 0:
        # Create a background with the solid color
        background = Image.new('RGBA', rotated.size, fill_color)
        # Composite the rotated image over the solid background
        final_image = Image.alpha_composite(background, rotated)
        return final_image
    
    return rotated

def flip_image(
    image: Image.Image,
    direction: str = 'horizontal'
) -> Image.Image:
    """
    Flip image horizontally or vertically.
    
    Args:
        image: Input image to flip
        direction: 'horizontal' or 'vertical'
    
    Returns:
        Flipped PIL Image
    """
    if direction == 'horizontal':
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif direction == 'vertical':
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError(f"Invalid flip direction: {direction}. Use 'horizontal' or 'vertical'.")

def crop_image(
    image: Image.Image,
    crop_box: tuple = None,
    crop_mode: str = 'custom',
    aspect_ratio: str = '1:1'
) -> Dict[str, Any]:
    """
    Crop image using various methods.
    
    Args:
        image: Input image to crop
        crop_box: (left, top, right, bottom) for custom crop
        crop_mode: 'custom', 'center', 'aspect_ratio'
        aspect_ratio: Target aspect ratio like '16:9', '4:3', '1:1' for aspect_ratio mode
    
    Returns:
        Dictionary with cropped image and crop info
    """
    width, height = image.size
    
    if crop_mode == 'custom' and crop_box:
        left, top, right, bottom = crop_box
        # Ensure crop box is within image bounds
        left = max(0, min(left, width))
        top = max(0, min(top, height))
        right = max(left, min(right, width))
        bottom = max(top, min(bottom, height))
        crop_box = (left, top, right, bottom)
        
    elif crop_mode == 'center':
        # Crop to center square
        size = min(width, height)
        left = (width - size) // 2
        top = (height - size) // 2
        crop_box = (left, top, left + size, top + size)
        
    elif crop_mode == 'aspect_ratio':
        # Crop to specific aspect ratio from center
        try:
            if ':' in aspect_ratio:
                w_ratio, h_ratio = map(float, aspect_ratio.split(':'))
            else:
                w_ratio, h_ratio = float(aspect_ratio), 1.0
                
            target_ratio = w_ratio / h_ratio
            current_ratio = width / height
            
            if current_ratio > target_ratio:
                # Image is wider - crop width
                new_width = int(height * target_ratio)
                left = (width - new_width) // 2
                crop_box = (left, 0, left + new_width, height)
            else:
                # Image is taller - crop height
                new_height = int(width / target_ratio)
                top = (height - new_height) // 2
                crop_box = (0, top, width, top + new_height)
                
        except (ValueError, ZeroDivisionError):
            # Fallback to center square
            size = min(width, height)
            left = (width - size) // 2
            top = (height - size) // 2
            crop_box = (left, top, left + size, top + size)
    else:
        # Default: no crop
        crop_box = (0, 0, width, height)
    
    cropped_image = image.crop(crop_box)
    
    crop_info = {
        'original_size': (width, height),
        'crop_box': crop_box,
        'cropped_size': cropped_image.size,
        'crop_mode': crop_mode,
        'aspect_ratio': aspect_ratio if crop_mode == 'aspect_ratio' else None
    }
    
    return {
        'image': cropped_image,
        'crop_info': crop_info
    }

def apply_post_processing(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0
) -> Image.Image:
    """Apply post-processing adjustments."""
    # Brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    
    # Contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    
    # Color saturation
    if saturation != 1.0:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(saturation)
    
    # Sharpness
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    
    return image

# -----------------------------------------------------------------------------
# Gallery and Processing System

def ensure_image_in_gallery(image: Image.Image, title: str = "Auto-Added Image", description: str = "Image automatically added to gallery") -> str:
    """
    Helper function to ensure an image is in the gallery and return its ID.
    If the image is already in the gallery (by checking image data), return existing ID.
    Otherwise, add it to the gallery and return the new ID.
    
    Args:
        image: PIL Image to add to gallery
        title: Title for the gallery entry
        description: Description for the gallery entry
        
    Returns:
        Gallery ID string
    """
    return add_to_gallery(image, title, description, {'type': 'auto_added'})

def initialize_gallery():
    """Initialize the image gallery in session state."""
    if 'gallery' not in st.session_state:
        st.session_state.gallery = {}
    if 'processing_logs' not in st.session_state:
        st.session_state.processing_logs = []

def add_to_gallery(image: Image.Image, title: str, description: str = "", metadata: Dict = None) -> str:
    """Add an image to the gallery and return its ID."""
    now = datetime.now()
    # timestamp = now.strftime("%H%M%S") + f"{now.microsecond//1000:03d}"               # HHMMSSMMM
    # display_timestamp = now.strftime("%H:%M:%S.") + f"{now.microsecond//1000:03d}"    # HH:MM:SS.MMM
    # timestamp = now.strftime("%H%M%S%f")[:-3]                   # Simple HHMMSSMMM format
    # display_timestamp = now.strftime("%H:%M:%S.%f")[:-3]        # Display format HH:MM:SS.MMM
    timestamp = now.strftime("%Y%m%dT%H%M%S%f")[:-3]              # Compact ISO 8601 with milliseconds
    display_timestamp = now.isoformat(timespec='milliseconds')

    img_id = f"img_{timestamp}"
    prefixed_title = f"[{timestamp}] {title}"
    
    st.session_state.gallery[img_id] = {
        'image': image,
        'title': prefixed_title,
        'description': description,
        'metadata': metadata or {},
        'timestamp': display_timestamp
    }
    
    return img_id

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable format."""
    if seconds is None:
        return "N/A"
    
    # ___ For format: 1:02:30.500000___
    delta = timedelta(seconds=seconds)
    return str(delta)

    # ___ For format: 1h2m30s ___
    # hours = int(seconds // 3600)
    # minutes = int((seconds % 3600) // 60)
    # secs = int(seconds % 60)
    # if hours > 0:
    #     return f"{hours}h{minutes}m{secs}s"
    # elif minutes > 0:
    #     return f"{minutes}m{secs}s"
    # else:
    #     return f"{secs}s"

def parse_duration_to_seconds(duration_str: str) -> float:
    """Parse ISO duration string directly to seconds."""
    if duration_str is None or duration_str == "":
        return 0.0

    # Handle numeric values (backward compatibility)
    if isinstance(duration_str, (int, float)):
        return float(duration_str)

    try:
        # Parse time format like "0:00:16.889484" directly
        if isinstance(duration_str, str) and ':' in duration_str:
            # ___ Approach 2: Using strptime to parse the time format ___
            time_obj = datetime.strptime(duration_str, "%H:%M:%S.%f")
            return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000
            # ___ Approach 1: Manual parsing ___
            # if len(time_parts) >= 3:
            #     hours = int(time_parts[0])
            #     minutes = int(time_parts[1]) 
            #     seconds = float(time_parts[2])
            #     return hours * 3600 + minutes * 60 + seconds
            #     # return timedelta(hours=hours, minutes=minutes, seconds=seconds).total_seconds()

        # Try parsing as float directly
        return float(duration_str)
        
    except (ValueError, TypeError):
        return 0.0

def add_processing_log(operation: str, source_id: str, result_id: str, params: Dict = None, status: str = "success", processed_duration: str = None, message: Dict = None):
    """Add a processing operation to the logs."""
    now = datetime.now()
    log_entry = {
        'timestamp': now.isoformat(),
        'operation': operation,
        'source_id': source_id,
        'result_id': result_id,
        'params': params or {},
        'status': status,
        'processed_duration': processed_duration
    }
    
    # Add message data for transparency if available
    if message is not None:
        log_entry['pipeline_message'] = message
        
        # Extract summary information from message
        if 'common' in message:
            log_entry['pipeline_status'] = message.get('status', 'unknown')
            log_entry['current_step'] = message['common'].get('current_step', 0)
            log_entry['total_steps'] = len(message['common'].get('operation_chain', []))
            log_entry['operation_chain'] = message['common'].get('operation_chain', [])
    
    st.session_state.processing_logs.append(log_entry)

def display_processing_logs(logs: List[Dict]) -> None:
    """Display a list of processing logs with consistent formatting."""
    for log in reversed(logs):
        status_icon = "✅" if log['status'] == 'success' else "❌"
        
        # Format processed time if available
        time_info = ""
        if log.get('processed_duration') is not None:
            time_info = f" - ⏱️ {log['processed_duration']}"
        
        # Add pipeline information if available
        pipeline_info = ""
        if log.get('operation_chain'):
            current_step = log.get('current_step', 0)
            total_steps = log.get('total_steps', len(log['operation_chain']))
            pipeline_info = f" - 🔗 Step {current_step}/{total_steps} in {' → '.join(log['operation_chain'])}"
        
        # Create expandable log entry
        with st.expander(f"{status_icon} `{log['timestamp']}` - {log['operation']} ({log['source_id']} → {log.get('result_id', 'failed')}){time_info}{pipeline_info}", expanded=False):
            st.write("Parameters:")
            params = log.get('params', {})
            if params:
                st.code(json.dumps(params, indent=2, default=str), language="json")
            else:
                st.write("  - No parameters")
            
            # Show processed time in the details
            if log.get('processed_duration') is not None:
                st.write(f"Processed Time: {log['processed_duration']}")
            
            # Show pipeline information
            if log.get('operation_chain'):
                st.write("Pipeline Chain:")
                st.write(" → ".join(log['operation_chain']))
                st.write(f"Pipeline Status: {log.get('pipeline_status', 'unknown')}")
                st.write(f"Current Step: {log.get('current_step', 0)}/{log.get('total_steps', 'unknown')}")
            
            # Show full pipeline message if available (for debugging)
            if log.get('pipeline_message') and st.checkbox(f"Show Full Pipeline Message", key=f"show_message_{log['timestamp']}"):
                st.write("Complete Pipeline Message:")
                st.code(json.dumps(log['pipeline_message'], indent=2, default=str), language="json")
            
            if log['status'] == 'failed':
                st.error("❌ Operation failed")
            else:
                st.success("✅ Operation completed successfully")

def process_single_operation(
    operation: str,
    source_image_id: str = None,
    models: Dict = None,
    params: Dict = None,
    message: Dict = None
) -> Dict[str, Any]:
    """
    Process a single operation on an image using standardized message format.
    
    Available operations:
    - detect_objects: Detect objects and visualize results
    - detect_license_plates: Detect license plates and read text with OCR
    - remove_background: Remove background with optional smooth edges
    - add_shadow: Add drop shadow or smart shadow effect
    - image_to_image: Generate image using ControlNet
    - create_mask: Create mask from objects or background
    - create_edge_map: Create Canny edge map
    - enhance_image: Apply post-processing adjustments
    - transform_image: Rotate, flip, or crop image
    - segment_objects: Create precise object masks from detections
    - blur_regions: Blur regions based on mask or detections
    - extract_objects: Extract objects as cutouts or crops
    - inpaint_regions: Remove/inpaint regions using various methods
    - generate_from_control: Generate image using control image
    - outpaint_image: Extend image canvas and generate content in new areas using AI
    
    Args:
        models: Dictionary of loaded AI models
        operation: Name of the operation to perform
        source_image_id: Gallery ID of source image
        params: Operation-specific parameters
        message: Standardized message containing all workflow data

    Returns:
        Dictionary with 'status', 'message', 'timestamp', and 'processed_duration'
    """
    start_time = time.time()

    # Always use standard message format - initialize if not provided.
    # None message indicates that the request is not from a pipeline (combined operations), but from an individual usage.
    if message is None:
        message = create_pipeline_message(
            source_image_id=source_image_id,
            operation_chain=[operation],
            operation_params={operation: params or {}}
        )

    try:
        # Condition: Require source image
        source_image = get_latest_image_from_message(message)
        if source_image is None:
            return update_message_with_result(
                message, operation, {
                    'status': 'ERROR',
                    'status_msg': 'No image found',
                    'processed_duration_seconds': time.time() - start_time
                }
            )
        
        # Get operation-specific parameters
        params = message.get(operation, {}).get('input_params', {})

        # ___ Operation: Object Detection ___
        if operation == "detect_objects":
            method = params.get('method', 'retinanet')
            query = params.get('text_query', 'person. human. car. vehicle.')
            detections = perform_object_detection(source_image, method, models, query)
            
            vis_image = visualize_detections(source_image, detections)

            # Add visualization image to gallery
            vis_id = add_to_gallery(
                vis_image,
                f"Object Detection - {method}",
                f"Detected {len(detections.get('boxes', []))} objects using {method}",
                {'type': 'visualization', 'operation': operation, 'method': method}
            )

            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'metadata': {'detections': detections},
                    'image_id': vis_id,
                    'is_reprocessable_image': False,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Detect License Plates ___
        elif operation == "detect_license_plates":
            confidence_threshold = params.get('confidence_threshold', 0.3)
            ocr_engine = params.get('ocr_engine', 'easyocr')
            read_text = params.get('read_text', True)
            
            # Detect license plates
            detections = detect_license_plates(source_image, models, confidence_threshold)
            
            # Read OCR text from detected license plates
            ocr_results = []
            if read_text and detections["boxes"]:
                st.info(f"🔍 Found {len(detections['boxes'])} license plates, reading text...")
                for i, box in enumerate(detections["boxes"]):
                    try:
                        ocr_result = read_license_plate_text(source_image, box, ocr_engine)
                        ocr_results.append(ocr_result)
                        
                        if ocr_result["text"] and ocr_result["text"] not in ["OCR_FAILED", "OCR_NOT_AVAILABLE"]:
                            st.success(f"📄 License Plate {i+1}: '{ocr_result['text']}' (confidence: {ocr_result['confidence']:.2f})")
                        else:
                            st.warning(f"⚠️ License Plate {i+1}: Could not read text")
                            
                    except Exception as e:
                        logger.warning(f"Failed to read license plate {i}: {e}")
                        ocr_results.append({
                            "text": "READ_FAILED",
                            "confidence": 0.0,
                            "engine": ocr_engine,
                            "bbox": box,
                            "error": str(e)
                        })
            
            # Create visualization with OCR results
            vis_image = visualize_license_plates(source_image, detections, ocr_results if read_text else None)
            
            # Add visualization to gallery
            plate_count = len(detections["boxes"])
            readable_count = len([r for r in ocr_results if r.get("text") and r["text"] not in ["OCR_FAILED", "OCR_NOT_AVAILABLE", "READ_FAILED"]])
            
            vis_description = f"Detected {plate_count} license plates"
            if read_text:
                vis_description += f", read text from {readable_count} plates"
                
            vis_id = add_to_gallery(
                vis_image,
                f"License Plate Detection - {ocr_engine.upper()}",
                vis_description,
                {
                    'type': 'visualization', 
                    'operation': operation, 
                    'ocr_engine': ocr_engine,
                    'plates_detected': plate_count,
                    'plates_read': readable_count
                }
            )
            
            # Prepare metadata with detection and OCR results
            metadata = {
                'detections': detections,
                'plates_detected': plate_count,
                'confidence_threshold': confidence_threshold
            }
            
            if read_text:
                metadata.update({
                    'ocr_results': ocr_results,
                    'ocr_engine': ocr_engine,
                    'plates_read': readable_count,
                    'license_plates': [r["text"] for r in ocr_results if r.get("text") and r["text"] not in ["OCR_FAILED", "OCR_NOT_AVAILABLE", "READ_FAILED"]]
                })

            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'metadata': metadata,
                    'image_id': vis_id,
                    'is_reprocessable_image': False,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Remove Background ___
        elif operation == "remove_background":
            if params.get('smooth_edges', False):
                result_image = smooth_background_removal(
                    source_image, 
                    models, 
                    smooth_edges=True,
                    feather_amount=params.get('feather_amount', 10),
                    edge_refinement=params.get('edge_refinement', True)
                )
            else:
                result_image = remove_background(source_image, models)

            # Add result image to gallery
            result_id = add_to_gallery(
                result_image,
                "Background Removed",
                f"Background removal {'with smooth edges' if params.get('smooth_edges', False) else 'standard'}",
                {'type': 'result', 'operation': operation, 'smooth_edges': params.get('smooth_edges', False)}
            )
            
            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'image_id': result_id,
                    'is_reprocessable_image': True,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Add Shadow Effect ___
        elif operation == "add_shadow":
            if params.get('method') == 'smart':
                result_image = add_smart_shadow_effect(source_image)
            else:
                result_image = add_drop_shadow(
                    source_image,
                    shadow_offset=params.get('shadow_offset', (5, 5)),
                    shadow_blur=params.get('shadow_blur', 10),
                    shadow_opacity=params.get('shadow_opacity', 0.5),
                    shadow_color=params.get('shadow_color', (0, 0, 0))
                )
            
            # Add result image to gallery
            result_id = add_to_gallery(
                result_image,
                f"Drop Shadow Applied",
                f"Shadow effect using {params.get('method', 'manual')} method",
                {'type': 'result', 'operation': 'add_shadow'}
            )
            
            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'image_id': result_id,
                    'is_reprocessable_image': True,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Image-to-Image Generation ___
        elif operation == "image_to_image":
            # Get control image source if specified
            control_img_source = None
            if params.get('control_img_source'):
                control_source_id = params.get('control_img_source')
                if control_source_id in st.session_state.gallery:
                    control_img_source = st.session_state.gallery[control_source_id]['image']
            
            # Map UI control method to function control method
            control_method = params.get('control_method', 'canny_edges')
            control_method_params = {}
            
            # Set method-specific parameters
            if control_method == "canny_edges":
                if 'canny_sigma' in params:
                    control_method_params['sigma'] = params['canny_sigma']
            elif control_method == "control_img_upload":
                control_method_params['control_img_source'] = control_img_source
            # Other mask methods (mask_background_removal, mask_foreground_objects, mask_largest_object) don't need extra params
            
            control_image = prepare_control_image(
                source_image,
                control_method=control_method,
                control_method_params=control_method_params,
                models=models
            )
            
            # Store intermediate control image in gallery
            control_type_map = {
                "canny_edges": "Canny Edges",
                "mask_background_removal": "Background Removal Mask",
                "mask_foreground_objects": "Objects Mask", 
                "mask_largest_object": "Largest Object Mask",
                "control_img_upload": "Control Image"
            }
            control_type = control_type_map.get(control_method, "Canny Edges")
            control_image_id = add_to_gallery(
                control_image,
                f"{control_type} (Intermediate)",
                f"Control image for image-to-image generation",
                {'type': 'intermediate', 'operation': 'image_to_image', 'step': 'control_image'}
            )
            
            generation_result = generate_image_controlled(
                control_image,
                params.get('style_prompt', 'high quality image'),
                models,
                style=params.get('style_type', 'realistic'),
                controlnet_scale=params.get('controlnet_scale', 0.6),
                inference_steps=params.get('inference_steps', 12)
            )
            
            # Add result image to gallery
            result_id = add_to_gallery(
                generation_result['image'],
                f"Generated Image - {params.get('style_type', 'realistic')}",
                f"Image-to-image generation using {control_method} control with {params.get('style_type', 'realistic')} style",
                {'type': 'result', 'operation': 'image_to_image', 'style': params.get('style_type', 'realistic'), 'control_method': control_method}
            )

            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'metadata': {
                        'control_image_id': control_image_id,
                        'debug_params': generation_result['debug_params'],
                    },
                    'image_id': result_id,
                    'is_reprocessable_image': True,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Create Mask ___
        elif operation == "create_mask":
            # Create mask image using object detection and segmentation
            method = params.get('method', 'auto_objects')
            mask_type = params.get('mask_type', 'foreground')  # 'foreground' or 'background'
            
            if method == 'auto_background_removal':
                # Use rembg for background mask
                try:
                    bg_removed = remove_background(source_image, models)
                    if bg_removed.mode == 'RGBA':
                        # Extract alpha channel as mask
                        alpha = np.array(bg_removed.split()[-1])
                        mask = alpha > 128
                    else:
                        # Fallback: create empty mask
                        mask = np.zeros((source_image.height, source_image.width), dtype=bool)
                except Exception as e:
                    st.error(f"Background mask generation failed: {e}")
                    mask = np.zeros((source_image.height, source_image.width), dtype=bool)
                    
            elif method in ['auto_objects', 'auto_largest']:
                # Use object detection + SAM2 for segmentation
                try:
                    detection_method = params.get('detection_method', 'retinanet')
                    
                    if detection_method == 'retinanet':
                        detections = detect_objects_retinanet(source_image, models, score_threshold=0.3)
                    else:  # grounding_dino
                        text_query = params.get('text_query', 'person. human. car. vehicle. object.')
                        detections = detect_objects_grounding_dino(source_image, text_query, models)
                    
                    if detections["boxes"]:
                        masks = []
                        
                        if method == 'auto_largest':
                            # Find largest bounding box
                            largest_idx = 0
                            largest_area = 0
                            for i, box in enumerate(detections["boxes"]):
                                area = (box[2] - box[0]) * (box[3] - box[1])
                                if area > largest_area:
                                    largest_area = area
                                    largest_idx = i
                            
                            # Segment only the largest object
                            if models.get('sam2') is not None:
                                obj_mask = segment_object(source_image, detections["boxes"][largest_idx], models)
                                masks.append(obj_mask)
                        else:  # auto_objects
                            # Segment all detected objects (limit to max_objects)
                            max_objects = params.get('max_objects', 5)
                            if models.get('sam2') is not None:
                                for box in detections["boxes"][:max_objects]:
                                    obj_mask = segment_object(source_image, box, models)
                                    masks.append(obj_mask)
                        
                        # Combine all object masks
                        if masks:
                            object_mask = np.logical_or.reduce(masks)
                        else:
                            object_mask = np.zeros((source_image.height, source_image.width), dtype=bool)
                        
                        # Apply mask_type selection
                        if mask_type == 'foreground':
                            mask = object_mask  # Objects are foreground
                        else:  # background
                            mask = ~object_mask  # Everything except objects is background
                    else:
                        # No objects detected
                        if mask_type == 'foreground':
                            mask = np.zeros((source_image.height, source_image.width), dtype=bool)  # No foreground
                        else:  # background
                            mask = np.ones((source_image.height, source_image.width), dtype=bool)   # All background
                        
                except Exception as e:
                    st.error(f"Object mask generation failed: {e}")
                    mask = np.zeros((source_image.height, source_image.width), dtype=bool)
            else:
                # Fallback: create empty mask
                mask = np.zeros((source_image.height, source_image.width), dtype=bool)
            
            # Convert mask to image (white regions = mask areas, black = background)
            mask_img = np.zeros((*mask.shape, 3), dtype=np.uint8)
            mask_img[mask] = 255  # White regions where mask is True
            result_image = Image.fromarray(mask_img)
            
            # Add result to gallery
            mask_pixels = int(np.sum(mask))
            coverage = float(np.sum(mask) / mask.size)
            mask_type_desc = mask_type.capitalize()
            result_id = add_to_gallery(
                result_image,
                f"{mask_type_desc} Mask Created",
                f"{mask_type_desc} mask using {method} method - {mask_pixels} pixels ({coverage:.1%} coverage)",
                {'type': 'result', 'operation': 'create_mask', 'method': method, 'mask_type': mask_type, 'coverage': coverage}
            )

            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'metadata': {
                        'mask_pixels': mask_pixels,
                        'mask_coverage': coverage
                    },
                    'image_id': result_id,
                    'is_reprocessable_image': True,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Create Edge Map ___
        elif operation == "create_edge_map":
            # Create Canny edge map from image
            sigma = params.get('sigma', 0.33)
            edge_thickness = params.get('edge_thickness', 1)
            
            try:
                # Extract Canny edges
                edges = extract_canny_edges(source_image, sigma)
                
                # Optional: thicken edges for better visibility
                if edge_thickness > 1:
                    kernel = np.ones((edge_thickness, edge_thickness), np.uint8)
                    edges = cv2.dilate(edges, kernel, iterations=1)
                
                # Convert to RGB (white edges on black background)
                edge_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                result_image = Image.fromarray(edge_img)
                
                # Add result to gallery
                result_id = add_to_gallery(
                    result_image,
                    f"Edge Map - σ={sigma}",
                    f"Canny edge detection with sigma={sigma}, thickness={edge_thickness}",
                    {'type': 'result', 'operation': operation, 'sigma': sigma, 'edge_thickness': edge_thickness}
                )

                # Update & return message with result
                return update_message_with_result(
                    message, operation, {
                        'status': 'OK',
                        'metadata': {
                            'edge_pixels': int(np.sum(edges > 0))
                        },
                        'image_id': result_id,
                        'is_reprocessable_image': True,
                        'processed_duration_seconds': time.time() - start_time
                    }
                )

            except Exception as e:
                st.error(f"Edge map generation failed: {e}")
                # Fallback: create empty edge map
                empty_edges = np.zeros((source_image.height, source_image.width, 3), dtype=np.uint8)
                result_image = Image.fromarray(empty_edges)
                
                operation_time = time.time() - start_time

                # Update & return message with error
                return update_message_with_result(
                    message, operation, {
                        'status': 'ERROR',
                        'status_msg': str(e),
                        'processed_duration_seconds': time.time() - start_time
                    }
                )

        # ___ Operation: Enhance Image ___
        elif operation == "enhance_image":
            result_image = apply_post_processing(
                source_image,
                brightness=params.get('brightness', 1.0),
                contrast=params.get('contrast', 1.0),
                saturation=params.get('saturation', 1.0),
                sharpness=params.get('sharpness', 1.0)
            )
            
            # Add result to gallery
            adjustments = params.copy()
            result_id = add_to_gallery(
                result_image,
                f"Enhanced Image",
                f"Image enhanced with brightness={adjustments.get('brightness', 1.0)}, contrast={adjustments.get('contrast', 1.0)}, saturation={adjustments.get('saturation', 1.0)}, sharpness={adjustments.get('sharpness', 1.0)}",
                {'type': 'result', 'operation': 'enhance_image', 'adjustments': adjustments}
            )

            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'image_id': result_id,
                    'is_reprocessable_image': True,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Transform Image ___
        elif operation == "segment_objects":
            # Segment objects using provided detection boxes
            detections = params.get('detections', {})
            max_objects = params.get('max_objects', 5)
            segmentation_method = params.get('segmentation_method', 'precise')  # precise (SAM2) or simple (bbox)
            
            if not detections.get('boxes'):
                # Update & return message with error
                return update_message_with_result(
                    message, operation, {
                        'status': 'ERROR',
                        'status_msg': 'No detection boxes provided for segmentation',
                        'processed_duration_seconds': time.time() - start_time
                    }
                )

            masks = []
            if detections["boxes"]:
                for i, box in enumerate(detections["boxes"][:max_objects]):
                    if segmentation_method == 'precise' and models.get('sam2'):
                        # Use SAM2 for precise segmentation
                        mask = segment_object(source_image, box, models)
                    else:
                        # Simple rectangular mask from bbox
                        mask = np.zeros((source_image.height, source_image.width), dtype=bool)
                        x1, y1, x2, y2 = box
                        mask[y1:y2, x1:x2] = True
                    
                    masks.append(mask)
            
            # Combine all masks
            if masks:
                combined_mask = np.logical_or.reduce(masks)
            else:
                combined_mask = np.zeros((source_image.height, source_image.width), dtype=bool)
            
            # Convert mask to image (white regions = mask areas, black = background)
            mask_img = np.zeros((*combined_mask.shape, 3), dtype=np.uint8)
            mask_img[combined_mask] = 255
            result_image = Image.fromarray(mask_img)
            
            # Add result to gallery
            mask_pixels = int(np.sum(combined_mask))
            coverage = float(np.sum(combined_mask) / combined_mask.size)
            result_id = add_to_gallery(
                result_image,
                f"Segmented Objects",
                f"{len(masks)} objects segmented using {segmentation_method} method - {mask_pixels} pixels ({coverage:.1%})",
                {'type': 'result', 'operation': 'segment_objects', 'method': segmentation_method, 'objects': len(masks)}
            )

            # Update & return message with result
            operation_time = time.time() - start_time
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'metadata': {
                        'objects_segmented': len(masks),
                        'mask_pixels': mask_pixels,
                        'mask_coverage': coverage
                    },
                    'image_id': result_id,
                    'is_reprocessable_image': True,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Blur Regions ___
        elif operation == "blur_regions":
            # Blur specific regions based on a mask image or detections
            mask_source = params.get('mask_source')  # Can be mask image or detections
            blur_strength = params.get('blur_strength', 15)
            blur_mode = params.get('blur_mode', 'gaussian')  # gaussian, motion, box
            detections = params.get('detections', {})
            segmentation_method = params.get('segmentation_method', 'precise')
            max_objects = params.get('max_objects', 5)
            
            # If using message format, try to get detections from previous detect_objects step
            if message is not None and not detections.get('boxes'):
                detect_output = message.get('detect_objects', {}).get('output', {})
                if detect_output.get('metadata', {}).get('detections'):
                    detections = detect_output['metadata']['detections']
            
            # Get mask using helper function
            mask = extract_mask_from_source(
                mask_source, source_image, models, detections, 
                segmentation_method, max_objects
            )
            
            if mask is None:
                # Update & return message with error
                return update_message_with_result(
                    message, operation, {
                        'status': 'ERROR',
                        'status_msg': 'No valid mask provided for blurring',
                        'processed_duration_seconds': time.time() - start_time
                    }
                )

            # Apply blur to masked regions
            img_array = np.array(source_image)
            
            if blur_mode == 'gaussian':
                blurred_region = cv2.GaussianBlur(img_array, (blur_strength*2+1, blur_strength*2+1), 0)
            elif blur_mode == 'motion':
                kernel = np.zeros((blur_strength, blur_strength))
                kernel[int((blur_strength-1)/2), :] = np.ones(blur_strength)
                kernel = kernel / blur_strength
                blurred_region = cv2.filter2D(img_array, -1, kernel)
            else:  # box blur
                blurred_region = cv2.blur(img_array, (blur_strength, blur_strength))
            
            img_array[mask] = blurred_region[mask]
            result_image = Image.fromarray(img_array)
            
            # Add result to gallery
            result_id = add_to_gallery(
                result_image,
                f"Blurred Regions - {blur_mode}",
                f"Applied {blur_mode} blur (strength: {blur_strength}) to {int(np.sum(mask))} pixels",
                {'type': 'result', 'operation': operation, 'blur_mode': blur_mode}
            )

            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'metadata': {
                        'pixels_blurred': int(np.sum(mask))
                    },
                    'image_id': result_id,
                    'is_reprocessable_image': True,
                    'processed_duration_seconds': time.time() - start_time
                }
            )
        
        # ___ Operation: Extract Objects ___
        elif operation == "extract_objects":
            # Extract objects based on detections or mask
            detections = params.get('detections', {})
            max_objects = params.get('max_objects', 5)
            extract_mode = params.get('extract_mode', 'cutout')  # cutout or crop
            segmentation_method = params.get('segmentation_method', 'precise')
            return_mode = params.get('return_mode', 'last')  # last, first, all
            
            # If using message format, try to get detections from previous detect_objects step
            if message is not None and not detections.get('boxes'):
                detect_output = message.get('detect_objects', {}).get('output', {})
                if detect_output.get('metadata', {}).get('detections'):
                    detections = detect_output['metadata']['detections']
            
            if not detections.get('boxes'):
                # Update & return message with error
                return update_message_with_result(
                    message, operation, {
                        'status': 'ERROR',
                        'status_msg': 'No detection boxes provided for extraction',
                        'processed_duration_seconds': time.time() - start_time
                    }
                )

            extracted_objects = []
            extracted_object_ids = []
            if detections["boxes"]:
                for i, box in enumerate(detections["boxes"][:max_objects]):
                    if extract_mode == "cutout":
                        # Create mask and extract with transparency
                        if segmentation_method == 'precise' and models.get('sam2'):
                            mask = segment_object(source_image, box, models)
                        else:
                            # Simple rectangular mask
                            mask = np.zeros((source_image.height, source_image.width), dtype=bool)
                            x1, y1, x2, y2 = box
                            mask[y1:y2, x1:x2] = True
                        
                        obj_img = source_image.copy().convert("RGBA")
                        obj_array = np.array(obj_img)
                        obj_array[~mask, 3] = 0  # Make background transparent
                        extracted_obj = Image.fromarray(obj_array)
                    else:
                        # Simple crop to bounding box
                        extracted_obj = source_image.crop(box)
                    
                    # Save extracted object to gallery
                    obj_id = add_to_gallery(
                        extracted_obj,
                        f"Extracted Object {i+1}",
                        f"Object extracted using {extract_mode} method",
                        {'type': 'result', 'operation': 'extract_objects', 'object_index': i}
                    )
                    extracted_objects.append(extracted_obj)
                    extracted_object_ids.append(obj_id)

            # Return based on return_mode
            if return_mode == 'first' and extracted_objects:
                # result_image = extracted_objects[0]
                result_id = extracted_object_ids[0]
            elif return_mode == 'last' and extracted_objects:
                # result_image = extracted_objects[-1]
                result_id = extracted_object_ids[-1]
            elif return_mode == 'all':
                # result_image = extracted_objects
                result_id = None  # TODO: Support multiple images
            else:
                # result_image = extracted_objects[-1] if extracted_objects else None
                result_id = extracted_object_ids[-1] if extracted_object_ids else None

            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'status_msg': f"Extracted {len(extracted_objects)} objects using {extract_mode} mode with {segmentation_method} segmentation",
                    'metadata': {
                        'objects_extracted': len(extracted_objects),
                    },
                    'image_id': result_id,  # TODO: Support multiple images
                    'is_reprocessable_image': True if result_id else False,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Inpaint/Remove Regions ___
        elif operation == "inpaint_regions":
            # Inpaint/remove regions based on mask or detections
            mask_source = params.get('mask_source')
            inpaint_method = params.get('inpaint_method', 'blur')  # blur, average_color, pattern, ai_inpaint
            detections = params.get('detections', {})
            segmentation_method = params.get('segmentation_method', 'precise')
            max_objects = params.get('max_objects', 5)
            
            # AI inpainting parameters
            ai_inpaint_prompt = params.get('ai_inpaint_prompt', 'seamless natural background, high quality')
            ai_inpaint_inference_steps = params.get('ai_inpaint_inference_steps', 20)
            ai_inpaint_guidance_scale = params.get('ai_inpaint_guidance_scale', 7.5)
            ai_inpaint_strength = params.get('ai_inpaint_strength', 0.7)
            
            # New features for background replacement with object preservation
            preserve_objects = params.get('preserve_objects', False)
            
            # If using message format, try to get detections from previous detect_objects step
            if message is not None and not detections.get('boxes'):
                detect_output = message.get('detect_objects', {}).get('output', {})
                if detect_output.get('metadata', {}).get('detections'):
                    detections = detect_output['metadata']['detections']
                    logger.info(f"Using detections from previous detect_objects operation: {detections}")

            # Handle special mask_source for background replacement
            if mask_source == 'background_from_detections':
                # Create background mask from detections (everything EXCEPT objects)
                if detections and detections.get('boxes'):
                    logger.info(f"Creating background mask from {len(detections['boxes'])} detected objects")
                    object_masks = []
                    for box in detections["boxes"][:max_objects]:
                        if segmentation_method == 'precise' and models.get('sam2'):
                            obj_mask = segment_object(source_image, box, models)
                        else:
                            # Simple rectangular mask from bbox
                            obj_mask = np.zeros((source_image.height, source_image.width), dtype=bool)
                            x1, y1, x2, y2 = map(int, box)
                            obj_mask[y1:y2, x1:x2] = True
                        object_masks.append(obj_mask)
                    
                    # Combine all object masks
                    if object_masks:
                        combined_object_mask = np.logical_or.reduce(object_masks)
                        # Background mask = everything EXCEPT objects
                        mask = ~combined_object_mask
                        logger.info(f"Background mask created: {np.sum(mask)} pixels ({np.sum(mask)/mask.size*100:.1f}%)")
                    else:
                        mask = np.ones((source_image.height, source_image.width), dtype=bool)  # All background
                else:
                    # No objects detected, mask entire image as background
                    mask = np.ones((source_image.height, source_image.width), dtype=bool)
                    logger.warning("No objects detected, treating entire image as background")
            else:
                # CRITICAL: If no mask_source provided, try to use the mask from the previous create_mask operation
                if mask_source is None and message is not None:
                    # Get the image_id from the previous create_mask step
                    create_mask_output = message.get('create_mask', {}).get('output', {})
                    if create_mask_output.get('image_id'):
                        mask_source = create_mask_output['image_id']
                        logger.info(f"Using mask from previous create_mask operation: {mask_source}")
                
                # Get mask using helper function
                mask = extract_mask_from_source(
                    mask_source, source_image, models, detections, 
                    segmentation_method, max_objects
                )
            
            if mask is None:
                # Update & return message with error
                return update_message_with_result(
                    message, operation, {
                        'status': 'ERROR',
                        'status_msg': 'No valid mask provided',
                        'processed_duration_seconds': time.time() - start_time
                    }
                )

            # Choose inpainting method
            if inpaint_method == 'ai_inpaint':
                # AI-powered inpainting using SDXL Inpainting
                if models.get('sdxl_inpaint') is None:
                    error_msg = "AI inpainting model not available. Please ensure SDXL Inpainting model is loaded."
                    st.error(error_msg)                 
                    # Update & return message with error
                    return update_message_with_result(
                        message, operation, {
                            'status': 'ERROR',
                            'status_msg': error_msg,
                            'processed_duration_seconds': time.time() - start_time
                        }
                    )

                try:
                    # Convert boolean mask to PIL Image (white = inpaint, black = keep)
                    mask_img = np.zeros((*source_image.size[::-1], 3), dtype=np.uint8)
                    mask_img[mask] = 255  # White where we want to inpaint
                    mask_pil = Image.fromarray(mask_img)
                    mask_pixels = int(np.sum(mask))
                    coverage = float(np.sum(mask) / mask.size)

                    # Store mask image to gallery for reference
                    mask_id = add_to_gallery(
                        mask_pil,
                        f"Inpainting Mask (Intermediate)",
                        f"Mask used for AI inpainting - {mask_pixels} pixels ({coverage:.1%} coverage)",
                        {'type': 'intermediate', 'operation': 'inpaint_regions', 'step': 'mask_image'}
                    )

                    st.info(f"Starting AI inpainting with prompt: '{ai_inpaint_prompt}'")
                    st.info(f"Mask covers {mask_pixels} pixels ({coverage:.1%} of image)")
                    
                    # Perform AI inpainting
                    ai_inpaint_result = generate_ai_inpainting(
                        source_image,
                        mask_pil,
                        ai_inpaint_prompt,
                        models,
                        num_inference_steps=ai_inpaint_inference_steps,
                        guidance_scale=ai_inpaint_guidance_scale,
                        strength=ai_inpaint_strength
                    )
                    
                    result_image = ai_inpaint_result['image']
                    ai_inpaint_debug_info = ai_inpaint_result['debug_params']
                    
                    # CRITICAL: Object preservation for background replacement
                    if preserve_objects and detections and detections.get('boxes'):
                        st.info("🎨 Compositing original objects back onto new background...")
                        
                        # Extract objects from original image and paste them on top of inpainted background
                        original_with_objects = result_image.copy()
                        objects_preserved = 0
                        
                        for i, box in enumerate(detections["boxes"][:max_objects]):
                            try:
                                # Get precise object mask
                                if segmentation_method == 'precise' and models.get('sam2'):
                                    obj_mask = segment_object(source_image, box, models)
                                else:
                                    # Simple rectangular mask
                                    obj_mask = np.zeros((source_image.height, source_image.width), dtype=bool)
                                    x1, y1, x2, y2 = map(int, box)
                                    obj_mask[y1:y2, x1:x2] = True
                                
                                # Extract object from original image
                                if source_image.mode != 'RGBA':
                                    original_rgba = source_image.convert('RGBA')
                                else:
                                    original_rgba = source_image.copy()
                                
                                # Create object cutout with transparency
                                obj_array = np.array(original_rgba)
                                obj_array[~obj_mask, 3] = 0  # Make non-object areas transparent
                                object_cutout = Image.fromarray(obj_array, 'RGBA')
                                
                                # Paste object onto result image
                                if original_with_objects.mode != 'RGBA':
                                    original_with_objects = original_with_objects.convert('RGBA')
                                
                                original_with_objects = Image.alpha_composite(original_with_objects, object_cutout)
                                objects_preserved += 1
                                
                            except Exception as e:
                                logger.warning(f"Failed to preserve object {i}: {e}")
                                continue
                        
                        # Convert back to RGB if needed
                        if original_with_objects.mode == 'RGBA':
                            # Create white background
                            white_bg = Image.new('RGB', original_with_objects.size, (255, 255, 255))
                            result_image = Image.alpha_composite(white_bg.convert('RGBA'), original_with_objects).convert('RGB')
                        else:
                            result_image = original_with_objects
                        
                        st.success(f"✅ Successfully preserved {objects_preserved} objects on new background!")
                        
                except Exception as e:
                    error_msg = f"AI inpainting failed: {type(e).__name__}: {str(e)}"
                    st.error(error_msg)
                    st.error(f"Failed during inpainting with parameters: steps={ai_inpaint_inference_steps}, guidance={ai_inpaint_guidance_scale}, strength={ai_inpaint_strength}")                    
                    # Update & return message with error
                    return update_message_with_result(
                        message, operation, {
                            'status': 'ERROR',
                            'status_msg': error_msg,
                            'processed_duration_seconds': time.time() - start_time
                        }
                    )

            else:
                # Traditional inpainting methods
                result_array = np.array(source_image)
                
                if inpaint_method == 'blur':
                    # Apply heavy blur to masked regions
                    blur_strength = params.get('blur_strength', 25)
                    blurred = cv2.GaussianBlur(result_array, (blur_strength*2+1, blur_strength*2+1), 0)
                    result_array[mask] = blurred[mask]
                elif inpaint_method == 'average_color':
                    # Fill with average color of surrounding area
                    kernel_size = params.get('surround_kernel_size', 15)
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=2)
                    surround_mask = dilated_mask & ~mask.astype(np.uint8)
                    
                    if np.any(surround_mask):
                        avg_color = np.mean(result_array[surround_mask.astype(bool)], axis=0)
                        result_array[mask] = avg_color
                elif inpaint_method == 'pattern':
                    # Simple pattern-based inpainting using cv2.inpaint
                    mask_uint8 = mask.astype(np.uint8) * 255
                    result_array = cv2.inpaint(result_array, mask_uint8, 3, cv2.INPAINT_TELEA)
                
                result_image = Image.fromarray(result_array)
                ai_inpaint_debug_info = None
            
            # Add result to gallery
            inpaint_description = f"Removed/inpainted {int(np.sum(mask))} pixels using {inpaint_method}"
            if inpaint_method == 'ai_inpaint':
                inpaint_description += f" with prompt: '{ai_inpaint_prompt}'"
                
            result_id = add_to_gallery(
                result_image,
                f"Inpainted Regions - {inpaint_method.replace('_', ' ').title()}",
                inpaint_description,
                {'type': 'result', 'operation': operation, 'inpaint_method': inpaint_method}
            )
            
            # Update message with result
            operation_time = time.time() - start_time
            metadata = {
                'inpaint_method': inpaint_method,
                'pixels_inpainted': int(np.sum(mask))
            }
            
            # Add AI-specific metadata if used
            if inpaint_method == 'ai_inpaint' and ai_inpaint_debug_info:
                metadata.update({
                    'prompt': ai_inpaint_prompt,
                    'inference_steps': ai_inpaint_inference_steps,
                    'guidance_scale': ai_inpaint_guidance_scale,
                    'strength': ai_inpaint_strength,
                    'debug_params': ai_inpaint_debug_info
                })

            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'metadata': metadata,
                    'image_id': result_id,
                    'is_reprocessable_image': True,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Image Generation with Control ___
        elif operation == "generate_from_control":
            # Generate image using a control image (edge map, mask, etc.)
            control_image_source = params.get('control_image_source')
            prompt = params.get('prompt', 'high quality professional image')
            style_type = params.get('style_type', 'realistic')
            controlnet_scale = params.get('controlnet_scale', 0.8)
            inference_steps = params.get('inference_steps', 20)
            
            # Get control image from different sources
            control_image = None
            
            # If using message format, try to get control image from previous step
            if message is not None and control_image is None:
                # Look for create_edge_map output first
                edge_output = message.get('create_edge_map', {}).get('output', {})
                if edge_output.get('image_id'):
                    gallery_id = edge_output['image_id']
                    if 'gallery' in st.session_state and gallery_id in st.session_state.gallery:
                        control_image = st.session_state.gallery[gallery_id]['image']
                
                # If no edge map found, try to get the latest image from common data
                if control_image is None:
                    control_image = get_latest_image_from_message(message)
            
            # Legacy method: get from params
            if control_image is None and control_image_source is not None:
                if hasattr(control_image_source, 'mode'):  # PIL Image
                    control_image = control_image_source
                elif isinstance(control_image_source, str) and 'gallery' in st.session_state and control_image_source in st.session_state.gallery:
                    # Get from gallery
                    control_image = st.session_state.gallery[control_image_source]['image']
                elif isinstance(control_image_source, str):
                    # Assume it's a gallery ID but check if it exists
                    if 'gallery' in st.session_state and control_image_source in st.session_state.gallery:
                        control_image = st.session_state.gallery[control_image_source]['image']
            
            if control_image is None:
                # Update & return message with error
                return update_message_with_result(
                    message, operation, {
                        'status': 'ERROR',
                        'status_msg': 'No valid control image provided',
                        'processed_duration_seconds': time.time() - start_time
                    }
                )

            if not models.get('sdxl'):
                # Update & return message with error
                return update_message_with_result(
                    message, operation, {
                        'status': 'ERROR',
                        'status_msg': 'SDXL model not available for generation',
                        'processed_duration_seconds': time.time() - start_time
                    }
                )

            generation_result = generate_image_controlled(
                control_image,
                prompt,
                models,
                style=style_type,
                controlnet_scale=controlnet_scale,
                inference_steps=inference_steps
            )
            
            # Add result to gallery
            result_id = add_to_gallery(
                generation_result['image'],
                f"Generated from Control - {style_type}",
                f"Generated using {style_type} style with prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}",
                {'type': 'result', 'operation': operation, 'style': style_type, 'prompt': prompt}
            )

            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'metadata': {
                        'prompt': prompt,
                        'style_type': style_type,
                        'controlnet_scale': controlnet_scale,
                        'inference_steps': inference_steps,
                        'debug_params': generation_result['debug_params']
                    },
                    'image_id': result_id,
                    'is_reprocessable_image': True,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Outpaint Image ___
        elif operation == "outpaint_image":
            # AI-powered outpainting to extend image canvas
            prompt = params.get('prompt', 'natural environment, seamless extension')
            extend_top = params.get('extend_top', 0)
            extend_bottom = params.get('extend_bottom', 0)
            extend_left = params.get('extend_left', 0)
            extend_right = params.get('extend_right', 0)
            ai_inpaint_negative_prompt = params.get('negative_prompt', 'blurry, low quality, distorted, deformed, artifacts, border, frame')
            ai_inpaint_inference_steps = params.get('inference_steps', 20)
            ai_inpaint_guidance_scale = params.get('guidance_scale', 7.5)
            ai_inpaint_strength = params.get('strength', 1.0)
            
            try:
                # Debug info before attempting outpainting
                st.info(f"🎨 Starting AI outpainting - extending {extend_top}↑ {extend_bottom}↓ {extend_left}← {extend_right}→ pixels")
                st.info(f"📝 Prompt: '{prompt}'")
                
                # Perform outpainting 
                ai_outpaint_result = generate_ai_outpainting(
                    source_image,
                    prompt,
                    models,
                    extend_top=extend_top,
                    extend_bottom=extend_bottom,
                    extend_left=extend_left,
                    extend_right=extend_right,
                    outpaint_method="inpaint",  # Use inpaint method for now
                    negative_prompt=ai_inpaint_negative_prompt,
                    num_inference_steps=ai_inpaint_inference_steps,
                    guidance_scale=ai_inpaint_guidance_scale,
                    strength=ai_inpaint_strength
                )

                # Save intermediate results to gallery
                canvas_id = add_to_gallery(
                    ai_outpaint_result['extended_canvas'],
                    "Extended Canvas (Before Outpainting)",
                    f"Canvas extended by {extend_top}↑ {extend_bottom}↓ {extend_left}← {extend_right}→ pixels",
                    {'type': 'intermediate', 'operation': 'outpaint_image', 'step': 'canvas'}
                )

                mask_id = add_to_gallery(
                    ai_outpaint_result['mask'],
                    "Outpainting Mask",
                    "Mask showing areas to be generated (white = generate, black = keep original)",
                    {'type': 'intermediate', 'operation': 'outpaint_image', 'step': 'mask'}
                )

                generated_raw_id = add_to_gallery(
                    ai_outpaint_result['generated_raw'],
                    "Generated Raw Image",
                    "Raw image generated by AI outpainting",
                    {'type': 'intermediate', 'operation': 'outpaint_image', 'step': 'generated_raw'}
                )

                # Add result to gallery
                result_image = ai_outpaint_result['image']
                ai_outpaint_debug_info = ai_outpaint_result['debug_params']
                extension_text = f"{extend_top}↑ {extend_bottom}↓ {extend_left}← {extend_right}→" if any([extend_top, extend_bottom, extend_left, extend_right]) else "no extension"
                result_id = add_to_gallery(
                    result_image,
                    f"AI Outpainted Image",
                    f"Extended image using AI outpainting: {extension_text} pixels with prompt '{prompt}'",
                    {
                        'type': 'result', 
                        'operation': 'outpaint_image',
                        'prompt': prompt,
                        'extensions': {
                            'top': extend_top,
                            'bottom': extend_bottom, 
                            'left': extend_left,
                            'right': extend_right
                        },
                        'ai_params': ai_outpaint_debug_info,
                        'canvas_id': canvas_id,
                        'mask_id': mask_id,
                        'generated_raw_id': generated_raw_id
                    }
                )

                # Update & return message with result
                return update_message_with_result(
                    message, operation, {
                        'status': 'OK',
                        'metadata': {
                            'prompt': prompt,
                            'extensions': {
                                'top': extend_top,
                                'bottom': extend_bottom,
                                'left': extend_left,
                                'right': extend_right
                            },
                            'original_size': source_image.size,
                            'extended_size': result_image.size,
                            'ai_params': ai_outpaint_debug_info,
                            'intermediate_ids': {
                                'canvas_id': canvas_id,
                                'mask_id': mask_id
                            }
                        },
                        'image_id': result_id,
                        'is_reprocessable_image': True,
                        'processed_duration_seconds': time.time() - start_time
                    }
                )
            
            except Exception as e:
                # Return error without fallback - show detailed error  
                error_msg = f"AI outpainting failed: {type(e).__name__}: {str(e)}"
                st.error(error_msg)
                st.error(f"Failed during outpainting with extensions: {extend_top}↑ {extend_bottom}↓ {extend_left}← {extend_right}→")
                st.error(f"Debug params: steps={ai_inpaint_inference_steps}, guidance={ai_inpaint_guidance_scale}, strength={ai_inpaint_strength}")
                st.info(f"Prompt was: '{prompt}'")

                # Update & return message with error
                return update_message_with_result(
                    message, operation, {
                        'status': 'ERROR',
                        'status_msg': error_msg,
                        'processed_duration_seconds': time.time() - start_time
                    }
                )

        # ___ Operation: Transform Image (Rotate, Flip, Crop) ___            
        elif operation == "transform_image":
            # Image transformation: rotate, flip, or crop
            transform_type = params.get('transform_type', 'rotate')
            
            try:
                if transform_type == "rotate":
                    angle = params.get('angle', 0)
                    expand = params.get('expand', True)
                    fill_color = params.get('fill_color_rgba', (255, 255, 255, 0))
                    background_type = params.get('background_type', 'transparent')
                    
                    st.info(f"🔄 Rotating image by {angle}° ({'expanding canvas' if expand else 'maintaining size'})")
                    st.info(f"📊 Parameters: angle={angle}, expand={expand}, background={background_type}")
                    st.info(f"🎨 Fill color: RGBA{fill_color}")
                    
                    # Add debugging - show if angle is actually non-zero
                    if angle == 0:
                        st.warning("⚠️ Rotation angle is 0° - no rotation will be applied")
                    else:
                        st.info(f"✅ Applying {angle}° rotation with {background_type} background...")
                    
                    result_image = rotate_image(
                        source_image,
                        angle=angle,
                        expand=expand,
                        fill_color=fill_color,
                        background_type=background_type
                    )
                    
                    transform_desc = f"Rotated {angle}° {'with canvas expansion' if expand else 'maintaining size'}"
                    
                elif transform_type == "flip":
                    direction = params.get('flip_direction', 'horizontal')
                    
                    st.info(f"🔄 Flipping image {direction}ly")
                    
                    result_image = flip_image(source_image, direction=direction)
                    
                    transform_desc = f"Flipped {direction}ly"
                    
                elif transform_type == "crop":
                    crop_mode = params.get('crop_mode', 'center')
                    
                    if crop_mode == "custom":
                        crop_box = (
                            params.get('crop_left', 0),
                            params.get('crop_top', 0), 
                            params.get('crop_right', source_image.width),
                            params.get('crop_bottom', source_image.height)
                        )
                    else:
                        crop_box = None
                    
                    aspect_ratio = params.get('aspect_ratio', '1:1')
                    
                    st.info(f"✂️ Cropping image using {crop_mode} mode")
                    
                    crop_result = crop_image(
                        source_image,
                        crop_box=crop_box,
                        crop_mode=crop_mode,
                        aspect_ratio=aspect_ratio
                    )
                    
                    result_image = crop_result['image']
                    crop_info = crop_result['crop_info']
                    
                    original_size = crop_info['original_size']
                    cropped_size = crop_info['cropped_size']
                    transform_desc = f"Cropped from {original_size[0]}×{original_size[1]} to {cropped_size[0]}×{cropped_size[1]} using {crop_mode}"
                    
                else:
                    raise ValueError(f"Unknown transform type: {transform_type}")
                
            except Exception as e:
                error_msg = f"Image transformation failed: {type(e).__name__}: {str(e)}"
                st.error(error_msg)
                st.error(f"Transform type: {transform_type}, Parameters: {params}")
                # Update & return message with error
                return update_message_with_result(
                    message, operation, {
                        'status': 'ERROR', 
                        'status_msg': error_msg,
                        'processed_duration_seconds': time.time() - start_time
                    }
                )
            
            # Add result to gallery
            result_id = add_to_gallery(
                result_image,
                f"Transformed Image - {transform_type.title()}",
                transform_desc,
                {
                    'type': 'result',
                    'operation': 'transform_image', 
                    'transform_type': transform_type,
                    'transform_params': params,
                    'original_size': source_image.size,
                    'result_size': result_image.size
                }
            )

            # Update & return message with result
            return update_message_with_result(
                message, operation, {
                    'status': 'OK',
                    'metadata': {
                        'transform_type': transform_type,
                        'transform_params': params,
                        'original_size': source_image.size,
                        'result_size': result_image.size,
                        'transform_description': transform_desc
                    },
                    'image_id': result_id,
                    'is_reprocessable_image': True,
                    'processed_duration_seconds': time.time() - start_time
                }
            )

        # ___ Operation: Unknown ___
        else:
            # Update & return message with error
            return update_message_with_result(message, operation, {
                'status': 'ERROR',
                'status_msg': f"Unknown operation: {operation}",
                'processed_duration_seconds': time.time() - start_time
            })
            
    except Exception as e:
        # Update & return message with error
        return update_message_with_result(
            message, operation, {
                'status': 'ERROR',
                'status_msg': str(e),
                'processed_duration_seconds': time.time() - start_time
            }
        )

def process_combined_operation(
    operation: str,
    source_image_id: str,
    models: Dict,
    params: Dict
) -> Dict[str, Any]:
    """
    Process combined operations using standardized message format for data flow.
    
    This function creates a standardized message that flows through each step,
    allowing operations to access previous results and maintain transparency.
    
    Available combined operations:
    - detect_blur_objects: detect_objects -> blur_regions
    - detect_extract_objects: detect_objects -> extract_objects  
    - detect_inpaint_objects: detect_objects -> inpaint_regions
    - detect_paint_background: detect_objects -> inpaint_regions (AI background replacement with object preservation)
    - extract_generate_objects: create_edge_map -> generate_from_control
    
    Args:
        operation: Name of the combined operation
        source_image: Input image
        models: Dictionary of loaded AI models
        params: Operation-specific parameters
        
    Returns:
        Dictionary with 'status', 'message', 'timestamp', and 'processed_duration_seconds'
    """
    # Note: source_image_id should be a gallery ID string
    # The caller should pass the image ID from gallery, not the image object

    start_time = time.time()
    message = create_pipeline_message(
        source_image_id=source_image_id,
        operation_chain=[operation],
        operation_params={operation: params or {}}
    )

    try:
        # ___ Combined Operation: Detect and Blur Objects ___
        if operation == "detect_blur_objects":
            # Define operation chain and prepare parameters
            operation_chain = ["detect_objects", "blur_regions"]
            operation_params = {
                "detect_objects": {
                    'method': params.get('detection_method', 'retinanet'),
                    'text_query': params.get('text_query', 'person. human. car. vehicle. object.')
                },
                "blur_regions": {
                    'blur_strength': params.get('blur_strength', 15),
                    'max_objects': params.get('max_objects', 5),
                    'segmentation_method': params.get('segmentation_method', 'precise'),
                    'blur_mode': params.get('blur_mode', 'gaussian')
                }
            }
            
            # Create initial message with combined operation start time
            message = create_pipeline_message(source_image_id, operation_chain, operation_params)
            
            # Step 1: Detect objects
            step1_result = process_single_operation(
                "detect_objects", 
                models=models, 
                message=message
            )
            
            if step1_result['status'] != 'OK':
                return step1_result
            
            # Step 2: Blur regions using detection results
            return process_single_operation(
                "blur_regions", 
                models=models, 
                message=step1_result
            )

        # ___ Combined Operation: Detect and Extract Objects ___
        elif operation == "detect_extract_objects":
            # Define operation chain and prepare parameters
            operation_chain = ["detect_objects", "extract_objects"]
            operation_params = {
                "detect_objects": {
                    'method': params.get('detection_method', 'retinanet'),
                    'text_query': params.get('text_query', 'person. human. car. vehicle. object.')
                },
                "extract_objects": {
                    'max_objects': params.get('max_objects', 5),
                    'extract_mode': params.get('extract_mode', 'cutout'),
                    'segmentation_method': params.get('segmentation_method', 'precise'),
                    'return_mode': params.get('return_mode', 'last')
                }
            }
            
            # Create initial message
            message = create_pipeline_message(source_image_id, operation_chain, operation_params)
            
            # Step 1: Detect objects
            step1_result = process_single_operation(
                "detect_objects", 
                models=models, 
                message=message
            )

            if step1_result['status'] != 'OK':
                return step1_result

            # Step 2: Extract objects using detection results
            return process_single_operation(
                "extract_objects", 
                models=models, 
                message=step1_result
            )

        # ___ Combined Operation: Detect and Remove/Inpaint Objects ___
        elif operation == "detect_inpaint_objects":
            # Define operation chain and prepare parameters
            operation_chain = ["detect_objects", "inpaint_regions"]
            operation_params = {
                "detect_objects": {
                    'method': params.get('detection_method', 'retinanet'),
                    'text_query': params.get('text_query', 'person. human. car. vehicle. object.')
                },
                "inpaint_regions": {
                    'max_objects': params.get('max_objects', 5),
                    'segmentation_method': params.get('segmentation_method', 'precise'),
                    'inpaint_method': params.get('inpaint_method', 'blur'),
                    'blur_strength': params.get('blur_strength', 25),
                    # AI inpainting parameters
                    'ai_inpaint_prompt': params.get('ai_inpaint_prompt', 'seamless natural background, high quality'),
                    'ai_inpaint_inference_steps': params.get('ai_inpaint_inference_steps', 20),
                    'ai_inpaint_guidance_scale': params.get('ai_inpaint_guidance_scale', 7.5),
                    'ai_inpaint_strength': params.get('ai_inpaint_strength', 0.7)
                }
            }
            
            # Create initial message
            message = create_pipeline_message(source_image_id, operation_chain, operation_params)
            
            # Step 1: Detect objects
            step1_result = process_single_operation(
                "detect_objects", 
                models=models, 
                message=message
            )

            if step1_result['status'] != 'OK':
                return step1_result

            # Step 2: Inpaint/remove objects using detection results
            return process_single_operation(
                "inpaint_regions", 
                models=models, 
                message=step1_result
            )

        # ___ Combined Operation: Detect + Paint Background ___
        elif operation == "detect_paint_background":
            # Define operation chain: detect objects -> inpaint background areas -> composite objects back
            operation_chain = ["detect_objects", "inpaint_regions"]
            operation_params = {
                "detect_objects": {
                    'method': params.get('detection_method', 'retinanet'),
                    'text_query': params.get('text_query', 'person. human. car. vehicle. object.')
                },
                "inpaint_regions": {
                    'inpaint_method': 'ai_inpaint',  # Use AI inpainting for background replacement
                    'mask_source': 'background_from_detections',  # Special flag to create background mask from detections
                    'ai_inpaint_prompt': params.get('background_prompt', 'beautiful natural background, professional photography, high quality, realistic'),
                    'ai_inpaint_negative_prompt': params.get('negative_prompt', 'blurry, low quality, distorted, deformed, artifacts, border, frame, text, watermark'),
                    'ai_inpaint_inference_steps': params.get('inference_steps', 25),
                    'ai_inpaint_guidance_scale': params.get('guidance_scale', 7.5),
                    'ai_inpaint_strength': params.get('strength', 0.95),
                    'max_objects': params.get('max_objects', 5),
                    'preserve_objects': True  # Special flag to preserve original objects
                }
            }
            
            # Create initial message
            message = create_pipeline_message(source_image_id, operation_chain, operation_params)
            
            # Step 1: Detect objects (preserve detection data for later compositing)
            step1_result = process_single_operation(
                "detect_objects",
                models=models,
                message=message
            )
            
            if step1_result['status'] != 'OK':
                return step1_result
            
            # Step 2: Inpaint background areas (with object preservation logic)
            return process_single_operation(
                "inpaint_regions",
                models=models,
                message=step1_result
            )

        # ___ Combined Operation: Edge Map and Image Generation ___            
        elif operation == "extract_generate_objects":
            # Define operation chain and prepare parameters
            operation_chain = ["create_edge_map", "generate_from_control"] if models.get('sdxl') else ["create_edge_map"]
            operation_params = {
                "create_edge_map": {
                    'sigma': params.get('sigma', 0.33),
                    'edge_thickness': params.get('edge_thickness', 1)
                }
            }
            
            if models.get('sdxl'):
                operation_params["generate_from_control"] = {
                    'prompt': params.get('prompt', 'high quality professional image'),
                    'style_type': params.get('style_type', 'realistic'),
                    'controlnet_scale': params.get('controlnet_scale', 0.8),
                    'inference_steps': params.get('inference_steps', 20)
                }
            
            # Create initial message
            message = create_pipeline_message(source_image_id, operation_chain, operation_params)
            
            # Step 1: Create edge map
            step1_result = process_single_operation(
                "create_edge_map", 
                models=models, 
                message=message
            )
            
            if step1_result['status'] != 'OK':
                return step1_result

            # Step 2: Generate image using edge map (if SDXL is available)
            if models.get('sdxl'):
                return process_single_operation(
                    "generate_from_control", 
                    models=models, 
                    message=step1_result
                )
        
        else:
            # Update & return message with error
            return update_message_with_result(
                message, operation, {
                    'status': 'ERROR',
                    'status_msg': f"Unknown combined operation: {operation}",
                    'processed_duration_seconds': time.time() - start_time
                }
            )

    except Exception as e:
        # Update & return message with error
        return update_message_with_result(
            message, operation, {
                'status': 'ERROR',
                'status_msg': str(e),
                'processed_duration_seconds': time.time() - start_time
            }
        )

# -----------------------------------------------------------------------------
# Utility Functions

def extract_mask_from_source(
    mask_source: Any, 
    source_image: Image.Image, 
    models: Dict, 
    detections: Dict = None,
    segmentation_method: str = 'precise',
    max_objects: int = 5
) -> Optional[np.ndarray]:
    """
    Extract mask from various sources (mask image, detections, etc.)
    
    Args:
        mask_source: Can be numpy array, PIL Image, gallery ID string, or None
        source_image: Original image for creating masks from detections
        models: Model dictionary for SAM2 segmentation
        detections: Detection results with boxes
        segmentation_method: 'precise' (SAM2) or 'simple' (bbox)
        max_objects: Maximum number of objects to process
        
    Returns:
        Boolean mask as numpy array or None if mask cannot be created
    """
    if isinstance(mask_source, np.ndarray):
        logger.info("Using provided numpy array as mask")
        return mask_source.astype(bool)
    
    elif hasattr(mask_source, 'mode'):  # PIL Image
        if mask_source.mode == 'RGBA':
            logger.info("Extracting alpha channel from RGBA mask image")
            mask_array = np.array(mask_source.split()[-1])
        elif mask_source.mode == 'L':
            logger.info("Using grayscale mask image")
            mask_array = np.array(mask_source)
        else:
            logger.info("Converting RGB mask image to grayscale")
            mask_array = np.array(mask_source.convert('L'))
        return mask_array > 128
    
    elif isinstance(mask_source, str) and 'gallery' in st.session_state and mask_source in st.session_state.gallery:
        logger.info(f"Extracting mask from gallery ID: {mask_source}")
        gallery_image = st.session_state.gallery[mask_source]['image']
        return extract_mask_from_source(gallery_image, source_image, models, detections, segmentation_method, max_objects)
    
    elif detections and detections.get('boxes'):
        logger.info("Creating mask from detection boxes")
        masks = []
        for i, box in enumerate(detections["boxes"][:max_objects]):
            if segmentation_method == 'precise' and models.get('sam2'):
                obj_mask = segment_object(source_image, box, models)
            else:
                # Simple rectangular mask from bounding box
                obj_mask = np.zeros((source_image.height, source_image.width), dtype=bool)
                x1, y1, x2, y2 = map(int, box)
                x1, x2 = max(0, x1), min(source_image.width, x2)
                y1, y2 = max(0, y1), min(source_image.height, y2)
                obj_mask[y1:y2, x1:x2] = True
            masks.append(obj_mask)
        
        if masks:
            return np.logical_or.reduce(masks)
    
    return None

def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()

def visualize_detections(image: Image.Image, detections: Dict[str, List]) -> Image.Image:
    """Visualize detection results on image."""
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
    
    for i, (box, label, score) in enumerate(zip(
        detections["boxes"], detections["labels"], detections["scores"]
    )):
        color = colors[i % len(colors)]
        draw.rectangle(box, outline=color, width=3)
        
        # Add label with score
        label_text = f"{label}: {score:.2f}"
        draw.text((box[0], box[1] - 20), label_text, fill=color)
    
    return vis_img

# -----------------------------------------------------------------------------
# Main Processing Pipeline

def process_image_express(
    image: Image.Image,
    models: Dict,
    detection_method: str = "retinanet",
    text_query: str = "",
    selected_objects: List[int] = None,
    remove_bg: bool = False,
    bg_params: Dict = None,
    add_shadow: bool = False,
    shadow_params: Dict = None,
    generate_image: bool = False,
    generation_params: Dict = None,
    post_process: bool = False,
    post_process_params: Dict = None
) -> Dict[str, Any]:
    """Main image processing pipeline."""
    results = {"original": image}
    
    try:
        # Step 1: Object Detection
        st.write("🔍 Detecting objects...")
        if detection_method == "retinanet":
            detections = detect_objects_retinanet(image, models)
        else:
            if not text_query:
                text_query = "person. human. car. vehicle. object."
            detections = detect_objects_grounding_dino(image, text_query, models)
        
        results["detections"] = detections
        
        # Visualize detections
        detection_vis = visualize_detections(image, detections)
        results["detection_visualization"] = detection_vis
        
        # Step 2: Process selected objects
        processed_image = image
        combined_mask = None
        
        if selected_objects and detections["boxes"]:
            st.write("✂️ Creating object masks...")
            masks = []
            for obj_idx in selected_objects:
                if obj_idx < len(detections["boxes"]):
                    box = detections["boxes"][obj_idx]
                    mask = segment_object(image, box, models)
                    masks.append(mask)
            
            if masks:
                # Combine all masks
                combined_mask = np.logical_or.reduce(masks)
                results["mask"] = combined_mask
        
        # Step 3: Background removal
        if remove_bg:
            st.write("🎭 Removing background...")
            bg_params = bg_params or {}
            
            if combined_mask is not None:
                # Use mask for selective background removal
                bg_removed = image.copy().convert("RGBA")
                bg_array = np.array(bg_removed)
                bg_array[~combined_mask, 3] = 0  # Make background transparent
                processed_image = Image.fromarray(bg_array)
            else:
                # Use rembg for background removal
                if bg_params.get('smooth_edges', False):
                    # Use smooth background removal
                    smooth_params = {k: v for k, v in bg_params.items() if k != 'smooth_edges'}
                    processed_image = smooth_background_removal(image, models, True, **smooth_params)
                else:
                    # Use standard background removal
                    processed_image = remove_background(image, models)
            
            results["background_removed"] = processed_image
        
        # Step 4: Image-to-image generation
        if generate_image and models.get('sdxl'):
            st.write("🎨 Generating image...")
            generation_params = generation_params or {}
            
            # Get control image source if specified
            control_img_source = None
            if generation_params.get('control_img_source'):
                control_source_id = generation_params.get('control_img_source')
                if control_source_id in st.session_state.gallery:
                    control_img_source = st.session_state.gallery[control_source_id]['image']
            
            # Map UI control method to function control method
            control_method = generation_params.get('control_method', 'canny_edges')
            control_method_params = {}
            
            # Set method-specific parameters
            if control_method == "canny_edges":
                if 'canny_sigma' in generation_params:
                    control_method_params['sigma'] = generation_params['canny_sigma']
                # Add pre-computed mask if available for canny
                if combined_mask is not None:
                    control_method_params['pre_computed_mask'] = combined_mask
            elif control_method == "control_img_upload":
                control_method_params['control_img_source'] = control_img_source
            # Other mask methods don't need extra params but can use pre-computed mask
            elif combined_mask is not None:
                control_method_params['pre_computed_mask'] = combined_mask
            
            control_image = prepare_control_image(
                image, 
                control_method=control_method,
                control_method_params=control_method_params,
                models=models
            )
            
            # Generate image using AI and get debug info
            generation_result = generate_image_controlled(
                control_image,
                generation_params.get('style_prompt', '') or "high quality image",
                models,
                style=generation_params.get('style_type', 'realistic'),
                controlnet_scale=generation_params.get('controlnet_scale', 0.6),
                inference_steps=generation_params.get('inference_steps', 12)
            )
            
            generated_image = generation_result['image']
            # Store debug info in session state for System Logs
            st.session_state.generation_debug_info = generation_result['debug_params']
            
            results["control_image"] = control_image
            results["generated"] = generated_image
            processed_image = generated_image
        
        # Step 5: Post-processing
        if post_process and post_process_params:
            st.write("⚡ Applying post-processing...")
            processed_image = apply_post_processing(processed_image, **post_process_params)
            results["post_processed"] = processed_image
        
        # Step 6: Shadow effect
        if add_shadow:
            st.write("🌟 Adding drop shadow...")
            shadow_params = shadow_params or {}
            
            if shadow_params.get('method') == 'smart':
                # Use smart shadow algorithm
                processed_image = add_smart_shadow_effect(processed_image)
            else:
                # Use manual shadow with custom parameters
                manual_params = {k: v for k, v in shadow_params.items() if k != 'method'}
                processed_image = add_drop_shadow(processed_image, **manual_params)
            
            results["with_shadow"] = processed_image
        
        results["final"] = processed_image
        st.write("✅ Processing complete!")
        
        return results
        
    except Exception as e:
        st.error(f"Processing failed: {e}")
        return results

# -----------------------------------------------------------------------------
# Streamlit App Main Function

def main():
    st.set_page_config(
        page_title="All-in-One Image Express Editor",
        page_icon="🎨",
        layout="wide"
    )
    
    st.title("🎨 All-in-One Image Express Editor")
    st.markdown("Upload any image and apply AI-powered editing!")
    
    # Initialize gallery system
    initialize_gallery()
    
    # Instructions - Show at the top for better UX
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        **Image Express Editor** provides comprehensive AI-powered image editing with a gallery system:
        
        ### 🚀 (Gallery-based) Workflow
        1. **Upload** your image to start
        2. **Select** any image from the gallery as source
        3. **Choose** a processing operation from the action list
        4. **Configure** parameters and apply
        5. **Results** are automatically added to the gallery
        6. **Repeat** with any image as source for complex workflows
        
        ### 🛠️ Available Operations
        
        **Single Operations:**
        - **🔍 Object Detection**: Find and visualize objects in images
        - **🎭 Background Removal**: Remove backgrounds (with smooth edge options)
        - **🌟 Drop Shadow**: Add realistic or custom shadows
        - **🎨 Image-to-Image Generation**: Transform with AI into realistic or cartoon styles using ControlNet
        - **🎯 Create Mask Image**: Generate object masks for precise control
        - **📝 Create Edge Map**: Extract edge maps for ControlNet input
        - **⚡ Image Enhancement**: Adjust brightness, contrast, saturation, sharpness
        
        **🔗 Combined Operations (Multi-Step Processing):**
        - **🔍🌫️ Detect + Blur Objects**: Automatically detect and blur specific objects
        - **🔍✂️ Detect + Extract Objects**: Find objects and extract each as separate images
        - **🔍🎨 Detect + Remove + Inpaint**: Remove detected objects and fill with inpainting
        - **📝🎨 Edge Map + Generate**: Create edge map and generate new image from it
        
        ### 📋 Gallery Features
        - **Persistent Results**: All processed images stay in the gallery
        - **Flexible Source**: Use any gallery image as input for new operations
        - **Processing Logs**: Track all operations for troubleshooting
        - **Download Options**: Get any image from the gallery        
        """)

    models, device_info = load_models()

    # Only show main features after models are loaded
    if 'models' in locals() and models:

        # Main interface
        col_upload, col_gallery = st.columns([1, 2])

        # Upload section
        with col_upload:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["png", "jpg", "jpeg", "webp"],
                help="Supported formats: PNG, JPG, JPEG, WEBP"
            )
            
            if uploaded_file is not None:
                # Load and add to gallery
                original_image = Image.open(uploaded_file).convert("RGB")
                
                if st.button("➕ Add to Gallery", type="primary"):
                    img_id = add_to_gallery(
                        original_image, 
                        f"Original: {uploaded_file.name}",
                        "Uploaded original image"
                    )
                    st.success(f"Added to gallery! ID: {img_id}")
                    st.rerun()
            
            st.markdown("---")
            st.subheader("🎭 Generate Image from Text")
            
            # Check if text-to-image model is available
            if models.get('sdxl_txt2img') is not None:
                with st.form("text_to_image_form"):
                    # Basic generation parameters
                    text_prompt = st.text_area(
                        "Generation Prompt",
                        placeholder="A beautiful landscape with mountains and lake, high quality, detailed",
                        help="Describe what you want to generate. Be specific for better results."
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        txt2img_style = st.selectbox(
                            "Style",
                            ["realistic", "cartoon"],
                            help="Realistic = photographic style, Cartoon = animated/stylized look"
                        )
                        image_width = st.selectbox(
                            "Width",
                            [512, 768, 1024, 1152],
                            index=2,
                            help="Image width in pixels"
                        )
                    
                    with col2:
                        txt2img_steps = st.slider(
                            "Inference Steps",
                            1, 100, 12,
                            help="Generation quality: 15 = fast, 20 = balanced, 30+ = high quality"
                        )
                        image_height = st.selectbox(
                            "Height", 
                            [512, 768, 1024, 1152],
                            index=2,
                            help="Image height in pixels"
                        )
                    
                    guidance_scale = st.slider(
                        "Guidance Scale",
                        1.0, 20.0, 7.5,
                        help="How closely to follow the prompt: 7.5 = balanced, 10+ = strict adherence"
                    )
                    
                    generate_button = st.form_submit_button("✨ Generate Image", type="primary")
                    
                    if generate_button and text_prompt.strip():
                        with st.spinner("🎨 Generating image from text..."):
                            try:
                                start_time = time.time()
                                generation_result = generate_image_from_text(
                                    prompt=text_prompt.strip(),
                                    models=models,
                                    style=txt2img_style,
                                    width=image_width,
                                    height=image_height,
                                    inference_steps=txt2img_steps,
                                    guidance_scale=guidance_scale
                                )
                                generation_time = time.time() - start_time
                                
                                # Add to gallery
                                img_id = add_to_gallery(
                                    generation_result['image'],
                                    f"Generated: {text_prompt[:30]}{'...' if len(text_prompt) > 30 else ''}",
                                    f"Generated from text prompt using {txt2img_style} style",
                                    {'type': 'generated', 'generation_params': generation_result['debug_params']}
                                )
                                
                                # Add to processing logs
                                add_processing_log(
                                    "text_to_image", 
                                    "text_prompt", 
                                    img_id, 
                                    {
                                        'prompt': text_prompt,
                                        'style': txt2img_style,
                                        'width': image_width,
                                        'height': image_height,
                                        'steps': txt2img_steps,
                                        'guidance_scale': guidance_scale
                                    }, 
                                    "success", 
                                    format_duration(generation_time)
                                )
                                
                                st.success(f"✅ Image generated successfully! Added to gallery as {img_id}\n⏱️ Generation time: {format_duration(generation_time)}")
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Generation failed: {str(e)}")
                                add_processing_log("text_to_image", "text_prompt", None, {'prompt': text_prompt}, "failed")
                    
                    elif generate_button and not text_prompt.strip():
                        st.warning("Please enter a text prompt to generate an image.")
            else:
                st.warning("⚠️ SDXL Text-to-Image model not available. Image generation from text is disabled.")
        
        # Gallery and Processing section
        with col_gallery:
            st.subheader("🖼️ Image Gallery")
            
            if st.session_state.gallery:
                # Show gallery images
                gallery_cols = st.columns(3)
                for i, (img_id, img_data) in enumerate(st.session_state.gallery.items()):
                    with gallery_cols[i % 3]:
                        # Check if it's an intermediate image
                        is_intermediate = img_data.get('metadata', {}).get('type') == 'intermediate'
                        image_border = "2px dashed #888" if is_intermediate else "none"
                        
                        # Display image with styling
                        st.image(
                            img_data['image'], 
                            caption=f"{img_data['title']}", 
                            width='stretch'
                        )
                        
                        # if is_intermediate:
                        #     st.caption("🔧 Intermediate image")
                        
                        # Download button for each image
                        img_bytes = image_to_bytes(img_data['image'], "PNG")
                        st.download_button(
                            f"⬇️ Download",
                            data=img_bytes,
                            file_name=f"{img_id}.png",
                            mime="image/png",
                            key=f"download_{img_id}"
                        )
            else:
                st.info("No images in gallery yet. Upload an image to get started!")
        
        # Processing Controls
        if st.session_state.gallery:
            st.markdown("---")
            st.subheader("🛠️ Processing Operations")
            
            # Create processing interface
            proc_col1, proc_col2 = st.columns([1, 2])
            
            with proc_col1:
                st.write("**Select Source Image:**")
                source_options = {img_id: f"{data['title']}" for img_id, data in st.session_state.gallery.items()}
                selected_source = st.selectbox(
                    "Source Image",
                    options=list(source_options.keys()),
                    format_func=lambda x: source_options[x],
                    key="source_selector"
                )
                
                st.write("**Select Operation:**")
                operation = st.selectbox(
                    "Processing Operation",
                    [
                        "detect_objects",
                        "detect_license_plates",
                        "remove_background", 
                        "add_shadow",
                        "image_to_image",
                        "outpaint_image",
                        "transform_image",
                        "enhance_image",
                        "create_mask",
                        "create_edge_map",
                        "combined_operations"
                    ],
                    format_func=lambda x: {
                        "detect_objects": "🔍 Object Detection",
                        "detect_license_plates": "🚗 License Plate Detection & OCR",
                        "remove_background": "🎭 Background Removal",
                        "add_shadow": "🌟 Drop Shadow",
                        "image_to_image": "🎨 Image-to-Image Generation",
                        "outpaint_image": "🖼️ AI Outpainting (Extend Canvas)",
                        "transform_image": "🔄 Transform (Rotate/Flip/Crop)",
                        "enhance_image": "⚡ Image Enhancement",
                        "create_mask": "🎯 Create Mask Image",
                        "create_edge_map": "📝 Create Edge Map",
                        "combined_operations": "🔗 Combined Operations (Multi-Step)"
                    }[x]
                )
            
            with proc_col2:
                st.write("**Operation Parameters:**")
                
                # Parameter configuration based on operation
                params = {}
                
                # Handle combined operations with sub-operation selection
                if operation == "combined_operations":
                    st.info("🔗 Combined Operations chain multiple individual operations together automatically, saving you from manual steps. Each combined operation creates intermediate results in the gallery.")
                    
                    st.write("Select Combined Operation:")
                    combined_operation = st.selectbox(
                        "Combined Operation Type",
                        [
                            "detect_blur_objects",
                            "detect_extract_objects", 
                            "detect_inpaint_objects",
                            "detect_paint_background",
                            "extract_generate_objects"
                        ],
                        format_func=lambda x: {
                            "detect_blur_objects": "🔍🌫️ Detect + Blur Objects",
                            "detect_extract_objects": "🔍✂️ Detect + Extract Objects",
                            "detect_inpaint_objects": "🔍🎨 Detect + Inpaint Objects",
                            "detect_paint_background": "🔍🌅 Detect + Replace Background",
                            "extract_generate_objects": "📝🎨 Extract Edge Map + Generate Objects"
                        }[x],
                        help="Multi-step operations that chain individual operations together"
                    )
                    
                    # Set the actual operation to the combined operation for processing
                    actual_operation = combined_operation
                    
                    # Show process flow for selected combined operation
                    flow_info = {
                        "detect_blur_objects": "Process: Object Detection → Precise Segmentation → Gaussian Blur",
                        "detect_extract_objects": "Process: Object Detection → Precise Segmentation → Object Extraction",
                        "detect_inpaint_objects": "Process: Object Detection → Precise Segmentation → Smart Inpainting",
                        "detect_paint_background": "Process: Object Detection → Create Background Mask → AI Inpainting for Background Replacement (preserves image size)",
                        "extract_generate_objects": "Process: Edge Detection → AI Image Generation (if SDXL available)"
                    }
                    st.markdown(f"📋 {flow_info[combined_operation]}")
                    
                    st.markdown("---")
                    st.write("Configure Parameters:")
                
                else:
                    actual_operation = operation
                
                if actual_operation == "detect_objects":
                    method = create_detection_method_ui("RetinaNet detects common objects automatically, Grounding DINO lets you search for specific things with text")
                    params['method'] = method
                    if method == "grounding_dino":
                        params['text_query'] = create_text_query_ui("person. human. car. vehicle.")
                
                elif actual_operation == "detect_license_plates":
                    st.write("🚗 License Plate Detection & OCR Settings:")
                    st.info("This feature detects license plates in images and reads the text using OCR technology.")
                    
                    # Detection confidence threshold
                    params['confidence_threshold'] = st.slider(
                        "Detection Confidence", 
                        0.1, 1.0, 0.3, step=0.05,
                        help="Minimum confidence for license plate detection. Lower = more detections but more false positives."
                    )
                    
                    # OCR options
                    params['read_text'] = st.checkbox(
                        "Read License Plate Text (OCR)", 
                        True, 
                        help="Extract text from detected license plates using OCR"
                    )
                    
                    if params['read_text']:
                        params['ocr_engine'] = st.selectbox(
                            "OCR Engine",
                            ["easyocr", "tesseract"],
                            format_func=lambda x: {
                                "easyocr": "EasyOCR (Recommended - Better accuracy for license plates)",
                                "tesseract": "Tesseract OCR (Fallback - Good general purpose OCR)"
                            }[x],
                            help="OCR engine for text recognition. EasyOCR typically performs better on license plates."
                        )
                        
                        if params['ocr_engine'] == "easyocr":
                            st.info("💡 EasyOCR: Uses deep learning for accurate text recognition, especially good for license plates with various fonts and backgrounds.")
                        else:
                            st.info("💡 Tesseract: Traditional OCR engine, good fallback option. May require better image quality.")
                    
                    st.warning("⚠️ Privacy Note: License plate detection should only be used on images you own or have permission to process. Be mindful of privacy laws in your jurisdiction.")
                
                elif actual_operation == "remove_background":
                    params['smooth_edges'] = st.checkbox("Smooth Background Removal", True, 
                                                        help="Apply advanced edge smoothing for professional results")
                    if params['smooth_edges']:
                        params['feather_amount'] = st.slider("Edge Feathering", 0, 100, 10, 
                                                            help="Edge softness: 0 = sharp cut, 25 = natural, 50 = soft, 100 = very blended")
                        params['edge_refinement'] = st.checkbox("Edge Refinement", True, 
                                                               help="Apply morphological operations to clean up mask edges and remove noise")
                
                elif actual_operation == "add_shadow":
                    shadow_method = st.selectbox("Shadow Method", ["smart", "manual"], 
                                                help="Smart = automatic realistic shadow, Manual = manual control over drop shadow")
                    params['method'] = shadow_method
                    if shadow_method == "manual":
                        params['shadow_offset'] = (
                            st.slider("Shadow Offset X", -200, 200, -20, 
                                    help="Horizontal shadow distance: negative = left, positive = right, 0 = centered"),
                            st.slider("Shadow Offset Y", -200, 200, 20, 
                                    help="Vertical shadow distance: negative = up, positive = down, 0 = centered")
                        )
                        params['shadow_blur'] = st.slider("Shadow Blur", 0, 100, 10, 
                                                         help="Shadow softness: 0 = sharp edge, 50 = soft, 100 = very diffuse")
                        params['shadow_opacity'] = st.slider("Shadow Opacity", 0.0, 1.0, 0.5, 
                                                            help="Shadow transparency: 0 = invisible, 0.5 = translucent, 1 = solid")
                        color_hex = st.color_picker("Shadow Color", "#000000", 
                                                   help="Shadow color (black is most realistic)")
                        params['shadow_color'] = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                
                elif actual_operation == "image_to_image":
                    params['style_type'] = st.selectbox("Output Style", ["realistic", "cartoon"], 
                                                       help="Realistic = photographic style, Cartoon = animated/stylized look")
                    params['style_prompt'] = st.text_input("Generation Prompt", "professional high quality image", 
                                                          help="Describe the desired look, lighting, mood, or artistic effect")
                    
                    # Control Method selection
                    
                    # Build available control methods
                    control_options = ["canny_edges", "mask_background_removal", "mask_foreground_objects", "mask_largest_object"]
                    control_labels = {
                        "canny_edges": "🔍 Canny Edge Detection",
                        # --> ControlNet-Canny model does not work well with masks.
                        "mask_background_removal": "🎭 Auto Mask: Background Removal",
                        "mask_foreground_objects": "📌 Auto Mask: All Objects",
                        "mask_largest_object": "🎯 Auto Mask: Largest Object"
                    }

                    # Add Control Image Upload option if gallery has images
                    gallery_options = {img_id: f"{data['title']}" for img_id, data in st.session_state.gallery.items()}
                    if gallery_options:
                        control_options.append("control_img_upload")
                        control_labels["control_img_upload"] = "👆 Control Image from Gallery"
                    
                    params['control_method'] = st.selectbox(
                        "Control Method",
                        options=control_options,
                        format_func=lambda x: control_labels[x],
                        help="Choose how to control where the AI applies changes. Each method uses different guidance for the image-to-image generation."
                    )
                    
                    # Method-specific parameters
                    if params['control_method'] == "canny_edges":
                        # Add sigma control for edge sensitivity
                        params['canny_sigma'] = st.slider(
                            "Edge Sensitivity", 
                            0.1, 0.8, 0.33, 0.05,
                            help="Lower = cleaner edges (fewer details), Higher = more detailed edges (may include noise). 0.33 is balanced for most images."
                        )
                        
                        st.info("🔍 Canny Edges: Detects image borders/outlines to guide image generation. Recommended for most images with clear subjects.")
                    
                    elif params['control_method'] == "mask_background_removal":
                        st.info("🎭 Background Removal Mask: Automatically removes background and generates new content for foreground subjects.")
                    
                    elif params['control_method'] == "mask_foreground_objects":
                        st.info("📌 All Objects Mask: Detects and segments all objects in the image for precise generation control.")
                    
                    elif params['control_method'] == "mask_largest_object":
                        st.info("🎯 Largest Object Mask: Focuses generation on the biggest detected object only.")
                    
                    elif params['control_method'] == "control_img_upload":
                        params['control_img_source'] = st.selectbox(
                            "Select Control Image",
                            options=list(gallery_options.keys()),
                            format_func=lambda x: gallery_options[x],
                            help="Choose an image from the gallery to use as control. Can be edge map (white edges on black) or mask (white regions on black)."
                        )
                        st.info("👆 Control Image: Use an image as control. Image types: Edge map (white lines on black) (recommended), mask (white regions).")
                    
                    params['controlnet_scale'] = st.slider(
                            "Guidance Scale", 0.0, 1.5, 0.7,
                            help="Control influence: 0.3 = subtle, 0.7 = balanced, 1.0 = strong, 1.5 = maximum")
                    params['inference_steps'] = st.slider(
                            "Inference Steps", 1, 100, 12, 
                            help="Generation quality: 10 = fast, 15 = balanced, 30 = high quality, 50+ = maximum (slower)")

                elif actual_operation == "create_mask":                    
                    mask_method = st.selectbox(
                        "Mask Method",
                        ["auto_largest", "auto_objects", "auto_background_removal"],
                        format_func=lambda x: {
                            "auto_objects": "🔍 Auto-detect All Objects",
                            "auto_largest": "🎯 Largest Object Only",
                            "auto_background_removal": "🎭 Background Removal Based"
                        }[x],
                        help="Choose how to create the mask from the image"
                    )
                    params['method'] = mask_method
                    
                    # Add mask type selection for foreground vs background
                    mask_type = st.selectbox(
                        "Mask Type",
                        ["foreground", "background"],
                        format_func=lambda x: {
                            "foreground": "🎯 Foreground (Objects/Subjects)",
                            "background": "🌅 Background (Everything Else)"
                        }[x],
                        help="Choose whether to create a mask for the foreground objects or background areas"
                    )
                    params['mask_type'] = mask_type
                    
                    if mask_method in ["auto_objects", "auto_largest"]:
                        # Object detection settings
                        detection_method = create_detection_method_ui()
                        params['detection_method'] = detection_method
                        
                        if detection_method == "grounding_dino":
                            params['text_query'] = create_text_query_ui()
                        
                        if mask_method == "auto_objects":
                            params['max_objects'] = st.slider(
                                "Max Objects",
                                1, 100, 5,
                                help="Maximum number of objects to include in mask"
                            )
                        
                        st.info(f"💡 {mask_method}: Uses {detection_method} to find objects, then creates {mask_type} mask")
                    
                    elif mask_method == "auto_background_removal":
                        st.info(f"🎭 Background Removal: Uses AI background removal to identify {mask_type} areas")
                
                elif actual_operation == "create_edge_map":                    
                    params['sigma'] = st.slider(
                        "Edge Sensitivity",
                        0.1, 0.8, 0.33, 0.05,
                        help="Lower = cleaner edges (fewer details), Higher = more detailed edges (may include noise)"
                    )
                    
                    params['edge_thickness'] = st.slider(
                        "Edge Thickness",
                        1, 5, 1,
                        help="Thickness of detected edges: 1 = thin lines, 5 = thick lines"
                    )
                    
                    st.info("📝 Edge Map: Creates white edges on black background using Canny edge detection. Perfect for ControlNet input!")
                
                elif actual_operation == "enhance_image":
                    st.write("🎨 Quick Presets:")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("🔳 Grayscale", help="Convert to grayscale (saturation = 0)"):
                            st.session_state.enhance_brightness = 1.0
                            st.session_state.enhance_contrast = 1.0
                            st.session_state.enhance_saturation = 0.0
                            st.session_state.enhance_sharpness = 1.0
                            st.rerun()
                    with col2:
                        if st.button("⚫ High Contrast B&W", help="High contrast black and white effect"):
                            st.session_state.enhance_brightness = 1.0
                            st.session_state.enhance_contrast = 2.5
                            st.session_state.enhance_saturation = 0.0
                            st.session_state.enhance_sharpness = 1.0
                            st.rerun()
                    with col3:
                        if st.button("🔄 Reset", help="Reset all values to normal"):
                            st.session_state.enhance_brightness = 1.0
                            st.session_state.enhance_contrast = 1.0
                            st.session_state.enhance_saturation = 1.0
                            st.session_state.enhance_sharpness = 1.0
                            st.rerun()
                    
                    st.write("🎛️ Manual Adjustments:")
                    
                    # Initialize session state values if not set
                    if 'enhance_brightness' not in st.session_state:
                        st.session_state.enhance_brightness = 1.0
                    if 'enhance_contrast' not in st.session_state:
                        st.session_state.enhance_contrast = 1.0
                    if 'enhance_saturation' not in st.session_state:
                        st.session_state.enhance_saturation = 1.0
                    if 'enhance_sharpness' not in st.session_state:
                        st.session_state.enhance_sharpness = 1.0
                    
                    params['brightness'] = st.slider("Brightness", 0.0, 5.0, st.session_state.enhance_brightness, 
                                                   help="0 = completely black, 1 = normal, 2+ = brighter, 5 = maximum brightness",
                                                   key="brightness_slider")
                    params['contrast'] = st.slider("Contrast", 0.0, 5.0, st.session_state.enhance_contrast, 
                                                  help="0 = flat gray, 1 = normal, 2+ = higher contrast, 5 = maximum contrast",
                                                  key="contrast_slider")
                    params['saturation'] = st.slider("Saturation", 0.0, 5.0, st.session_state.enhance_saturation, 
                                                    help="0 = grayscale, 1 = normal, 2+ = more colorful, 5 = extremely vivid",
                                                    key="saturation_slider")
                    params['sharpness'] = st.slider("Sharpness", 0.0, 5.0, st.session_state.enhance_sharpness, 
                                                   help="0 = very blurry, 1 = normal, 2+ = sharper, 5 = maximum sharpness",
                                                   key="sharpness_slider")
                    
                    # Update session state when sliders change
                    st.session_state.enhance_brightness = params['brightness']
                    st.session_state.enhance_contrast = params['contrast']
                    st.session_state.enhance_saturation = params['saturation']
                    st.session_state.enhance_sharpness = params['sharpness']

                elif actual_operation == "outpaint_image":
                    st.write("🖼️ Canvas Extension Settings:")
                    st.info("Outpainting extends your image canvas and generates new content in the expanded areas using AI.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        params['extend_top'] = st.number_input(
                            "Extend Top ↑ (pixels)", 
                            min_value=0, max_value=1024, 
                            value=st.session_state.get('outpaint_top', 0), 
                            step=32,
                            help="Pixels to add above the image",
                            key="outpaint_top_input"
                        )
                        params['extend_left'] = st.number_input(
                            "Extend Left ← (pixels)", 
                            min_value=0, max_value=1024, 
                            value=st.session_state.get('outpaint_left', 0), 
                            step=32,
                            help="Pixels to add to the left of the image",
                            key="outpaint_left_input"
                        )
                    with col2:
                        params['extend_bottom'] = st.number_input(
                            "Extend Bottom ↓ (pixels)", 
                            min_value=0, max_value=1024, 
                            value=st.session_state.get('outpaint_bottom', 0), 
                            step=32,
                            help="Pixels to add below the image",
                            key="outpaint_bottom_input"
                        )
                        params['extend_right'] = st.number_input(
                            "Extend Right → (pixels)", 
                            min_value=0, max_value=1024, 
                            value=st.session_state.get('outpaint_right', 0), 
                            step=32,
                            help="Pixels to add to the right of the image",
                            key="outpaint_right_input"
                        )
                    
                    # Quick presets for common extension patterns
                    st.write("🎯 Quick Presets:")
                    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
                    with preset_col1:
                        if st.button("📐 Square (128px all)", help="Extend 128px in all directions"):
                            st.session_state.outpaint_top = 128
                            st.session_state.outpaint_bottom = 128
                            st.session_state.outpaint_left = 128
                            st.session_state.outpaint_right = 128
                            st.rerun()
                    with preset_col2:
                        if st.button("📏 Landscape (256px ←→)", help="Extend 256px left and right"):
                            st.session_state.outpaint_top = 0
                            st.session_state.outpaint_bottom = 0
                            st.session_state.outpaint_left = 256
                            st.session_state.outpaint_right = 256
                            st.rerun()
                    with preset_col3:
                        if st.button("📱 Portrait (256px ↑↓)", help="Extend 256px top and bottom"):
                            st.session_state.outpaint_top = 256
                            st.session_state.outpaint_bottom = 256
                            st.session_state.outpaint_left = 0
                            st.session_state.outpaint_right = 0
                            st.rerun()
                    with preset_col4:
                        if st.button("🔄 Reset", help="Reset all extensions to 0"):
                            st.session_state.outpaint_top = 0
                            st.session_state.outpaint_bottom = 0
                            st.session_state.outpaint_left = 0
                            st.session_state.outpaint_right = 0
                            st.rerun()
                    
                    # Check that at least one extension is specified
                    total_extension = params['extend_top'] + params['extend_bottom'] + params['extend_left'] + params['extend_right']
                    if total_extension == 0:
                        st.warning("⚠️ Please specify at least one direction to extend (top, bottom, left, or right).")
                    else:
                        original_width = 800  # Default assumption, will be updated from actual image
                        original_height = 600
                        if selected_source != "upload_new":
                            # Try to get actual dimensions from selected image
                            try:
                                if 'gallery' in st.session_state and selected_source in st.session_state.gallery:
                                    selected_image = st.session_state.gallery[selected_source]['image']
                                    original_width, original_height = selected_image.size
                            except:
                                pass
                        
                        new_width = original_width + params['extend_left'] + params['extend_right']
                        new_height = original_height + params['extend_top'] + params['extend_bottom']
                        st.info(f"📊 Canvas will expand from {original_width}×{original_height} to {new_width}×{new_height} pixels")
                    
                    st.write("🎨 Generation Settings:")
                    
                    # Outpainting method selection
                    outpaint_method = st.selectbox(
                        "Outpainting Method",
                        ["inpaint"],  # Only inpaint for now, controlnet coming soon
                        format_func=lambda x: {
                            "inpaint": "🎨 Inpainting-based (Stable, Good Quality)",
                            "controlnet": "🎯 ControlNet-based (Advanced, Best Quality) - Coming Soon"
                        }[x],
                        help="Method for outpainting: Inpaint = direct generation in extended areas"
                    )
                    
                    params['outpaint_method'] = outpaint_method
                    
                    if outpaint_method == "inpaint":
                        st.info("🎨 Inpaint Method: Extends canvas with white background, then generates content in extended areas using SDXL inpainting. Original image is perfectly preserved.")
                    elif outpaint_method == "controlnet":
                        st.info("🎯 ControlNet Method: Uses advanced ControlNet guidance for higher quality results. (Coming soon)")
                    
                    params['prompt'] = st.text_area(
                        "Generation Prompt",
                        "natural environment, seamless extension, matching style and lighting",
                        help="Describe what should be generated in the extended areas. Be specific about environment, style, and lighting to match the original image."
                    )
                    
                    with st.expander("🔧 Advanced AI Settings", expanded=False):
                        params['negative_prompt'] = st.text_area(
                            "Negative Prompt",
                            "blurry, low quality, distorted, deformed, artifacts, border, frame, text, watermark",
                            help="Describe what to avoid in the generated content"
                        )
                        params['inference_steps'] = st.slider(
                            "Inference Steps", 1, 100, 10,
                            help="More steps = higher quality but slower generation"
                        )
                        params['guidance_scale'] = st.slider(
                            "Guidance Scale", 1.0, 20.0, 7.5,
                            help="Higher values follow the prompt more closely"
                        )
                        params['strength'] = st.slider(
                            "Strength", 0.1, 1.0, 1.0, step=0.01,
                            help="How much to change the extended areas. For outpainting: 0.7-0.8=subtle, 0.85-0.95=balanced, 1.0=full generation"
                        )

                elif actual_operation == "transform_image":
                    st.write("🔄 Image Transformation Settings:")
                    st.info("Transform your image with rotation, flipping, or cropping operations.")
                    
                    transform_type = st.selectbox(
                        "Transformation Type",
                        ["rotate", "flip", "crop"],
                        format_func=lambda x: {
                            "rotate": "🔄 Rotate Image",
                            "flip": "🔄 Flip Image", 
                            "crop": "✂️ Crop Image"
                        }[x]
                    )
                    params['transform_type'] = transform_type
                    
                    if transform_type == "rotate":
                        st.write("🔄 Rotation Settings:")
                        col1, col2 = st.columns(2)
                        with col1:
                            # Initialize session state if not exists
                            if 'rotate_angle' not in st.session_state:
                                st.session_state.rotate_angle = 0
                            
                            params['angle'] = st.slider(
                                "Rotation Angle (degrees)",
                                -180, 180, st.session_state.rotate_angle, step=1,
                                help="Positive = clockwise, negative = counter-clockwise",
                                key="rotation_angle_slider"
                            )
                            
                            # Update session state when slider changes
                            st.session_state.rotate_angle = params['angle']
                        with col2:
                            params['expand'] = st.checkbox(
                                "Expand Canvas", value=True,
                                help="Expand canvas to fit the rotated image (prevents cropping)"
                            )
                        
                        # Quick angle presets
                        st.write("🎯 Quick Angles:")
                        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
                        with preset_col1:
                            if st.button("↻ 90° CW", key="rotate_90_cw"):
                                st.session_state.rotate_angle = 90
                                st.rerun()
                        with preset_col2:
                            if st.button("↺ 90° CCW", key="rotate_90_ccw"):
                                st.session_state.rotate_angle = -90
                                st.rerun()
                        with preset_col3:
                            if st.button("↻ 180°", key="rotate_180"):
                                st.session_state.rotate_angle = 180
                                st.rerun()
                        with preset_col4:
                            if st.button("🔄 Reset", key="rotate_reset"):
                                st.session_state.rotate_angle = 0
                                st.rerun()
                        
                        # Background options
                        st.write("🎨 Background Options:")
                        background_type = st.radio(
                            "Background Type",
                            ["transparent", "solid_color"],
                            format_func=lambda x: {
                                "transparent": "🔍 Transparent Background",
                                "solid_color": "🎨 Solid Color Background"
                            }[x],
                            horizontal=True,
                            help="Choose whether to use transparent background or solid color fill"
                        )
                        
                        if background_type == "transparent":
                            # For transparent background, set color to transparent
                            params['fill_color_rgba'] = (255, 255, 255, 0)  # Fully transparent
                            params['background_type'] = background_type
                            st.info("🔍 Using transparent background - rotated areas will be see-through")
                        else:
                            # Initialize session state for color if not exists
                            if 'bg_color' not in st.session_state:
                                st.session_state.bg_color = "#FFFFFF"
                            if 'bg_opacity' not in st.session_state:
                                st.session_state.bg_opacity = 1.0
                            
                            # Background Fill Color picker - only show for solid color
                            params['fill_color'] = st.color_picker(
                                "Background Fill Color", 
                                st.session_state.bg_color,
                                help="Color for transparent areas after rotation"
                            )
                            
                            # Update session state when color picker changes
                            st.session_state.bg_color = params['fill_color']
                            # Quick background presets
                            st.write("🎯 Quick Background Presets:")
                            bg_col1, bg_col2, bg_col3, bg_col4 = st.columns(4)
                            with bg_col1:
                                if st.button("⚪ White", key="bg_white"):
                                    st.session_state.bg_color = "#FFFFFF"
                                    st.session_state.bg_opacity = 1.0
                                    st.rerun()
                            with bg_col2:
                                if st.button("⚫ Black", key="bg_black"):
                                    st.session_state.bg_color = "#000000" 
                                    st.session_state.bg_opacity = 1.0
                                    st.rerun()
                            with bg_col3:
                                if st.button("🔵 Blue", key="bg_blue"):
                                    st.session_state.bg_color = "#0066CC"
                                    st.session_state.bg_opacity = 1.0
                                    st.rerun()
                            with bg_col4:
                                if st.button("🟢 Green", key="bg_green"):
                                    st.session_state.bg_color = "#00AA00"
                                    st.session_state.bg_opacity = 1.0
                                    st.rerun()
                            
                            # Update color picker if preset was used
                            if 'bg_color' in st.session_state:
                                params['fill_color'] = st.session_state.bg_color
                            
                            # Background Opacity control
                            opacity = st.slider(
                                "Background Opacity", 0.0, 1.0, st.session_state.bg_opacity, step=0.1,
                                help="0.0 = fully transparent, 1.0 = fully opaque",
                                key="bg_opacity_slider"
                            )
                            st.session_state.bg_opacity = opacity
                            
                            # Convert hex to RGBA with proper opacity
                            hex_color = params['fill_color'].lstrip('#')
                            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                            alpha = int(opacity * 255)
                            params['fill_color_rgba'] = rgb + (alpha,)
                            st.info(f"🎨 Using solid color background: #{hex_color.upper()} with {int(opacity*100)}% opacity")
                        
                        params['background_type'] = background_type
                    
                    elif transform_type == "flip":
                        st.write("🔄 Flip Settings:")
                        params['flip_direction'] = st.selectbox(
                            "Flip Direction",
                            ["horizontal", "vertical"],
                            format_func=lambda x: {
                                "horizontal": "↔️ Horizontal (Left ⟷ Right)",
                                "vertical": "↕️ Vertical (Top ⟷ Bottom)"
                            }[x]
                        )
                        
                        # Show preview info
                        direction_desc = "left and right sides" if params['flip_direction'] == 'horizontal' else "top and bottom"
                        st.info(f"📖 This will flip the image, swapping the {direction_desc}.")
                    
                    elif transform_type == "crop":
                        st.write("✂️ Crop Settings:")
                        crop_mode = st.selectbox(
                            "Crop Mode",
                            ["aspect_ratio", "center", "custom"],
                            format_func=lambda x: {
                                "aspect_ratio": "📐 Aspect Ratio Crop",
                                "center": "⚪ Center Square Crop",
                                "custom": "🎯 Custom Coordinates"
                            }[x]
                        )
                        params['crop_mode'] = crop_mode
                        
                        if crop_mode == "aspect_ratio":
                            aspect_options = ["1:1", "4:3", "3:2", "16:9", "2:3", "9:16"]
                            params['aspect_ratio'] = st.selectbox(
                                "Target Aspect Ratio",
                                aspect_options,
                                help="Choose the desired aspect ratio for the cropped image"
                            )
                            st.info(f"📊 Image will be cropped to {params['aspect_ratio']} ratio from the center.")
                        
                        elif crop_mode == "center":
                            st.info("📊 Image will be cropped to a square from the center.")
                        
                        elif crop_mode == "custom":
                            st.write("📐 Custom Crop Coordinates:")
                            st.info("💡 Set coordinates as percentages (0-100) of image dimensions for better control.")
                            
                            # Get image dimensions for reference
                            img_width, img_height = 800, 600  # Default
                            if selected_source != "upload_new":
                                try:
                                    if 'gallery' in st.session_state and selected_source in st.session_state.gallery:
                                        selected_image = st.session_state.gallery[selected_source]['image']
                                        img_width, img_height = selected_image.size
                                except:
                                    pass
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                left_pct = st.slider("Left (%)", 0, 100, 0, help="Left edge as % of image width")
                                top_pct = st.slider("Top (%)", 0, 100, 0, help="Top edge as % of image height")
                            with col2:
                                right_pct = st.slider("Right (%)", 0, 100, 100, help="Right edge as % of image width")
                                bottom_pct = st.slider("Bottom (%)", 0, 100, 100, help="Bottom edge as % of image height")
                            
                            # Convert percentages to pixel coordinates
                            params['crop_left'] = int(img_width * left_pct / 100)
                            params['crop_top'] = int(img_height * top_pct / 100)
                            params['crop_right'] = int(img_width * right_pct / 100)
                            params['crop_bottom'] = int(img_height * bottom_pct / 100)
                            
                            # Show calculated crop info
                            crop_width = params['crop_right'] - params['crop_left']
                            crop_height = params['crop_bottom'] - params['crop_top']
                            
                            if crop_width <= 0 or crop_height <= 0:
                                st.error("❌ Invalid crop dimensions. Right must be > Left and Bottom must be > Top.")
                            else:
                                st.info(f"📊 Crop region: {crop_width}×{crop_height} pixels from ({params['crop_left']}, {params['crop_top']})")

                elif actual_operation == "detect_paint_background":
                    st.write("🔍🌅 Detect + Replace Background:")
                    st.info("This operation detects objects in your image and replaces the background with AI-generated content, preserving the original image size.")
                    
                    st.write("🔍 Object Detection Settings:")
                    # Detection method
                    detection_method = st.selectbox(
                        "Detection Method",
                        ["auto_largest", "auto_objects", "auto_background_removal"],
                        format_func=lambda x: {
                            "auto_background_removal": "🎭 AI Background Removal",
                            "auto_objects": "🔍 Object Detection: All Objects",
                            "auto_largest": "🎯 Object Detection: Largest Object"
                        }[x],
                        help="Method for identifying objects to keep"
                    )
                    params['detection_method'] = detection_method
                    
                    if detection_method in ["auto_objects", "auto_largest"]:
                        if detection_method == "auto_objects":
                            params['max_objects'] = st.slider(
                                "Max Objects to Keep",
                                1, 10, 3,
                                help="Maximum number of detected objects to preserve"
                            )
                    
                    st.write("🌅 Background Replacement:")
                    params['background_prompt'] = st.text_area(
                        "New Background Description",
                        "beautiful natural landscape, professional photography, high quality, realistic lighting",
                        help="Describe the new background you want to generate behind the detected objects"
                    )
                    
                    with st.expander("🔧 Advanced AI Settings", expanded=False):
                        params['negative_prompt'] = st.text_area(
                            "Negative Prompt",
                            "blurry, low quality, distorted, deformed, artifacts, border, frame, text, watermark",
                            help="Describe what to avoid in the generated background"
                        )
                        params['inference_steps'] = st.slider(
                            "Inference Steps", 1, 100, 20,
                            help="More steps = higher quality but slower generation"
                        )
                        params['guidance_scale'] = st.slider(
                            "Guidance Scale", 1.0, 20.0, 7.5,
                            help="Higher values follow the prompt more closely"
                        )
                        params['strength'] = st.slider(
                            "Strength", 0.1, 1.0, 1.0, step=0.01,
                            help="How much to change the areas (1.0 = full generation)"
                        )
                    
                    st.info("🔍🌅 Process: Detects objects to keep → Creates background mask → AI generates new background (same image size)")

                # Combined Operations Parameters
                elif actual_operation == "detect_blur_objects":
                    
                    # Detection settings
                    detection_method = create_detection_method_ui("Method for detecting objects to blur")
                    params['detection_method'] = detection_method
                    
                    if detection_method == "grounding_dino":
                        params['text_query'] = create_text_query_ui(
                            "person. human. face. car.",
                            "Describe what objects to detect and blur"
                        )
                    
                    # Blur settings
                    params['max_objects'] = st.slider(
                        "Max Objects to Blur",
                        1, 10, 5,
                        help="Maximum number of detected objects to blur"
                    )
                    
                    params['blur_strength'] = st.slider(
                        "Blur Strength",
                        1, 50, 15,
                        help="Blur intensity: 5 = light blur, 15 = medium, 30+ = heavy blur"
                    )
                    
                    st.info("🔍🌫️ Process: Detects objects → Creates precise masks → Applies Gaussian blur to each object")
                
                elif actual_operation == "detect_extract_objects":
                    
                    # Detection settings
                    detection_method = create_detection_method_ui("Method for detecting objects to extract")
                    params['detection_method'] = detection_method
                    
                    if detection_method == "grounding_dino":
                        params['text_query'] = create_text_query_ui(
                            "person. human. car. vehicle. object.",
                            "Describe what objects to detect and extract"
                        )
                    
                    # Extraction settings
                    params['max_objects'] = st.slider(
                        "Max Objects to Extract",
                        1, 10, 3,
                        help="Maximum number of detected objects to extract (each saved separately)"
                    )
                    
                    params['extract_mode'] = st.selectbox(
                        "Extraction Mode",
                        ["cutout", "crop"],
                        format_func=lambda x: {
                            "cutout": "✂️ Precise Cutout (with transparency)",
                            "crop": "📐 Bounding Box Crop"
                        }[x],
                        help="Cutout = precise mask with transparent background, Crop = rectangular region"
                    )
                    
                    st.info("🔍✂️ Process: Detects objects → Creates individual extractions → Saves each object separately to gallery")
                
                elif actual_operation == "detect_inpaint_objects":
                    
                    # Detection settings
                    detection_method = create_detection_method_ui("Method for detecting objects to remove")
                    params['detection_method'] = detection_method
                    
                    if detection_method == "grounding_dino":
                        params['text_query'] = create_text_query_ui(
                            "person. human. car. unwanted object.",
                            "Describe what objects to detect and remove"
                        )
                    
                    # Removal settings
                    params['max_objects'] = st.slider(
                        "Max Objects to Remove",
                        1, 10, 3,
                        help="Maximum number of detected objects to remove from image"
                    )
                    
                    params['inpaint_method'] = st.selectbox(
                        "Inpainting Method",
                        ["blur", "average_color", "pattern", "ai_inpaint"],
                        format_func=lambda x: {
                            "blur": "🌫️ Blur Fill (smooth transition)",
                            "average_color": "🎨 Average Color Fill",
                            "pattern": "🖼️ Pattern Fill (OpenCV)",
                            "ai_inpaint": "🤖 AI Inpainting (SDXL)"
                        }[x],
                        help="How to fill the removed object areas"
                    )
                    
                    # AI-specific parameters if AI inpainting selected
                    if params['inpaint_method'] == 'ai_inpaint':
                        params['ai_inpaint_prompt'] = st.text_input(
                            "Inpainting Prompt",
                            "seamless natural background, high quality",
                            help="Describe what should be generated in the removed areas"
                        )
                        
                        col_ai1, col_ai2 = st.columns(2)
                        with col_ai1:
                            params['ai_inpaint_strength'] = st.slider(
                                "Strength",
                                0.1, 1.0, 1.0, step=0.01,
                                help="How much the AI changes the masked area (1.0 = full generation)"
                            )
                        with col_ai2:
                            params['ai_inpaint_guidance_scale'] = st.slider(
                                "Guidance Scale",
                                1.0, 20.0, 7.5,
                                help="How closely to follow the prompt"
                            )

                        params['ai_inpaint_inference_steps'] = st.slider(
                            "Inference Steps",
                            1, 100, 20,
                            help="More steps = higher quality but slower"
                        )
                    
                    st.info("🔍🎨 Process: Detects objects → Creates removal masks → Applies inpainting to fill removed areas")
                
                elif actual_operation == "extract_generate_objects":
                    
                    # Edge map settings
                    params['sigma'] = st.slider(
                        "Edge Sensitivity",
                        0.1, 0.8, 0.33, 0.05,
                        help="Lower = cleaner edges, Higher = more detailed edges"
                    )
                    
                    params['edge_thickness'] = st.slider(
                        "Edge Thickness",
                        1, 5, 1,
                        help="Thickness of detected edges"
                    )
                    
                    # Generation settings
                    if models.get('sdxl'):
                        params['prompt'] = st.text_input(
                            "Generation Prompt",
                            "professional high quality architectural drawing, clean lines, detailed",
                            help="Describe the style and content for the generated image"
                        )
                        
                        params['style_type'] = st.selectbox(
                            "Output Style",
                            ["realistic", "cartoon"],
                            help="Style for the generated image"
                        )
                        
                        params['controlnet_scale'] = st.slider(
                            "Guidance Scale",
                            0.0, 1.5, 0.7,
                            help="How closely to follow the edge map: 0.5 = loose, 0.7 = balanced, 1.2 = strict"
                        )
                        
                        params['inference_steps'] = st.slider(
                            "Inference Steps",
                            1, 100, 12,
                            help="Generation quality steps"
                        )
                        
                        st.info("📝🎨 Process: Creates edge map from image → Uses edge map to guide AI image generation → Produces stylized result")
                    else:
                        st.warning("⚠️ SDXL not available - will only create edge map")
                        st.info("📝 Process: Creates edge map from image (AI generation disabled)")

                # Process button
                if st.button("🚀 Apply Operation", type="primary", width='stretch'):
                    if selected_source in st.session_state.gallery:
                        # Use the gallery ID directly instead of retrieving the image object
                        source_image_id = selected_source
                        
                        with st.spinner(f"Processing {actual_operation}..."):
                            # Track processed time
                            start_time = time.time()
                            
                            # Check if it's a combined operation
                            if actual_operation in ["detect_blur_objects", "detect_extract_objects", "detect_inpaint_objects", "detect_paint_background", "extract_generate_objects"]:
                                result = process_combined_operation(actual_operation, source_image_id, models, params)
                            else:
                                result = process_single_operation(actual_operation, source_image_id, models, params)

                            if result['status'] == 'OK':
                                # Get final result image from message
                                message = result  # NEW: The whole result is actually the message
                                final_image_id = None
                                final_image = None

                                # Get the final image from the latest data in common
                                if 'common' in message and 'data' in message['common']:
                                    latest_data = message['common']['data'][-1]
                                    if latest_data.get('type') == 'image' and latest_data.get('source', '').startswith('gallery:'):
                                        final_image_id = latest_data['source'].replace('gallery:', '')
                                        if final_image_id in st.session_state.gallery:
                                            final_image = st.session_state.gallery[final_image_id]['image']
                                
                                # Fallback: if no final image found, skip adding duplicate to gallery
                                if final_image_id and final_image:
                                    # Add to processing logs with enhanced information for combined operations
                                    log_params = params.copy()
                                    
                                    add_processing_log(
                                        actual_operation, 
                                        selected_source, 
                                        final_image_id, 
                                        log_params, 
                                        "success", 
                                        result['common']['total_time'], 
                                        message=message
                                    )
                                    
                                    operation_names = {
                                        "detect_objects": "Object Detection",
                                        "detect_license_plates": "License Plates Detected",
                                        "remove_background": "Background Removed",
                                        "add_shadow": "With Shadow",
                                        "text_to_image": "Txt2Img Generated",
                                        "image_to_image": "Img2Img Generated",
                                        "outpaint_image": "AI Outpainted",
                                        "transform_image": "Image Transformed",
                                        "enhance_image": "Image Enhanced",
                                        "create_mask": "Mask Created",
                                        "create_edge_map": "Edge Map Created",
                                        "detect_blur_objects": "Objects Detected + Blurred",
                                        "detect_extract_objects": "Objects Detected + Extracted",
                                        "detect_inpaint_objects": "Objects Detected + Removed + Inpainted",
                                        "detect_paint_background": "AI Background Replaced",
                                        "extract_generate_objects": "Edge Map + Generated Image"
                                    }
                                    
                                    success_msg = f"✅ {operation_names[actual_operation]} completed! Final result: {final_image_id}"
                                    
                                    # Show pipeline information for combined operations
                                    if message and 'common' in message:
                                        operation_chain = message['common'].get('operation_chain', [])
                                        if len(operation_chain) > 1:
                                            success_msg += f"\n🔗 Pipeline: {' → '.join(operation_chain)}"
                                            success_msg += f"\n📊 Steps completed: {message['common'].get('current_step', 0)}/{len(operation_chain)}"
                                        
                                        # Show total processing time if available
                                        total_time = message['common'].get('total_time')
                                        if total_time:
                                            success_msg += f"\n⏱️ Total time: {total_time}"
                                    
                                    st.success(success_msg)
                                    st.rerun()
                                else:
                                    st.error("❌ Could not find final result image in processing pipeline")
                            else:
                                # error_message = result.get('status_msg', 'Unknown error')
                                error_message = result['status_msg']
                                
                                message = result  # NEW: The whole result is actually the message
                                
                                st.error(f"❌ Processing failed: {error_message}")
                                add_processing_log(
                                    actual_operation, 
                                    selected_source, 
                                    None, 
                                    params, 
                                    "failed", 
                                    result['common']['total_time'],
                                    message=message
                                )

        # Visual separator before status section
        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)

        # System Status and Processing History Tabs
        st.markdown("### 📊 System Information")

        tab1, tab2 = st.tabs(["📋 Processing History", "🖥️ System Status"])

        with tab1:
            display_processing_history_tab()

        with tab2:
            display_system_status_tab()


if __name__ == "__main__":
    main()

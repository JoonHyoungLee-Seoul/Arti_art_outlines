#!/usr/bin/env python3
"""
DexiNed CUDA Background Edge Detection
Processes background-only images to extract clean edge outlines using DexiNed with CUDA acceleration.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from PIL import Image

# Add DexiNed to path
DEXI_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DexiNed"))
sys.path.append(DEXI_REPO)
from model import DexiNed

# === 기본 설정 ===
DEFAULT_MODEL = "../DexiNed/checkpoints/BIPED/10/10_model.pth"
DEFAULT_INPUT_DIR = "/home/elicer/ARTI/image_clipping/clipped_images_bg"
DEFAULT_OUTPUT_DIR = "/home/elicer/ARTI/image_clipping/bg_outlines_dexined"

def get_device():
    """Get CUDA device if available, otherwise CPU"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    if torch.cuda.is_available():
        print(f"[Info] GPU: {torch.cuda.get_device_name()}")
        print(f"[Info] CUDA version: {torch.version.cuda}")
    return device

def load_dexined_model(model_path: str, device: torch.device) -> DexiNed:
    """Load DexiNed model with CUDA support"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"[Info] Loading DexiNed model from: {model_path}")
    model = DexiNed().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    print(f"[Info] Model loaded successfully on {device}")
    return model

def preprocess_image(img_path: str, target_size: int = 1024) -> tuple[torch.Tensor, tuple, tuple]:
    """
    Load and preprocess background image for DexiNed
    Background images have: transparent areas (figure) + visible areas (background)
    We need to: keep background pixels, fill figure areas (transparent) with white
    Returns: (tensor, original_shape, resized_shape)
    """
    # Load image (background has visible background pixels, transparent figure areas)
    img_pil = Image.open(img_path).convert("RGBA")
    img_array = np.array(img_pil)
    original_shape = img_array.shape[:2]  # (H, W)
    
    # Create RGB image: keep background pixels, fill transparent (figure) areas with white
    rgb_img = np.ones((img_array.shape[0], img_array.shape[1], 3), dtype=np.uint8) * 255
    
    # Where alpha > 0 (visible background), keep the original RGB values
    # Where alpha = 0 (transparent figure), keep white (255,255,255)
    alpha_mask = img_array[:, :, 3] > 0  # True for visible background pixels
    for c in range(3):
        rgb_img[:, :, c] = np.where(alpha_mask, img_array[:, :, c], 255)
    
    # Resize to target size to avoid tensor dimension issues
    h, w = original_shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Make dimensions divisible by 32 (common requirement for deep networks)
    new_h = ((new_h + 31) // 32) * 32
    new_w = ((new_w + 31) // 32) * 32
    
    resized_shape = (new_h, new_w)
    rgb_resized = cv2.resize(rgb_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Convert to BGR and apply DexiNed normalization (subtract mean BGR)
    bgr_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR)
    img_float = bgr_resized.astype(np.float32)
    
    # Subtract mean BGR values (same as DexiNed training)
    mean_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    img_float -= mean_bgr[np.newaxis, np.newaxis, :]
    
    # Convert to tensor (C, H, W) format
    tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
    
    return tensor, original_shape, resized_shape

def postprocess_edges(edge_tensor: torch.Tensor, original_shape: tuple, resized_shape: tuple) -> np.ndarray:
    """
    Postprocess DexiNed output to clean binary edges and resize back to original
    """
    # Convert tensor to numpy
    if isinstance(edge_tensor, (list, tuple)):
        edge_map = edge_tensor[-1]  # Use last output
    else:
        edge_map = edge_tensor
    
    # Apply sigmoid and get first batch item
    edge_map = torch.sigmoid(edge_map)[0, 0].detach().cpu().numpy()
    
    # Use image_normalization as DexiNed does, then invert
    from utils.image import image_normalization
    edge_normalized = image_normalization(edge_map, img_min=0, img_max=255)
    edge_normalized = edge_normalized.astype(np.uint8)
    
    # Invert edges (as DexiNed typically does with cv2.bitwise_not)
    edge_inverted = cv2.bitwise_not(edge_normalized)
    
    # Apply Gaussian blur for smoothing
    edge_blurred = cv2.GaussianBlur(edge_inverted, (3, 3), 0)
    
    # Apply thresholding for clean binary edges
    _, binary_edges = cv2.threshold(edge_blurred, 128, 255, cv2.THRESH_BINARY)
    
    # If still too few edges, try adaptive threshold
    if np.count_nonzero(binary_edges) < 5000:
        binary_edges = cv2.adaptiveThreshold(edge_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Resize back to original dimensions
    h_orig, w_orig = original_shape
    final_edges = cv2.resize(binary_edges, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    
    return final_edges

@torch.no_grad()
def process_one_image(model: DexiNed, img_path: str, output_path: str, device: torch.device):
    """Process single background image to extract edges"""
    
    # Preprocess image
    tensor, original_shape, resized_shape = preprocess_image(img_path)
    tensor = tensor.to(device)
    
    # Run DexiNed inference
    print(f"[Info] Processing: {os.path.basename(img_path)}")
    edges = model(tensor)
    
    # Postprocess edges
    clean_edges = postprocess_edges(edges, original_shape, resized_shape)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(clean_edges).save(output_path)
    print(f"[OK] Saved edges: {output_path}")
    
    return output_path

def process_batch(model: DexiNed, input_dir: str, output_dir: str, device: torch.device):
    """Process all background images in directory"""
    
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all _bg.png files
    bg_files = [f for f in os.listdir(input_dir) if f.endswith('_bg.png')]
    print(f"[Info] Found {len(bg_files)} background images in {input_dir}")
    
    success_count = 0
    for bg_file in bg_files:
        input_path = os.path.join(input_dir, bg_file)
        
        # Generate output filename: xxx_bg.png -> xxx_bg_edges.png
        base_name = bg_file.replace('_bg.png', '')
        output_filename = f"{base_name}_bg_edges.png"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            process_one_image(model, input_path, output_path, device)
            success_count += 1
        except Exception as e:
            print(f"[Warn] Failed processing {bg_file}: {e}")
    
    print(f"[Info] Successfully processed {success_count}/{len(bg_files)} background images")
    return success_count

def main():
    parser = argparse.ArgumentParser(
        description="DexiNed CUDA Background Edge Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single background image
  python dexined_bg_cuda.py -i path/to/image_bg.png -o ./output/
  
  # Batch process all background images
  python dexined_bg_cuda.py -b /home/elicer/ARTI/image_clipping/clipped_images_bg/
  
  # Custom model and output directory
  python dexined_bg_cuda.py -b ./bg_images/ -o ./edge_output/ -m ./custom_model.pth
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--image', help='Single background image file path')
    input_group.add_argument('-b', '--batch', help='Input directory containing _bg.png files')
    
    # Output directory
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_DIR,
                       help=f'Output directory for edge images (default: {DEFAULT_OUTPUT_DIR})')
    
    # Model settings
    parser.add_argument('-m', '--model', default=DEFAULT_MODEL,
                       help=f'DexiNed model checkpoint path (default: {DEFAULT_MODEL})')
    
    args = parser.parse_args()
    
    # Setup device and model
    device = get_device()
    model = load_dexined_model(args.model, device)
    
    # Process input
    if args.image:
        # Single image mode
        if not os.path.exists(args.image):
            print(f"[Error] Image not found: {args.image}")
            sys.exit(2)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        output_filename = f"{base_name}_edges.png"
        output_path = os.path.join(args.output, output_filename)
        
        process_one_image(model, args.image, output_path, device)
    
    elif args.batch:
        # Batch mode
        process_batch(model, args.batch, args.output, device)

if __name__ == "__main__":
    main()
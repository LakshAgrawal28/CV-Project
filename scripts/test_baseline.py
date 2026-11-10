import os
import sys
import time
import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Add baseline/UIE_Benchmark to path
ROOT_DIR = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT_DIR / "baseline" / "UIE_Benchmark"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

from model.UIEC2Net import UIEC2Net
from myutils.quality_refer import normalize_img

def print_device_info():
    cuda_avail = torch.cuda.is_available()
    print("=" * 50)
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {cuda_avail}")
    if cuda_avail:
        print(f"GPU Device Name : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version    : {torch.version.cuda}")
    else:
        print("Running on CPU mode")
    print("=" * 50)

def load_uiec2net_model(checkpoint_path: str, device: torch.device) -> UIEC2Net:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    model = UIEC2Net().to(device)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    
    # Strip 'model.' prefix if present from Lightning checkpoints
    new_ckpt = {}
    for key, value in ckpt.items():
        if key.startswith("model."):
            new_ckpt[key[6:]] = value
        else:
            new_ckpt[key] = value
            
    missing, unexpected = model.load_state_dict(new_ckpt, strict=False)
    if missing:
        print(f"Notice - Missing keys: {missing}")
    if unexpected:
        print(f"Notice - Unexpected keys: {unexpected}")
        
    model.eval()
    return model

def process_single_image(img_path: Path, model: UIEC2Net, output_dir: Path, device: torch.device):
    transform = transforms.ToTensor()
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda"):
                y_hat = model(x)
        else:
            y_hat = model(x)
            
        pred_tensor = normalize_img(y_hat)
    end_time = time.perf_counter()
    
    infer_ms = (end_time - start_time) * 1000
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / img_path.name
    save_image(pred_tensor[0], out_path, normalize=False)
    
    print(f"Processed: {img_path.name} | Resolution: {img.size[0]}x{img.size[1]} | Inference Time: {infer_ms:.2f} ms | Saved to: {out_path}")
    return infer_ms

def main():
    parser = argparse.ArgumentParser(description="UIEC2Net Baseline Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input image file or directory")
    parser.add_argument("--output", type=str, default="outputs/baseline", help="Directory to save enhanced output images")
    parser.add_argument("--checkpoint", type=str, default=str(BASELINE_DIR / "checkpoints" / "UIEB" / "UIEC2Net.ckpt"), help="Path to model checkpoint")
    parser.add_argument("--file_list", type=str, default=None, help="Optional text file with list of image filenames to process (e.g. data/UIEB/test.txt)")
    args = parser.parse_args()

    print_device_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading UIEC2Net baseline model from {args.checkpoint} ...")
    model = load_uiec2net_model(args.checkpoint, device)
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path '{input_path}' does not exist.")
        
    if input_path.is_file():
        image_files = [input_path]
    elif args.file_list and os.path.exists(args.file_list):
        with open(args.file_list) as fl:
            target_names = set(line.strip() for line in fl if line.strip())
        image_files = [input_path / name for name in target_names if (input_path / name).exists()]
    else:
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        
    if not image_files:
        print(f"No valid image files found in {input_path}")
        return
        
    print(f"Found {len(image_files)} image(s) to process. Device: {device}")
    total_time = 0.0
    for img_file in image_files:
        infer_ms = process_single_image(img_file, model, output_dir, device)
        total_time += infer_ms
        
    print("-" * 50)
    print(f"Completed {len(image_files)} image(s) in {total_time:.2f} ms total (Average: {total_time / len(image_files):.2f} ms/image).")
    print(f"Enhanced outputs stored in: {output_dir.resolve()}")

if __name__ == "__main__":
    main()

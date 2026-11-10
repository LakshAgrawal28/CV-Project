import os
import sys
import time
import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from proposed.model import DegradationAwareUIEC2Net
from baseline.UIE_Benchmark.myutils.quality_refer import normalize_img

def main():
    parser = argparse.ArgumentParser(description="Test Proposed Degradation-Aware UIEC2Net")
    parser.add_argument("--input", type=str, required=True, help="Input image file or directory")
    parser.add_argument("--output", type=str, default="outputs/proposed", help="Output directory")
    parser.add_argument("--baseline_ckpt", type=str, default="baseline/UIE_Benchmark/checkpoints/UIEB/UIEC2Net.ckpt", help="Baseline checkpoint")
    parser.add_argument("--proposed_ckpt", type=str, default="checkpoints/proposed_degradation.pth", help="Proposed model checkpoint")
    parser.add_argument("--file_list", type=str, default=None, help="Optional text file with list of image filenames to process (e.g. data/UIEB/test.txt)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 50)
    print(f"PyTorch Version : {torch.__version__}")
    print(f"CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name : {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU mode")
    print("=" * 50)

    model = DegradationAwareUIEC2Net().to(device)
    if os.path.exists(args.baseline_ckpt):
        model.load_baseline_weights(args.baseline_ckpt, device)
        
    if os.path.exists(args.proposed_ckpt):
        print(f"Loading trained proposed model weights from {args.proposed_ckpt}...")
        ckpt = torch.load(args.proposed_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt, strict=False)
    else:
        print(f"Notice: Proposed checkpoint {args.proposed_ckpt} not found. Running with baseline initial weights + degradation encoder.")

    model.eval()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_files = [input_path]
    elif args.file_list and os.path.exists(args.file_list):
        with open(args.file_list) as fl:
            target_names = set(line.strip() for line in fl if line.strip())
        image_files = [input_path / name for name in target_names if (input_path / name).exists()]
    else:
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]

    transform = transforms.ToTensor()
    total_time = 0.0

    print(f"Processing {len(image_files)} image(s) using Proposed Method...")
    for img_p in image_files:
        img = Image.open(img_p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        
        t0 = time.perf_counter()
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    y_hat = model(x)
            else:
                y_hat = model(x)
            pred_tensor = normalize_img(y_hat)
        t1 = time.perf_counter()
        
        infer_ms = (t1 - t0) * 1000
        total_time += infer_ms
        
        out_p = output_dir / img_p.name
        save_image(pred_tensor[0], out_p, normalize=False)
        print(f"Processed: {img_p.name} | Resolution: {img.size[0]}x{img.size[1]} | Inference Time: {infer_ms:.2f} ms | Saved to: {out_p}")

    print("-" * 50)
    print(f"Completed {len(image_files)} image(s) in {total_time:.2f} ms total.")

if __name__ == "__main__":
    main()

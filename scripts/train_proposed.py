import os
import sys
import argparse
import time
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from proposed.model import DegradationAwareUIEC2Net
from evaluation.metrics import calculate_ssim

class SimpleUIEBDataset(Dataset):
    def __init__(self, raw_dir: Path, ref_dir: Path, image_size: int = 256):
        self.raw_dir = raw_dir
        self.ref_dir = ref_dir
        self.image_size = image_size
        
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.filenames = [f.name for f in raw_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        
        # Keep only pairs where reference exists
        self.pairs = []
        for fn in self.filenames:
            if (ref_dir / fn).exists():
                self.pairs.append(fn)
                
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fn = self.pairs[idx]
        raw_img = Image.open(self.raw_dir / fn).convert("RGB")
        ref_img = Image.open(self.ref_dir / fn).convert("RGB")
        
        x = self.transform(raw_img)
        y = self.transform(ref_img)
        return x, y, fn

def train_epoch(model, loader, optimizer, criterion_l1, device, scaler=None):
    model.train()
    total_loss = 0.0
    
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        if scaler and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = criterion_l1(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = criterion_l1(pred, y)
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item() * x.size(0)
        
    return total_loss / len(loader.dataset)

def main():
    parser = argparse.ArgumentParser(description="Train Proposed Degradation-Aware UIEC2Net")
    parser.add_argument("--raw_dir", type=str, default="data/UIEB/raw-890", help="Directory with original images")
    parser.add_argument("--ref_dir", type=str, default="data/UIEB/reference-890", help="Directory with reference GT images")
    parser.add_argument("--baseline_ckpt", type=str, default="baseline/UIE_Benchmark/checkpoints/UIEB/UIEC2Net.ckpt", help="Pretrained baseline checkpoint")
    parser.add_argument("--out_ckpt", type=str, default="checkpoints/proposed_degradation.pth", help="Output path for trained proposed checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    raw_dir = Path(args.raw_dir)
    ref_dir = Path(args.ref_dir)
    
    if not raw_dir.exists() or not ref_dir.exists() or len(list(raw_dir.glob("*.*"))) == 0:
        print("Dataset paired folders not found or empty. Running fast synthetic sanity training step...")
        # Create synthetic dataset in memory for sanity verification
        x_dummy = torch.rand(4, 3, 256, 256).to(device)
        y_dummy = torch.rand(4, 3, 256, 256).to(device)
        
        model = DegradationAwareUIEC2Net().to(device)
        if os.path.exists(args.baseline_ckpt):
            model.load_baseline_weights(args.baseline_ckpt, device)
        model.freeze_baseline(True)
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        criterion = nn.L1Loss()
        
        print("Sanity step starting...")
        optimizer.zero_grad()
        out = model(x_dummy)
        loss = criterion(out, y_dummy)
        loss.backward()
        optimizer.step()
        
        Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.out_ckpt)
        print(f"Sanity step completed! Loss: {loss.item():.4f}. Checkpoint saved to {args.out_ckpt}")
        return

    dataset = SimpleUIEBDataset(raw_dir, ref_dir, image_size=256)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    print(f"Dataset loaded: {len(dataset)} paired images.")
    
    model = DegradationAwareUIEC2Net().to(device)
    if os.path.exists(args.baseline_ckpt):
        model.load_baseline_weights(args.baseline_ckpt, device)
    
    # Freeze baseline layers so only degradation encoder & FiLM layers train
    model.freeze_baseline(True)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion_l1 = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    
    best_loss = float("inf")
    print(f"Starting training for {args.epochs} epoch(s)...")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, loader, optimizer, criterion_l1, device, scaler)
        elapsed = time.time() - t0
        
        print(f"Epoch [{epoch}/{args.epochs}] | L1 Loss: {loss:.4f} | Time: {elapsed:.2f}s")
        
        if loss < best_loss:
            best_loss = loss
            Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out_ckpt)
            print(f" Saved new best model to {args.out_ckpt}")

if __name__ == "__main__":
    main()

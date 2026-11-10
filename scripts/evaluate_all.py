import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from evaluation.metrics import evaluate_pair, calculate_uiqm, calculate_uciqe
from PIL import Image
import numpy as np

def evaluate_dataset(pred_dir: Path, gt_dir: Path = None) -> pd.DataFrame:
    records = []
    
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    pred_files = [f for f in pred_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    
    print(f"Evaluating {len(pred_files)} image(s) in {pred_dir} ...")
    
    for pred_path in tqdm(pred_files):
        gt_path = None
        if gt_dir and gt_dir.exists():
            candidate = gt_dir / pred_path.name
            if candidate.exists():
                gt_path = candidate
                
        res = evaluate_pair(str(pred_path), str(gt_path) if gt_path else None)
        record = {
            "image": pred_path.name,
            "has_reference": bool(gt_path),
            "psnr": res["psnr"],
            "ssim": res["ssim"],
            "lpips": res["lpips"],
            "uiqm": res["uiqm"],
            "uciqe": res["uciqe"]
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate Enhancement Results")
    parser.add_argument("--pred_dir", type=str, default="outputs/baseline", help="Directory containing enhanced predictions")
    parser.add_argument("--gt_dir", type=str, default="data/UIEB/reference-890", help="Directory containing ground truth reference images")
    parser.add_argument("--method_name", type=str, default="Baseline", help="Method identifier (e.g., Baseline, Proposed)")
    parser.add_argument("--output_csv", type=str, default="outputs/results.csv", help="Path to save evaluation CSV")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    
    if not pred_dir.exists():
        print(f"Prediction directory '{pred_dir}' does not exist.")
        return
        
    df = evaluate_dataset(pred_dir, gt_dir)
    df["method"] = args.method_name
    
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        combined_df = pd.concat([existing_df[existing_df["method"] != args.method_name], df], ignore_index=True)
        combined_df.to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, index=False)
        
    print(f"\nEvaluation summary for method [{args.method_name}]:")
    print(f"Average PSNR  : {df['psnr'].mean():.4f}")
    print(f"Average SSIM  : {df['ssim'].mean():.4f}")
    print(f"Average LPIPS : {df['lpips'].mean():.4f}")
    print(f"Average UIQM  : {df['uiqm'].mean():.4f}")
    print(f"Average UCIQE : {df['uciqe'].mean():.4f}")
    print(f"Results saved to: {out_csv.resolve()}")

if __name__ == "__main__":
    main()

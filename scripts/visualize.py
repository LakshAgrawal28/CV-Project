import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def create_4way_side_by_side(
    raw_path: Path, 
    baseline_path: Path, 
    proposed_path: Path, 
    ref_path: Path = None,
    target_width: int = 384
) -> Image.Image:
    """
    Creates a 4-panel horizontal comparison:
    [ Raw (Original) | Baseline (UIEC2Net) | Proposed (FDM-Net) | Reference (Ground Truth) ]
    """
    raw_img = Image.open(raw_path).convert("RGB")
    aspect = raw_img.height / raw_img.width
    panel_w = target_width
    panel_h = int(panel_w * aspect)
    
    raw_img = raw_img.resize((panel_w, panel_h), Image.Resampling.LANCZOS)
    
    panels = [raw_img]
    labels = ["Original (Raw)"]
    
    if baseline_path and baseline_path.exists():
        base_img = Image.open(baseline_path).convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        panels.append(base_img)
        labels.append("Baseline (UIEC2Net)")
    else:
        blank = Image.new("RGB", (panel_w, panel_h), (40, 40, 40))
        panels.append(blank)
        labels.append("Baseline (Missing)")

    if proposed_path and proposed_path.exists():
        prop_img = Image.open(proposed_path).convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        panels.append(prop_img)
        labels.append("Proposed (FDM-Net)")
    else:
        blank = Image.new("RGB", (panel_w, panel_h), (40, 40, 40))
        panels.append(blank)
        labels.append("Proposed (Missing)")

    if ref_path and ref_path.exists():
        ref_img = Image.open(ref_path).convert("RGB").resize((panel_w, panel_h), Image.Resampling.LANCZOS)
        panels.append(ref_img)
        labels.append("Reference (Ground Truth)")
    else:
        blank = Image.new("RGB", (panel_w, panel_h), (40, 40, 40))
        panels.append(blank)
        labels.append("Reference (None)")

    banner_height = 42
    total_w = panel_w * len(panels)
    total_h = panel_h + banner_height
    
    canvas = Image.new("RGB", (total_w, total_h), (18, 22, 30))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        
    for i, (img, label) in enumerate(zip(panels, labels)):
        canvas.paste(img, (i * panel_w, banner_height))
        # Draw column header banner
        draw.rectangle([(i * panel_w, 0), ((i + 1) * panel_w, banner_height)], fill=(24, 28, 38))
        # Text alignment
        draw.text((i * panel_w + 14, 11), label, fill=(240, 240, 245), font=font)
        # Separator line between panels
        if i > 0:
            draw.line([(i * panel_w, 0), (i * panel_w, total_h)], fill=(60, 68, 85), width=2)
            
    return canvas

def create_4way_contact_sheet(image_tuples: list, output_path: Path, max_samples: int = 6):
    """
    Generates a 4-column contact sheet grid.
    image_tuples: list of (name, raw_p, base_p, prop_p, ref_p)
    """
    samples = image_tuples[:max_samples]
    if not samples:
        return
        
    n_rows = len(samples)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 3.8 * n_rows))
    if n_rows == 1:
        axes = [axes]
        
    col_titles = ["Original (Raw)", "Baseline (UIEC2Net)", "Proposed (FDM-Net)", "Reference (Ground Truth)"]
    
    for row_idx, (name, raw_p, base_p, prop_p, ref_p) in enumerate(samples):
        # Raw
        raw_img = Image.open(raw_p).convert("RGB")
        axes[row_idx][0].imshow(raw_img)
        axes[row_idx][0].set_title(f"{name} | {col_titles[0]}", fontsize=9, fontweight="bold")
        axes[row_idx][0].axis("off")
        
        # Baseline
        if base_p.exists():
            base_img = Image.open(base_p).convert("RGB").resize(raw_img.size)
            axes[row_idx][1].imshow(base_img)
            axes[row_idx][1].set_title(f"{name} | {col_titles[1]}", fontsize=9, fontweight="bold")
        axes[row_idx][1].axis("off")
        
        # Proposed
        if prop_p.exists():
            prop_img = Image.open(prop_p).convert("RGB").resize(raw_img.size)
            axes[row_idx][2].imshow(prop_img)
            axes[row_idx][2].set_title(f"{name} | {col_titles[2]}", fontsize=9, fontweight="bold")
        axes[row_idx][2].axis("off")
        
        # Reference
        if ref_p and ref_p.exists():
            ref_img = Image.open(ref_p).convert("RGB").resize(raw_img.size)
            axes[row_idx][3].imshow(ref_img)
            axes[row_idx][3].set_title(f"{name} | {col_titles[3]}", fontsize=9, fontweight="bold")
        else:
            axes[row_idx][3].axis("off")
            
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"4-way contact sheet saved to: {output_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Generate 4-Way Visual Comparisons")
    parser.add_argument("--raw_dir", type=str, default="data/UIEB/raw-890", help="Directory containing original images")
    parser.add_argument("--baseline_dir", type=str, default="outputs/baseline", help="Directory containing baseline predictions")
    parser.add_argument("--proposed_dir", type=str, default="outputs/proposed", help="Directory containing proposed predictions")
    parser.add_argument("--ref_dir", type=str, default="data/UIEB/reference-890", help="Directory containing ground truth references")
    parser.add_argument("--out_dir", type=str, default="outputs/visualizations/comparisons", help="Output comparison directory")
    parser.add_argument("--max_contact_samples", type=int, default=8, help="Number of rows in contact sheet")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    baseline_dir = Path(args.baseline_dir)
    proposed_dir = Path(args.proposed_dir)
    ref_dir = Path(args.ref_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_dir.exists() and not proposed_dir.exists():
        print("Neither baseline nor proposed output directory exists.")
        return

    # Find common images
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    active_dir = baseline_dir if baseline_dir.exists() else proposed_dir
    img_files = sorted([f for f in active_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    
    tuples = []
    for f in img_files:
        raw_p = raw_dir / f.name
        base_p = baseline_dir / f.name
        prop_p = proposed_dir / f.name
        ref_p = ref_dir / f.name
        
        if raw_p.exists():
            sbs = create_4way_side_by_side(raw_p, base_p, prop_p, ref_p if ref_p.exists() else None)
            sbs.save(out_dir / f"compare_4way_{f.name}")
            tuples.append((f.stem, raw_p, base_p, prop_p, ref_p))
            
    if tuples:
        contact_path = out_dir.parent / "4way_contact_sheet.png"
        create_4way_contact_sheet(tuples, contact_path, max_samples=args.max_contact_samples)
        print(f"Generated 4-way side-by-side comparisons for {len(tuples)} image(s) under {out_dir.resolve()}")

if __name__ == "__main__":
    main()

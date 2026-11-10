import io
import sys
import time
import base64
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baseline.UIE_Benchmark.model.UIEC2Net import UIEC2Net
from baseline.UIE_Benchmark.myutils.quality_refer import normalize_img
from proposed.model import DegradationAwareUIEC2Net
from evaluation.metrics import calculate_uiqm, calculate_uciqe

app = FastAPI(title="AquaVision — Degradation-Aware Underwater Image Enhancement API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount visualization output directory for web access
VIS_DIR = ROOT_DIR / "outputs" / "visualizations"
if VIS_DIR.exists():
    app.mount("/visualizations", StaticFiles(directory=str(VIS_DIR)), name="visualizations")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASELINE_CKPT = ROOT_DIR / "baseline" / "UIE_Benchmark" / "checkpoints" / "UIEB" / "UIEC2Net.ckpt"
PROPOSED_CKPT = ROOT_DIR / "checkpoints" / "proposed_degradation.pth"

# Load models globally at startup
baseline_model = None
proposed_model = None

def init_models():
    global baseline_model, proposed_model
    print(f"Initializing AquaVision backend models on device: {DEVICE}")
    
    # 1. Baseline
    baseline_model = UIEC2Net().to(DEVICE)
    if BASELINE_CKPT.exists():
        ckpt = torch.load(str(BASELINE_CKPT), map_location=DEVICE, weights_only=False)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        new_ckpt = { (k[6:] if k.startswith("model.") else k): v for k, v in ckpt.items() }
        baseline_model.load_state_dict(new_ckpt, strict=False)
    baseline_model.eval()

    # 2. Proposed
    proposed_model = DegradationAwareUIEC2Net().to(DEVICE)
    if BASELINE_CKPT.exists():
        proposed_model.load_baseline_weights(str(BASELINE_CKPT), DEVICE)
    if PROPOSED_CKPT.exists():
        p_ckpt = torch.load(str(PROPOSED_CKPT), map_location=DEVICE, weights_only=False)
        proposed_model.load_state_dict(p_ckpt, strict=False)
    proposed_model.eval()
    print("Models initialized successfully.")

@app.on_event("startup")
def startup_event():
    init_models()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_version": torch.version.cuda
    }

@app.get("/info")
def get_info():
    return {
        "project": "AquaVision — Degradation-Aware Underwater Image Enhancement",
        "baseline_model": "UIEC2Net (Pretrained on UIEB)",
        "proposed_method": "Degradation-Aware Feature Conditioning (z = D(X) + FiLM Modulation)",
        "device": str(DEVICE),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "checkpoint_status": {
            "baseline_loaded": BASELINE_CKPT.exists(),
            "proposed_loaded": PROPOSED_CKPT.exists()
        }
    }

@app.get("/benchmark")
def get_benchmark_summary():
    summary_path = ROOT_DIR / "outputs" / "benchmark_summary.json"
    if summary_path.exists():
        import json
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Benchmark summary not found. Run scripts/generate_quantitative_results.py"}

@app.get("/benchmark/details")
def get_benchmark_details():
    paired_csv = ROOT_DIR / "outputs" / "paired_comparison.csv"
    if paired_csv.exists():
        import pandas as pd
        df = pd.read_csv(paired_csv)
        return df.to_dict(orient="records")
    return {"error": "Paired benchmark comparison not found."}

def process_tensor_inference(model, transform, img_pil, device):
    x = transform(img_pil).unsqueeze(0).to(device)
    
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
    
    out_pil = transforms.ToPILImage()(pred_tensor[0].cpu())
    buf = io.BytesIO()
    out_pil.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    out_np = np.array(out_pil)
    uiqm = calculate_uiqm(out_np)
    uciqe = calculate_uciqe(out_np)
    
    return {
        "enhanced_b64": f"data:image/png;base64,{img_b64}",
        "inference_ms": round(infer_ms, 2),
        "image_shape": [out_pil.height, out_pil.width],
        "metrics": {
            "uiqm": round(uiqm, 4),
            "uciqe": round(uciqe, 4)
        }
    }

@app.post("/enhance")
async def enhance_image(file: UploadFile = File(...), method: str = Form("proposed")):
    if method.lower() == "baseline":
        return await enhance_baseline(file)
    else:
        return await enhance_proposed(file)

@app.post("/enhance/baseline")
async def enhance_baseline(file: UploadFile = File(...)):
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    transform = transforms.ToTensor()
    res = process_tensor_inference(baseline_model, transform, img_pil, DEVICE)
    res["method"] = "UIEC2Net Baseline"
    res["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return res

@app.post("/enhance/proposed")
async def enhance_proposed(file: UploadFile = File(...)):
    contents = await file.read()
    img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    transform = transforms.ToTensor()
    res = process_tensor_inference(proposed_model, transform, img_pil, DEVICE)
    res["method"] = "Proposed (Degradation-Aware FiLM)"
    res["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return res

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

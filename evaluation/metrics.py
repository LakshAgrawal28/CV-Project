import os
import math
import numpy as np
import cv2
from PIL import Image
import torch
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

try:
    import lpips
    _lpips_fn = lpips.LPIPS(net='alex')
    _lpips_fn.eval()
except Exception:
    _lpips_fn = None

def calculate_psnr(img1_np: np.ndarray, img2_np: np.ndarray) -> float:
    """
    Computes PSNR between two RGB images [0, 255] uint8 or [0, 1] float.
    """
    if img1_np.dtype == np.uint8:
        data_range = 255.0
    else:
        data_range = 1.0
    return float(compute_psnr(img2_np, img1_np, data_range=data_range))

def calculate_ssim(img1_np: np.ndarray, img2_np: np.ndarray) -> float:
    """
    Computes SSIM between two RGB images [0, 255] uint8 or [0, 1] float.
    """
    if img1_np.dtype == np.uint8:
        data_range = 255.0
    else:
        data_range = 1.0
    return float(compute_ssim(img1_np, img2_np, channel_axis=2, data_range=data_range))

def calculate_lpips(img1_np: np.ndarray, img2_np: np.ndarray, device: torch.device = None) -> float:
    """
    Computes LPIPS distance using pretrained AlexNet backbone.
    """
    global _lpips_fn
    if _lpips_fn is None:
        return float('nan')
        
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    try:
        _lpips_fn = _lpips_fn.to(device)
        
        # Convert numpy [H, W, C] in [0, 255] to tensor [1, C, H, W] in [-1, 1]
        t1 = torch.from_numpy(img1_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t2 = torch.from_numpy(img2_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        
        t1 = t1.to(device)
        t2 = t2.to(device)
        
        with torch.no_grad():
            dist = _lpips_fn(t1, t2).item()
        return float(dist)
    except Exception as e:
        print(f"LPIPS computation warning: {e}")
        return float('nan')

def calculate_uciqe(img_np: np.ndarray) -> float:
    """
    Computes UCIQE (Underwater Color Image Quality Evaluation).
    Uses Chroma std, Lightness contrast, and Saturation mean in CIELab/HSV spaces.
    c1=0.4680, c2=0.2745, c3=0.2576
    """
    if img_np.dtype != np.uint8:
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    l = l.astype(np.float64)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    # 1. Chroma std
    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)
    
    # 2. Lightness contrast (difference between top 1% and bottom 1% mean)
    l_sorted = np.sort(l.ravel())
    top_1p = np.mean(l_sorted[int(0.99 * len(l_sorted)):])
    bot_1p = np.mean(l_sorted[:int(0.01 * len(l_sorted)) + 1])
    con_l = top_1p - bot_1p
    
    # 3. Saturation mean in HSV space
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float64) / 255.0
    mu_s = np.mean(sat)
    
    uciqe = 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
    return float(uciqe)

def _eme(img_gray: np.ndarray, block_size: int = 8) -> float:
    """Entropy Enhancement Measure helper for UIQM"""
    h, w = img_gray.shape
    h_blocks = h // block_size
    w_blocks = w // block_size
    if h_blocks == 0 or w_blocks == 0:
        return 0.0
        
    eme_val = 0.0
    k = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            sub = img_gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            max_v = np.max(sub)
            min_v = np.min(sub)
            if min_v > 0 and max_v > 0 and max_v != min_v:
                eme_val += 20.0 * np.log(max_v / min_v)
                k += 1
    return eme_val / max(k, 1)

def calculate_uiqm(img_np: np.ndarray) -> float:
    """
    Computes UIQM (Underwater Image Quality Measure).
    Combines UICM (Colorfulness), UISM (Sharpness), and UIConM (Contrast).
    c1=0.0282, c2=0.2953, c3=3.5753
    """
    if img_np.dtype != np.uint8:
        img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
        
    r = img_np[:, :, 0].astype(np.float64)
    g = img_np[:, :, 1].astype(np.float64)
    b = img_np[:, :, 2].astype(np.float64)
    
    # 1. UICM - Underwater Image Colorfulness Measure
    rg = r - g
    yb = 0.5 * (r + g) - b
    
    mu_rg = np.mean(rg)
    sig_rg = np.std(rg)
    mu_yb = np.mean(yb)
    sig_yb = np.std(yb)
    
    uicm = -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(sig_rg**2 + sig_yb**2)
    
    # 2. UISM - Underwater Image Sharpness Measure via EME on Sobel edges
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sobel_r = cv2.Sobel(r, cv2.CV_64F, 1, 1)
    sobel_g = cv2.Sobel(g, cv2.CV_64F, 1, 1)
    sobel_b = cv2.Sobel(b, cv2.CV_64F, 1, 1)
    
    uism = 0.299 * _eme(np.abs(sobel_r)) + 0.587 * _eme(np.abs(sobel_g)) + 0.114 * _eme(np.abs(sobel_b))
    
    # 3. UIConM - Underwater Image Contrast Measure
    uiconm = _eme(gray.astype(np.float64))
    
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    return float(uiqm)

def evaluate_pair(pred_img_path: str, gt_img_path: str = None) -> dict:
    """
    Evaluates a single prediction image against optional ground truth reference.
    """
    pred_pil = Image.open(pred_img_path).convert("RGB")
    pred_np = np.array(pred_pil)
    
    results = {
        "uiqm": calculate_uiqm(pred_np),
        "uciqe": calculate_uciqe(pred_np)
    }
    
    if gt_img_path and os.path.exists(gt_img_path):
        gt_pil = Image.open(gt_img_path).convert("RGB").resize(pred_pil.size)
        gt_np = np.array(gt_pil)
        
        results["psnr"] = calculate_psnr(pred_np, gt_np)
        results["ssim"] = calculate_ssim(pred_np, gt_np)
        results["lpips"] = calculate_lpips(pred_np, gt_np)
    else:
        results["psnr"] = float('nan')
        results["ssim"] = float('nan')
        results["lpips"] = float('nan')
        
    return results

if __name__ == "__main__":
    test_img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    print("Testing Metrics Module:")
    print("UIQM :", calculate_uiqm(test_img))
    print("UCIQE:", calculate_uciqe(test_img))
    print("PSNR :", calculate_psnr(test_img, test_img))
    print("SSIM :", calculate_ssim(test_img, test_img))

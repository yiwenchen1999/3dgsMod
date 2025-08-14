#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def list_latents(folder):
    p = Path(folder)
    return sorted([f for f in p.rglob("*") if f.suffix.lower() == ".npy"])

def blur_latent(latent: np.ndarray, sigma: float):
    """
    latent: (4,h,w) or (1,4,h,w), any float dtype
    - Upcast to fp32 for SciPy
    - Gaussian blur per-channel
    - Return in original dtype
    """
    orig_dtype = latent.dtype
    if latent.ndim == 4:   # (1,4,h,w) -> (4,h,w)
        latent = latent.squeeze(0)
    x32 = latent.astype(np.float32, copy=False)

    blurred32 = np.empty_like(x32)
    for c in range(x32.shape[0]):
        blurred32[c] = gaussian_filter(x32[c], sigma=sigma)

    # cast back to original dtype (e.g., float16)
    return blurred32.astype(orig_dtype, copy=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Folder with .npy latents")
    parser.add_argument("--output", required=True, help="Folder to save blurred .npy latents")
    parser.add_argument("--sigma", type=float, default=1.0, help="Gaussian blur sigma")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_latents(in_dir)
    if not files:
        print(f"No .npy latents found in {in_dir}")
        return

    for f in tqdm(files, desc="Blurring"):
        rel = f.relative_to(in_dir)
        out_path = (out_dir / rel).with_suffix(".npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            continue

        lat = np.load(f)  # shape (4,h,w) or (1,4,h,w), dtype often float16
        blurred = blur_latent(lat, args.sigma)
        np.save(out_path, blurred)

    print(f"Done. Blurred latents saved to: {out_dir}")

if __name__ == "__main__":
    main()

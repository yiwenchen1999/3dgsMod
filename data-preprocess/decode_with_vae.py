#!/usr/bin/env python3
import argparse, os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL

SCALE = 0.18215

def list_latents(folder):
    p = Path(folder)
    return sorted([f for f in p.rglob("*") if f.suffix.lower() == ".npy"])

def load_vae(vae_repo: str = None, sd_repo: str = None, device="cuda", dtype=torch.float16):
    # Default: SD 1.5's bundled VAE
    if vae_repo:
        vae = AutoencoderKL.from_pretrained(vae_repo, torch_dtype=dtype)
    elif sd_repo:
        vae = AutoencoderKL.from_pretrained(sd_repo, subfolder="vae", torch_dtype=dtype)
    else:
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=dtype)
    vae.to(device).eval()
    return vae

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = (t.clamp(-1, 1) + 1) / 2
    t = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    t = (t * 255).round().astype("uint8")
    return Image.fromarray(t)

@torch.inference_mode()
def decode_latents(vae: AutoencoderKL, latents: torch.Tensor):
    # Expect latents already in "scaled" space, divide out SCALE then decode
    latents = latents / SCALE
    return vae.decode(latents).sample  # (1,3, 8*h, 8*w) in [-1,1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input folder with .npy latents (square or *_og8)")
    parser.add_argument("--output", required=True, help="Output folder for decoded images")
    parser.add_argument("--vae", default=None, help="Optional VAE repo/path")
    parser.add_argument("--sd", default=None, help="Optional SD repo/path; uses /vae")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", help="Force float16 inference")
    parser.add_argument("--ext", default=".png", choices=[".png", ".jpg", ".webp"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    latents_files = list_latents(args.input)
    if not latents_files:
        print(f"No .npy latents found in {args.input}")
        return

    dtype = torch.float16 if args.device.startswith("cuda") or args.fp16 else torch.float32
    vae = load_vae(args.vae, args.sd, device=args.device, dtype=dtype)

    for lat_path in tqdm(latents_files, desc="Decoding"):
        rel = Path(lat_path).relative_to(args.input)
        out_img = (Path(args.output) / rel).with_suffix(args.ext)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        if out_img.exists() and not args.overwrite:
            continue

        arr = np.load(lat_path)  # (4,h,w) — h,w can be square size/8 or og H/8, W/8
        lat = torch.from_numpy(arr).to(args.device, dtype=dtype)
        print(lat.shape)
        if lat.dim() == 3:
            lat = lat.unsqueeze(0)  # (1,4,h,w)

        img_tensor = decode_latents(vae, lat)   # (1,3, 8*h, 8*w)
        img = tensor_to_pil(img_tensor)
        img.save(out_img)

    print(f"Done. Decoded images saved to: {args.output}")

if __name__ == "__main__":
    main()

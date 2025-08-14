#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKL

SCALE = 0.18215
def match_hw(z, target_hw):
    # z: (B,C,H,W) -> (B,C,target_h,target_w)
    return F.interpolate(z, size=target_hw, mode="bilinear", align_corners=False)

@torch.no_grad()
def slerp(z0, z1, alpha, per_pixel=False, eps=1e-7):
    """
    z0, z1: (B, C, H, W) tensors (same shape)
    alpha: float in [0,1] or tensor broadcastable to (B,1,1,1)
    per_pixel=False -> slerp over flattened (C*H*W) vector per item
    per_pixel=True  -> slerp over channel vector at each (h,w)
    """
    assert z0.shape == z1.shape, "Shapes must match. Resize first if needed."
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=z0.device, dtype=z0.dtype)
    # Make alpha broadcastable
    while alpha.dim() < z0.dim():
        alpha = alpha.view(*(alpha.shape + (1,)))

    if per_pixel:
        # Normalize over channels at each pixel
        # norms: (B,1,H,W)
        n0 = z0.norm(dim=1, keepdim=True) + eps
        n1 = z1.norm(dim=1, keepdim=True) + eps
        v0 = z0 / n0
        v1 = z1 / n1
        dot = (v0 * v1).sum(dim=1, keepdim=True).clamp(-1+eps, 1-eps)  # (B,1,H,W)
        theta = torch.acos(dot)
        sin_t = torch.sin(theta) + eps
        w0 = torch.sin((1 - alpha) * theta) / sin_t
        w1 = torch.sin(alpha * theta) / sin_t
        z = w0 * z0 + w1 * z1
    else:
        # Global slerp across all dims except batch
        B = z0.shape[0]
        z0f = z0.flatten(1)  # (B, C*H*W)
        z1f = z1.flatten(1)
        n0 = z0f.norm(dim=1, keepdim=True) + eps
        n1 = z1f.norm(dim=1, keepdim=True) + eps
        v0 = z0f / n0
        v1 = z1f / n1
        dot = (v0 * v1).sum(dim=1, keepdim=True).clamp(-1+eps, 1-eps)  # (B,1)

        theta = torch.acos(dot)
        sin_t = torch.sin(theta) + eps
        # shape to broadcast: (B,1,1,1)
        w0 = (torch.sin((1 - alpha) * theta) / sin_t).view(B,1,1,1)
        w1 = (torch.sin(alpha * theta) / sin_t).view(B,1,1,1)
        z = (w0 * z0) + (w1 * z1)
    return z

def list_images(folder, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
    p = Path(folder)
    return sorted([f for f in p.rglob("*") if f.suffix.lower() in exts])

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

def pil_to_tensor(image: Image.Image, size: int):
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if w == h:
        resized = image.resize((size, size), Image.BICUBIC)
    else:
        if w < h:
            new_w, new_h = size, int(round(h * size / w))
        else:
            new_w, new_h = int(round(w * size / h)), size
        resized = image.resize((new_w, new_h), Image.BICUBIC)
        left = (resized.width - size) // 2
        top = (resized.height - size) // 2
        resized = resized.crop((left, top, left + size, top + size))
    t = TF.to_tensor(resized) * 2.0 - 1.0  # [0,1] -> [-1,1]
    return t.unsqueeze(0)  # (1,3,H,W)

@torch.inference_mode()
def encode_image(vae: AutoencoderKL, img_tensor: torch.Tensor, sample: bool = False):
    posterior = vae.encode(img_tensor).latent_dist
    latents = posterior.sample() if sample else posterior.mean
    return latents * SCALE  # (1,4,H/8,W/8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output folder for .npy latents")
    parser.add_argument("--vae", default=None, help="Optional VAE repo/path")
    parser.add_argument("--sd", default=None, help="Optional SD repo/path; loads its /vae")
    parser.add_argument("--size", type=int, default=512, help="Square size after resize+crop")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", help="Force float16 inference/storage")
    parser.add_argument("--sample", action="store_true", help="Sample from latent distribution")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_meta", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    images = list_images(args.input)
    if not images:
        print(f"No images found in {args.input}")
        return

    dtype = torch.float16 if args.device.startswith("cuda") or args.fp16 else torch.float32
    vae = load_vae(args.vae, args.sd, device=args.device, dtype=dtype)

    for img_path in tqdm(images, desc="Encoding"):
        rel = Path(img_path).relative_to(args.input)
        out_base = Path(args.output) / rel
        out_base.parent.mkdir(parents=True, exist_ok=True)
        out_latent = out_base.with_suffix(".npy")
        out_latent_og8 = out_base.with_name(out_base.stem + "_down").with_suffix(".npy")

        if out_latent.exists() and not args.overwrite:
            # still create og8 if missing
            if out_latent_og8.exists():
                continue

        img = Image.open(img_path)
        W0, H0 = img.size  # original image size BEFORE our resize/crop

        # Encode (square path)
        x = pil_to_tensor(img, size=args.size).to(args.device, dtype=dtype)
        latents = encode_image(vae, x, sample=args.sample).squeeze(0)  # (4,h,w) on device

        # Save base latent (square size/8)
        lat_cpu = latents.to("cpu")
        np.save(out_latent, lat_cpu.half().numpy() if dtype == torch.float16 else lat_cpu.numpy())

        # --- Interpolate latent to original image's latent-scale (≈ H0/8, W0/8) ---
        h_og8 = H0//64
        w_og8 = W0//64
        # lat_og8 = F.interpolate(
        #     latents.unsqueeze(0).float(), size=(h_og8, w_og8), mode="nearest"
        # ).squeeze(0).to("cpu")
        # # save in same dtype policy
        # np.save(out_latent_og8, lat_og8.half().numpy() if dtype == torch.float16 else lat_og8.numpy())

        if args.save_meta:
            meta = {
                "source": str(Path(img_path)),
                "orig_size_hw": [H0, W0],
                "size_input": args.size,
                "latent_shape_square": list(lat_cpu.shape),
                "latent_shape_og8": [4, h_og8, w_og8],
                "dtype": "float16" if dtype == torch.float16 else "float32",
                "scale_factor": SCALE,
                "mode": "sample" if args.sample else "mean",
                "vae": args.vae or (args.sd + "/vae" if args.sd else "runwayml/stable-diffusion-v1-5/vae"),
            }
            with open(out_base.with_suffix(".json"), "w") as f:
                json.dump(meta, f, indent=2)

    print(f"Done. Saved .npy latents (square and _og8) to: {args.output}")

if __name__ == "__main__":
    main()

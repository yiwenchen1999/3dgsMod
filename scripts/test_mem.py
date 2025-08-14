# gs_mem_sanity.py
# Minimal, robust memory sanity test for your Gaussian rasterizer.

import torch
from gaussian_renderer import GaussianRasterizationSettings, GaussianRasterizer

def finite(name, t):
    if torch.isnan(t).any() or torch.isinf(t).any():
        raise RuntimeError(f"{name} has NaN/Inf")

def mem_tag(tag):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024**2)
    peak  = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"[{tag}] cuda_alloc={alloc:.2f} MiB  peak={peak:.2f} MiB")

def make_settings(H=100, W=100):
    device, dtype = torch.device("cuda"), torch.float32
    # NOTE: your settings expect plain floats for tanfovx/y and a float for scale_modifier
    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=0.5,                  # ~53° half-FoV → sane
        tanfovy=0.5,
        bg=torch.tensor([0.,0.,0.], device=device, dtype=dtype).contiguous(),
        scale_modifier=1.0,
        viewmatrix=torch.eye(4, device=device, dtype=dtype).contiguous(),
        projmatrix=(torch.eye(4, device=device, dtype=dtype).contiguous().index_put_((torch.tensor([2],device=device), torch.tensor([3],device=device)), torch.tensor([-1.0],device=device))),
        sh_degree=0,
        campos=torch.tensor([0.,0.,0.], device=device, dtype=dtype).contiguous(),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    # quick sanity on tensors in settings
    finite("bg", settings.bg)
    finite("view", settings.viewmatrix)
    finite("proj", settings.projmatrix)
    finite("campos", settings.campos)
    return settings

def make_tiny_batch(N=1024):
    device, dtype = torch.device("cuda"), torch.float32
    means3D = torch.tensor([[0.0, 0.0, 3.0]], device=device, dtype=dtype).repeat(N,1).contiguous()
    means2D = torch.zeros(N, 2, device=device, dtype=dtype).contiguous()  # ignored in 3D path but required by API
    opac    = torch.full((N,), 0.5, device=device, dtype=dtype).contiguous()
    colors  = torch.ones(N, 3, device=device, dtype=dtype).contiguous()
    shs     = torch.zeros(N, 1, 3, device=device, dtype=dtype).contiguous()  # degree=0 → 1 coef
    shs[:,0,:] = 1.0

    scales  = torch.full((N,3), 1e-2, device=device, dtype=dtype).contiguous()
    rots    = torch.zeros(N,4, device=device, dtype=dtype).contiguous()
    rots[:,0] = 1.0  # identity quaternion [w,x,y,z]

    cov3D   = torch.diag_embed(torch.tensor([[1e-4,1e-4,1e-4]], device=device, dtype=dtype)).repeat(N,1,1).contiguous()

    for n,t in [("means3D",means3D),("means2D",means2D),("opac",opac),("colors",colors),("shs",shs),("scales",scales),("rots",rots),("cov3D",cov3D)]:
        finite(n,t)
    return means3D, means2D, opac, colors, shs, scales, rots, cov3D

def run_case(tag, rast, **kwargs):
    try:
        torch.cuda.reset_peak_memory_stats()
        out = rast(**kwargs)
        # expected: color, radii, invdepths
        if isinstance(out, (tuple, list)):
            color, radii, invdepths = out
        else:
            color, radii, invdepths = out, None, None
        mem_tag(tag)
        if torch.is_tensor(color):
            print(f"[{tag}] color shape: {tuple(color.shape)}")
        if torch.is_tensor(radii):
            print(f"[{tag}] radii shape: {tuple(radii.shape)}")
        if torch.is_tensor(invdepths):
            print(f"[{tag}] invdepths shape: {tuple(invdepths.shape)}")
    except torch.cuda.OutOfMemoryError as e:
        mem_tag(tag + " (OOM)")
        print(f"[{tag}] OOM: {e}")
    except Exception as e:
        print(f"[{tag}] ERROR: {type(e).__name__}: {e}")

def main():
    torch.cuda.init()
    settings = make_settings(H=16, W=16)
    rasterizer = GaussianRasterizer(settings).to("cuda")
    means3D, means2D, opac, colors, shs, scales, rots, cov3D = make_tiny_batch(N=1024)

    print("=== CASE A: colors_precomp + cov3D_precomp (no SHs) ===")
    run_case(
        "A",
        rasterizer,
        means3D=means3D,
        means2D=means2D,
        opacities=opac,
        shs=None,
        colors_precomp=colors,
        scales=None,
        rotations=None,
        cov3D_precomp=cov3D,
    )

    print("\n=== CASE B: colors_precomp + scales/rotations (no cov3D_precomp) ===")
    run_case(
        "B",
        rasterizer,
        means3D=means3D,
        means2D=means2D,
        opacities=opac,
        shs=None,
        colors_precomp=colors,
        scales=scales,
        rotations=rots,
        cov3D_precomp=None,
    )

    print("\n=== CASE C: SHs + cov3D_precomp (no colors_precomp) ===")
    run_case(
        "C",
        rasterizer,
        means3D=means3D,
        means2D=means2D,
        opacities=opac,
        shs=shs,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=cov3D,
    )

if __name__ == "__main__":
    main()

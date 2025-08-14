#!/usr/bin/env python3
import argparse, json, math, os, re, random
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image  # to read image size

def load_json(p: Path):
    with p.open("r") as f:
        return json.load(f)

def save_json(p: Path, data: Dict[str, Any]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(data, f, indent=2)

def fov_to_intrinsics(w: int, h: int, fov_deg: float, fov_axis: str = "horizontal"):
    fov_rad = math.radians(fov_deg)
    if fov_axis == "horizontal":
        fx = 0.5 * w / math.tan(0.5 * fov_rad)
        fy = fx
    elif fov_axis == "vertical":
        fy = 0.5 * h / math.tan(0.5 * fov_rad)
        fx = fy
    else:
        raise ValueError("fov_axis must be 'horizontal' or 'vertical'")
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    return fx, fy, cx, cy, fov_rad

def mat_mult4(a, b):
    return [[sum(a[r][k]*b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]

def apply_coord_conv(c2w: List[List[float]], mode: str) -> List[List[float]]:
    if mode == "passthrough":
        return c2w
    elif mode == "opencv_to_nerf":
        M = [
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1],
        ]
        return mat_mult4(c2w, M)
    else:
        raise ValueError(f"Unknown coord-conv mode: {mode}")

def find_envs(scene_dir: Path) -> List[Tuple[Path, Path]]:
    envs = []
    for env_dir in sorted(scene_dir.glob("white_env_*")):
        if not env_dir.is_dir():
            continue
        cam_json = scene_dir / f"{env_dir.name}_cameras.json"
        if cam_json.exists():
            envs.append((env_dir, cam_json))
    return envs

def collect_frames(env_dir: Path, cam_meta: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    frames = {}
    meta_by_eye = {int(entry["eye_idx"]): entry for entry in cam_meta}
    for img in sorted(env_dir.glob("gt_*.png")):
        m = re.match(r"gt_(\d+)\.png$", img.name)
        if not m:
            continue
        eye = int(m.group(1))
        if eye not in meta_by_eye:
            print(f"[WARN] {img} has no camera entry; skipping.")
            continue
        entry = meta_by_eye[eye]
        frames[eye] = {
            "file_path": str(img.relative_to(env_dir.parent).as_posix()),
            "c2w": entry["c2w"],
            "fov": float(entry["fov"]),
        }
    return frames

def split_frames(frames: List[Dict[str, Any]], train_ratio: float, shuffle: bool, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    frames_sorted = sorted(frames, key=lambda f: f.get("_eye_idx", 0))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(frames_sorted)
    n = len(frames_sorted)
    if n == 0:
        return [], []
    k = max(1, int(round(train_ratio * n))) if n >= 5 else max(1, n - 1)  # keep at least 1 for test when possible
    k = min(k, n)  # avoid overflow
    train = frames_sorted[:k]
    test = frames_sorted[k:] if k < n else []
    return train, test

def main():
    ap = argparse.ArgumentParser(description="Convert synthetic scene to NeRF transforms.json with train/test split")
    ap.add_argument("scene_dir", type=str, help="Path to a single scene folder OR a parent (use --recursive)")
    ap.add_argument("--recursive", action="store_true", help="Process all child scene folders under scene_dir")
    ap.add_argument("--coord-conv", type=str, default="passthrough", choices=["passthrough", "opencv_to_nerf"])
    ap.add_argument("--fov-axis", type=str, default="horizontal", choices=["horizontal", "vertical"])
    ap.add_argument("--emit-intrinsics", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--output-in-env", action="store_true", help="Write JSONs inside each env dir instead of scene root")

    # NEW: split controls
    ap.add_argument("--train-ratio", type=float, default=0.8, help="Fraction for training split (default: 0.8)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle frames before splitting")
    ap.add_argument("--seed", type=int, default=42, help="Seed used when --shuffle is set")

    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)

    # Determine targets (single scene vs recursive)
    def is_scene_dir(p: Path) -> bool:
        for env_dir in p.glob("white_env_*"):
            if env_dir.is_dir() and (p / f"{env_dir.name}_cameras.json").exists():
                return True
        return False

    def list_child_scenes(root: Path):
        return [d for d in sorted(root.iterdir()) if d.is_dir() and is_scene_dir(d)]

    targets = []
    if args.recursive and scene_dir.is_dir() and not is_scene_dir(scene_dir):
        targets = list_child_scenes(scene_dir)
        if not targets:
            raise SystemExit(f"No scene folders found under {scene_dir}")
    else:
        targets = [scene_dir]

    for sc in targets:
        print(f"[SCENE] {sc}")
        envs = find_envs(sc)
        if not envs:
            print(f"[WARN] No envs in {sc}; skipping.")
            continue

        for env_dir, cam_json in envs:
            cam_meta = load_json(cam_json)
            frames_map = collect_frames(env_dir, cam_meta)
            if not frames_map:
                print(f"[WARN] No frames found for {env_dir.name}; skipping.")
                continue

            # Resolve image size
            first = next(iter(frames_map.values()))
            # If file_path is like 'white_env_X/gt_0.png', anchor resolution by joining with scene root
            img_abs = (sc / first["file_path"]).resolve()
            with Image.open(img_abs) as im:
                w, h = im.size

            # FOV handling (top-level)
            fovs = {f["fov"] for f in frames_map.values()}
            any_fov = next(iter(fovs))
            fx, fy, cx, cy, cam_angle_x = fov_to_intrinsics(w, h, any_fov, args.fov_axis)

            # Build frames with coord conversion + scale
            frames_all: List[Dict[str, Any]] = []
            for eye in sorted(frames_map.keys()):
                rec = frames_map[eye]
                c2w = rec["c2w"]
                c2w_scaled = [row[:] for row in c2w]
                c2w_scaled[0][3] *= args.scale
                c2w_scaled[1][3] *= args.scale
                c2w_scaled[2][3] *= args.scale
                c2w_final = apply_coord_conv(c2w_scaled, args.coord_conv)

                entry = {
                    "file_path": rec["file_path"],
                    "transform_matrix": c2w_final,
                    "_eye_idx": eye,  # helper for stable sorting/splitting
                }
                if len(fovs) > 1:
                    _, _, _, _, ang_x = fov_to_intrinsics(w, h, rec["fov"], args.fov_axis)
                    entry["camera_angle_x"] = ang_x
                frames_all.append(entry)

            # Compose base header
            base_header = {
                "camera_angle_x": cam_angle_x,
            }
            if args.emit_instruments if False else False:  # keeps IDEs from folding; real flag below
                pass
            if args.emit_intrinsics:
                base_header.update({"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "w": w, "h": h})

            # Write full transforms
            out_all = dict(base_header)
            out_all["frames"] = [{k: v for k, v in f.items() if k != "_eye_idx"} for f in frames_all]

            # Decide output paths
            if args.output_in_env:
                base = env_dir
                suffix = ""
            else:
                base = sc
                suffix = f""

            path_all  = base / f"transforms{suffix}.json"
            save_json(path_all, out_all)
            print(f"[OK] Wrote {path_all}")

            # Train/Test split
            train_frames, test_frames = split_frames(frames_all, args.train_ratio, args.shuffle, args.seed)

            def write_split(name: str, frames_subset: List[Dict[str, Any]]):
                if not frames_subset:
                    return None
                out = dict(base_header)
                out["frames"] = [{k: v for k, v in f.items() if k != "_eye_idx"} for f in frames_subset]
                p = base / f"transforms_{name}{suffix}.json"
                save_json(p, out)
                print(f"[OK] Wrote {p}")
                return p

            write_split("train", train_frames)
            write_split("test",  test_frames)

if __name__ == "__main__":
    main()

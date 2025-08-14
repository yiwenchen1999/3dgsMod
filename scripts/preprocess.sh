cd ../data-preprocess

# Process all immediate subfolders in ../datasamples
for d in ../datasamples/*/; do
  echo "[RUN] $d"
  python make_nerf_transforms.py "$d" --emit-intrinsics
done

# # If your c2w is OpenCV-style (+z forward, +y down), convert to NeRF/Blender:
# python make_nerf_transforms.py /path/to/scene --coord-conv opencv_to_nerf

# # Emit intrinsics fields for instant-ngp/nerfstudio:
# python make_nerf_transforms.py /path/to/scene --emit-intrinsics

# # If your JSON fov is vertical:
# python make_nerf_transforms.py /path/to/scene --fov-axis vertical

# # Write transforms.json inside each env folder:
# python make_nerf_transforms.py /path/to/scene --output-in-env

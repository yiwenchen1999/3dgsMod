cd ../LangSplat
python train.py -s ../nerf_synthetic/ship_latents -m output/ship_latents --start_checkpoint ../nerf_synthetic/ship_latents/output/ship_latents/chkpnt30000.pth --feature_level 1 --eval
python render.py -s ../nerf_synthetic/ship_latents -m output/ship_latents_1 --feature_level 1
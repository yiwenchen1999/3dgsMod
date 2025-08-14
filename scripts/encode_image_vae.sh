cd ../data-preprocess
python encode_with_vae.py \
--input ../nerf_synthetic/ship_latents/images \
--output ../nerf_synthetic/ship_latents/vae_latents

# python apply_gaussian.py --input ../nerf_synthetic/ship_latents/vae_latents --output ../nerf_synthetic/ship_latents/latents_gaussian_3r --sigma 4.0

# python decode_with_vae.py \
# --input ../nerf_synthetic/ship_latents/vae_latents_ogsize \
# --output ../nerf_synthetic/ship_latents/vae_decoded_images_ogsize

python decode_with_vae.py \
--input ../nerf_synthetic/ship_latents/vae_latents \
--output ../nerf_synthetic/ship_latents/vae_decoded_images
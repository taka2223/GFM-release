"""
Real Image GFM Interpolation with EDM2.

Pipeline: real image → VAE encode → add noise → GFM optimize → denoise → VAE decode

Usage:
    python edm2_real_interp.py \
        --imgA /path/to/dog.jpg --imgB /path/to/cat.jpg \
        --preset edm2-img512-s-guid-dino \
        --noise_level 0.5 --lam 1.0
"""

import os
import sys
sys.path.insert(0, '/export_home/zzhixuan/code/GFM-release/edm2')
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Dict, Any
from tqdm import tqdm
from diffusers.models import AutoencoderKL

from gfm import dnnlib
dnnlib.util.set_cache_dir('/cns/USERS/zzhixuan/weights/edm2')

from gfm.path.edm import (
    EDM2GFMPipeline,
    EDM2ModelWrapper,
    EDM2Autoencoder,
    edm_sampler,
    denoise_path_cross_class,
    save_interpolation_strip,
    save_frames,
)


# ============================================================
# 1. Real image encode/decode with EDM2's VAE convention
# ============================================================

class RealImageEncoder:
    """Handles real image → EDM2 latent space conversion.
    
    EDM2 img512 models are trained on VAE latents stored as
    mean-std pairs [8, H, W] where first 4 channels = mean, last 4 = std.
    At inference, we just use the mean (deterministic encoding).
    
    Scaling: EDM2 trains on raw VAE latents (no 0.18215 scaling).
    The VAE encoder outputs are used directly.
    """

    def __init__(self, vae, device: str = "cuda"):
        self.vae = vae
        self.device = device
        # EDM2's dataset_tool.py uses StabilityVAEEncoder which does NOT
        # apply the 0.18215 scaling. The raw VAE output is used directly.
        # But the standard VAE decode expects unscaled input.
        # We need to match whatever convention the model was trained with.
        self.scaling_factor = 0.18215  # SD VAE standard

    def encode_image(self, image: Image.Image, resolution: int = 512) -> torch.Tensor:
        """Encode a real image to VAE latent.
        
        Args:
            image: PIL Image
            resolution: target resolution (must match model training)
            
        Returns:
            latent: [1, 4, H//8, W//8] VAE latent
        """
        # Preprocess: resize + center crop + normalize to [-1, 1]
        image = self._center_crop_resize(image, resolution)
        arr = np.array(image).astype(np.float32) / 255.0
        arr = 2.0 * arr - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=self.vae.dtype)

        with torch.no_grad():
            posterior = self.vae.encode(tensor).latent_dist
            latent = posterior.mean  # deterministic encoding
        
        return latent * self.scaling_factor

    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        """Decode VAE latent to PIL Image."""
        with torch.no_grad():
            img_tensor = self.vae.decode(
                latent.to(self.vae.dtype) / self.scaling_factor
            ).sample
        arr = img_tensor.squeeze(0).float().permute(1, 2, 0).cpu().numpy()
        arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def _center_crop_resize(self, image: Image.Image, size: int) -> Image.Image:
        """Center crop to square, then resize. Matches ADM/EDM2 preprocessing."""
        image = image.convert("RGB")
        w, h = image.size
        crop = min(w, h)
        left = (w - crop) // 2
        top = (h - crop) // 2
        image = image.crop((left, top, left + crop, top + crop))
        image = image.resize((size, size), Image.LANCZOS)
        return image


# ============================================================
# 2. Real image interpolation pipeline
# ============================================================

def run_real_image_interpolation(
    pipe: EDM2GFMPipeline,
    imgA: Image.Image,
    imgB: Image.Image,
    classA: int = 207,
    classB: int = 281,
    noise_level: float = 0.5,
    num_steps: int = 10,
    lam: float = 1.0,
    lr: float = 0.01,
    max_iters: int = 800,
    interpolator_type: str = "el",
    num_denoise_steps: int = 18,
    resolution: int = 512,
) -> Dict[str, Any]:
    """
    GFM interpolation between two real images.
    
    Pipeline:
        1. Encode real images → VAE latents
        2. Verify reconstruction quality
        3. Add EDM noise (x_σ = x₀ + σ·ε)
        4. GFM path optimization with per-frame soft class labels
        5. Per-frame denoise with interpolated class conditioning
        6. VAE decode → output images
    
    Args:
        pipe: loaded EDM2GFMPipeline
        imgA, imgB: PIL Images (will be center-cropped and resized)
        classA, classB: ImageNet class indices for conditioning
            Use the closest matching class for each image.
            Common classes: 207=golden retriever, 281=tabby cat, 1=goldfish,
            153=Maltese, 229=Old English sheepdog, 248=husky
        noise_level: EDM sigma for noise injection
        num_steps: path waypoints
        lam: force strength in EL equation
        interpolator_type: "el", "seq_el", or "spherical"
    """
    from gfm.path.geodesic_interpolation import (
        ELInterpolator,
        SeqELInterpolator,
        SphericalInterpolator,
    )

    device = pipe.device
    encoder = RealImageEncoder(pipe.autoencoder.vae, device=str(device))

    # ---- Step 1: Encode ----
    print("Encoding real images...")
    latA = encoder.encode_image(imgA, resolution=resolution)
    latB = encoder.encode_image(imgB, resolution=resolution)
    print(f"  Latent shape: {latA.shape}")
    print(f"  Latent A: mean={latA.mean():.4f}, std={latA.std():.4f}, norm={latA.view(-1).norm():.2f}")
    print(f"  Latent B: mean={latB.mean():.4f}, std={latB.std():.4f}, norm={latB.view(-1).norm():.2f}")

    # ---- Step 2: Verify reconstruction ----
    print("Verifying VAE reconstruction...")
    reconA = encoder.decode_latent(latA)
    reconB = encoder.decode_latent(latB)

    # ---- Step 3: Set up interpolation labels ----
    pipe.model.set_interpolation_labels(classA, classB, num_steps)

    # ---- Step 4: Select interpolator ----
    if interpolator_type == "spherical":
        interpolator = SphericalInterpolator(pipe.model, pipe.autoencoder, device=str(device))
    elif interpolator_type == "el":
        interpolator = ELInterpolator(pipe.model, pipe.autoencoder, device=str(device))
    elif interpolator_type == "seq_el":
        interpolator = SeqELInterpolator(pipe.model, pipe.autoencoder, device=str(device))
    else:
        raise ValueError(f"Unknown interpolator: {interpolator_type}")

    # ---- Step 5: Add noise (EDM: x_σ = x₀ + σ·ε) ----
    print(f"Adding noise (sigma={noise_level})...")
    noisy_latents = interpolator.add_noise(
        torch.cat([latA, latB], dim=0), sigma=noise_level
    )
    print(f"  Noisy norm A: {noisy_latents[0].view(-1).norm():.2f}")
    print(f"  Noisy norm B: {noisy_latents[1].view(-1).norm():.2f}")

    # ---- Step 6: Optimize path ----
    print(f"Running GFM ({interpolator_type}, steps={num_steps}, λ={lam}, iters={max_iters})...")
    dummy_label = torch.tensor([classA], device=device)

    if interpolator_type == "spherical":
        path = interpolator.optimize_path(
            start_latent=noisy_latents[0:1],
            end_latent=noisy_latents[1:2],
            init_with_slerp=False,
            num_steps=num_steps,
        )
        info = {}
    else:
        path, info = interpolator.optimize_path(
            start_latent=noisy_latents[0:1],
            end_latent=noisy_latents[1:2],
            num_steps=num_steps,
            sigma=noise_level,
            lr=lr,
            max_iters=max_iters,
            lam=lam,
            class_label=dummy_label,
            verbose=True,
        )

    if isinstance(path, tuple):
        path = path[0]

    # ---- Step 7: Denoise with per-frame interpolated labels ----
    pipe.model.clear_interpolation_labels()

    print(f"Denoising path ({num_denoise_steps} steps, per-frame soft labels)...")
    clean_path = denoise_path_cross_class(
        net=pipe.model.net,
        gnet=pipe.model.gnet,
        guidance=pipe.model.guidance,
        path=path,
        classA=classA,
        classB=classB,
        sigma_start=noise_level,
        num_denoise_steps=num_denoise_steps,
        label_dim=pipe.model.label_dim,
    )

    # ---- Step 8: Decode ----
    print("Decoding to images...")
    images = []
    # First frame: use original image (not VAE round-trip)
    images.append(imgA.convert("RGB").resize((resolution, resolution), Image.LANCZOS))
    
    for i in range(1, clean_path.shape[1] - 1):
        images.append(encoder.decode_latent(clean_path[0, i:i+1]))
    
    # Last frame: use original image
    images.append(imgB.convert("RGB").resize((resolution, resolution), Image.LANCZOS))

    return {
        "images": images,
        "latents_noisy": path,
        "latents_clean": clean_path,
        "losses": info.get("losses", []) if isinstance(info, dict) else [],
        "reconA": reconA,
        "reconB": reconB,
        "classA": classA,
        "classB": classB,
    }


# ============================================================
# 3. Entry point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real Image GFM with EDM2")
    # Images
    parser.add_argument("--imgA", type=str, required=True, help="Path to image A")
    parser.add_argument("--imgB", type=str, required=True, help="Path to image B")
    
    # Class conditioning (closest ImageNet class for each image)
    parser.add_argument("--classA", type=int, default=207,
                        help="ImageNet class for image A (207=golden retriever)")
    parser.add_argument("--classB", type=int, default=281,
                        help="ImageNet class for image B (281=tabby cat)")

    # Model
    parser.add_argument("--preset", type=str, default="edm2-img512-s-guid-dino")
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--gnet", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--vae", type=str, default="mse", choices=["mse", "ema"])
    parser.add_argument("--resolution", type=int, default=512)

    # GFM parameters
    parser.add_argument("--noise_level", type=float, default=0.5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=800)
    parser.add_argument("--interpolator", type=str, default="el",
                        choices=["spherical", "el", "seq_el"])
    parser.add_argument("--num_denoise_steps", type=int, default=18)

    # Output
    parser.add_argument("--output_dir", type=str, default="./output/edm2_real_interp")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Load pipeline
    extra = {}
    if args.guidance is not None:
        extra['guidance'] = args.guidance

    print(f"Loading EDM2 pipeline (preset={args.preset})...")
    pipe = EDM2GFMPipeline.load(
        preset=args.preset,
        net_path=args.net,
        gnet_path=args.gnet,
        vae_type=args.vae,
        device=args.device,
        **extra,
    )

    # Load images
    print(f"Loading images...")
    imgA = Image.open(args.imgA)
    imgB = Image.open(args.imgB)
    print(f"  Image A: {imgA.size} from {args.imgA}")
    print(f"  Image B: {imgB.size} from {args.imgB}")

    # Run interpolation
    results = run_real_image_interpolation(
        pipe,
        imgA=imgA,
        imgB=imgB,
        classA=args.classA,
        classB=args.classB,
        noise_level=args.noise_level,
        num_steps=args.num_steps,
        lam=args.lam,
        lr=args.lr,
        max_iters=args.max_iters,
        interpolator_type=args.interpolator,
        num_denoise_steps=args.num_denoise_steps,
        resolution=args.resolution,
    )

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    save_interpolation_strip(
        results["images"],
        os.path.join(args.output_dir, "strip.png"),
        size=(256, 256),
    )
    save_frames(results["images"], os.path.join(args.output_dir, "frames"))

    # Save reconstructions for sanity check
    results["reconA"].save(os.path.join(args.output_dir, "recon_A.png"))
    results["reconB"].save(os.path.join(args.output_dir, "recon_B.png"))

    print(f"\nClasses: {args.classA} -> {args.classB}")
    print(f"VAE reconstructions saved to {args.output_dir}/recon_*.png")
    if results["losses"]:
        print(f"Final GFM loss: {results['losses'][-1]:.6f}")
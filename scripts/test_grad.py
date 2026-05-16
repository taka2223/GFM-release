"""
NFSD Score Direction Sanity Check

Tests whether the NFSD score field provides a meaningful gradient direction
by starting from random points on the hypersphere and following the score.

Two tests:
  Test A: Random → Realistic
    Start from random noise on the sphere, follow score gradient.
    If NFSD is correct, decoded images should become progressively more realistic.

  Test B: Real Image Perturbation → Recovery  
    Start from a real image's inverted latent, perturb it off-manifold,
    then follow the score. Should recover toward realistic images.

  Test C: Score Alignment Check
    At a real image's latent, check if the score direction is consistent
    across multiple evaluations (not random noise).

Usage:
    python scripts/test_nfsd_sanity.py \
        --prompt "a girl with a flower in her hair" \
        --noise_level 0.1 \
        --device cuda

    python scripts/test_nfsd_sanity.py \
        --img real_photo.png \
        --prompt "a white cat" \
        --noise_level 0.1 \
        --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gfm.path.sd_wrapper import SDPipeline

OUTPUT_DIR = os.path.join(ROOT_DIR, "test_outputs", "nfsd_sanity")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================================== #
# Helpers
# =========================================================================== #

def save_strip(images, labels, out_path, size=(256, 256)):
    """Save horizontal strip with labels."""
    W, H = size
    pad = 25
    strip = Image.new("RGB", (W * len(images), H + pad), (240, 240, 240))
    draw = ImageDraw.Draw(strip)
    for i, (img, label) in enumerate(zip(images, labels)):
        strip.paste(img.resize(size), (i * W, pad))
        draw.text((i * W + 3, 3), label, fill=(0, 0, 0))
    strip.save(out_path)
    print(f"  → saved: {out_path}")


def compute_tangent_force(z, force, sphere_radius):
    """Project force to tangent space of sphere, return projected force and norms."""
    z_flat = z.view(1, -1)
    f_flat = force.view(1, -1)

    z_hat = z_flat / (z_flat.norm(dim=-1, keepdim=True) + 1e-8)
    radial = (f_flat * z_hat).sum(dim=-1, keepdim=True) * z_hat
    tangent = f_flat - radial

    return (
        tangent.view_as(z),
        tangent.norm().item(),
        radial.norm().item(),
        f_flat.norm().item(),
    )


def decode_latent(sd_pipe, z, embed, noise_level, cfg_scale=0.5):
    """Denoise + decode a noisy latent to PIL image."""
    with torch.no_grad():
        lat_clean = sd_pipe.ddim_backward(z, embed, noise_level, cfg_scale)
        img = sd_pipe.autoencoder.decode(lat_clean)
    return img


# =========================================================================== #
# Test A: Random → Realistic (follow score from noise)
# =========================================================================== #

def test_A_random_to_realistic(
    sd_pipe, prompt, noise_level, sphere_radius,
    n_steps=20, step_size=0.5, cfg_scale=0.5
):
    """
    Start from random noise on the sphere.
    Follow NFSD gradient for n_steps.
    Decode at each step — images should become more realistic.
    """
    print("\n=== Test A: Random → Realistic ===")
    device = sd_pipe.device

    timestep = sd_pipe.get_timestep(noise_level)
    sigma = sd_pipe.get_sigma_from_timestep(timestep)
    embed = sd_pipe.encode_prompt(prompt)
    sd_pipe.model.set_conditioning(embed, timestep)

    # Start from random noise on sphere
    z = torch.randn(1, 4, 64, 64, device=device)
    z = SDPipeline.project_to_sphere(z, sphere_radius)

    images = []
    labels = []
    force_norms = []

    sample_steps = [0, 1, 2, 5, 10, 15, 19]  # which steps to decode (expensive)
    sample_steps = [s for s in sample_steps if s < n_steps]

    for step in range(n_steps):
        # Decode at selected steps
        if step in sample_steps:
            img = decode_latent(sd_pipe, z, embed, noise_level, cfg_scale)
            images.append(img)
            labels.append(f"step {step}")

        # Compute NFSD force
        with torch.no_grad():
            sigma_tensor = torch.full((1,), sigma, device=device)
            x_pred = sd_pipe.model(z, sigma_tensor)
            force = -(z - x_pred) / (sigma ** 2)

            f_tangent, t_norm, r_norm, total_norm = compute_tangent_force(
                z, force, sphere_radius
            )
            force_norms.append(t_norm)

        # Step along tangent direction
        z = z + step_size * f_tangent
        z = SDPipeline.project_to_sphere(z, sphere_radius)

        if step % 5 == 0:
            print(f"  step {step:3d}: tangent_norm={t_norm:.4f}, "
                  f"radial_norm={r_norm:.4f}, total={total_norm:.4f}")

    # Decode final
    img = decode_latent(sd_pipe, z, embed, noise_level, cfg_scale)
    images.append(img)
    labels.append(f"step {n_steps}")

    save_strip(images, labels,
               os.path.join(OUTPUT_DIR, "A_random_to_realistic.png"))

    # Report: force norm trend
    early = np.mean(force_norms[:5])
    late = np.mean(force_norms[-5:])
    print(f"  Force norm: early={early:.4f}, late={late:.4f}")
    if late < early:
        print(f"  ✓ Force decreasing → converging toward high-density region")
    else:
        print(f"  ⚠ Force not decreasing — score may not be meaningful at this noise level")


# =========================================================================== #
# Test B: Perturb Real Image → Recovery
# =========================================================================== #

def test_B_perturb_and_recover(
    sd_pipe, input_image, prompt, noise_level,
    perturb_scale=0.3, n_steps=20, step_size=0.5, cfg_scale=0.5
):
    """
    Start from a real image's inverted latent.
    Perturb it in a random tangent direction.
    Follow NFSD gradient — should recover toward realistic images.
    """
    print("\n=== Test B: Perturb Real Image → Recovery ===")
    device = sd_pipe.device

    timestep = sd_pipe.get_timestep(noise_level)
    sigma = sd_pipe.get_sigma_from_timestep(timestep)
    embed = sd_pipe.encode_prompt(prompt)
    sd_pipe.model.set_conditioning(embed, timestep)

    # Encode and invert
    with torch.no_grad():
        lat_clean = sd_pipe.autoencoder.encode(input_image)
        z_real = sd_pipe.ddim_forward(lat_clean, embed, noise_level, cfg_scale)
        sphere_radius = z_real.view(-1).norm().item()
        z_real = SDPipeline.project_to_sphere(z_real, sphere_radius)

    # Decode original for reference
    img_ref = decode_latent(sd_pipe, z_real, embed, noise_level, cfg_scale)

    # Perturb in random tangent direction
    noise = torch.randn_like(z_real)
    z_flat = z_real.view(1, -1)
    n_flat = noise.view(1, -1)
    z_hat = z_flat / z_flat.norm(dim=-1, keepdim=True)
    n_tangent = n_flat - (n_flat * z_hat).sum(dim=-1, keepdim=True) * z_hat
    n_tangent = n_tangent / (n_tangent.norm(dim=-1, keepdim=True) + 1e-8)

    z_perturbed = z_real + perturb_scale * sphere_radius * n_tangent.view_as(z_real)
    z_perturbed = SDPipeline.project_to_sphere(z_perturbed, sphere_radius)

    # Decode perturbed
    img_perturbed = decode_latent(sd_pipe, z_perturbed, embed, noise_level, cfg_scale)

    # Follow score from perturbed point
    z = z_perturbed.clone()
    images = [input_image, img_ref, img_perturbed]
    labels = ["original", "inverted", f"perturbed ({perturb_scale})"]

    sample_steps = [0, 2, 5, 10, 15, 19]
    sample_steps = [s for s in sample_steps if s < n_steps]

    for step in range(n_steps):
        if step in sample_steps:
            img = decode_latent(sd_pipe, z, embed, noise_level, cfg_scale)
            images.append(img)
            labels.append(f"recover {step}")

        with torch.no_grad():
            sigma_tensor = torch.full((1,), sigma, device=device)
            x_pred = sd_pipe.model(z, sigma_tensor)
            force = -(z - x_pred) / (sigma ** 2)
            f_tangent, t_norm, _, _ = compute_tangent_force(z, force, sphere_radius)

        z = z + step_size * f_tangent
        z = SDPipeline.project_to_sphere(z, sphere_radius)

        if step % 5 == 0:
            dist = (z.view(-1) - z_real.view(-1)).norm().item()
            print(f"  step {step:3d}: tangent_force={t_norm:.4f}, "
                  f"dist_to_real={dist:.2f}")

    # Final
    img_final = decode_latent(sd_pipe, z, embed, noise_level, cfg_scale)
    images.append(img_final)
    labels.append(f"recover {n_steps}")

    save_strip(images, labels,
               os.path.join(OUTPUT_DIR, "B_perturb_recover.png"))


# =========================================================================== #
# Test C: Score Consistency Check
# =========================================================================== #

def test_C_score_consistency(
    sd_pipe, input_image, prompt, noise_level,
    n_evals=5, cfg_scale=0.5
):
    """
    Evaluate NFSD score at the same point multiple times.
    Since model is deterministic (no dropout at eval), results should be identical.
    If they differ, something is wrong with the wrapper.
    Also compare score direction at nearby points — should be similar (smooth field).
    """
    print("\n=== Test C: Score Consistency ===")
    device = sd_pipe.device

    timestep = sd_pipe.get_timestep(noise_level)
    sigma = sd_pipe.get_sigma_from_timestep(timestep)
    embed = sd_pipe.encode_prompt(prompt)
    sd_pipe.model.set_conditioning(embed, timestep)

    with torch.no_grad():
        lat_clean = sd_pipe.autoencoder.encode(input_image)
        z = sd_pipe.ddim_forward(lat_clean, embed, noise_level, cfg_scale)
        sphere_radius = z.view(-1).norm().item()

    # Test 1: Same-point consistency
    forces = []
    with torch.no_grad():
        for i in range(n_evals):
            sigma_tensor = torch.full((1,), sigma, device=device)
            x_pred = sd_pipe.model(z, sigma_tensor)
            force = -(z - x_pred) / (sigma ** 2)
            forces.append(force.clone())

    # All should be identical (deterministic model)
    max_diff = max(
        (forces[i] - forces[0]).abs().max().item()
        for i in range(1, n_evals)
    )
    print(f"  Same-point max diff over {n_evals} evals: {max_diff:.2e}")
    if max_diff > 1e-4:
        print(f"  ⚠ Non-deterministic! Check model.eval() or autocast.")
    else:
        print(f"  ✓ Deterministic (diff={max_diff:.2e})")

    # Test 2: Nearby-point smoothness
    with torch.no_grad():
        perturbations = [0.001, 0.01, 0.1]
        f0 = forces[0].view(-1)
        f0_hat = f0 / (f0.norm() + 1e-8)

        for eps in perturbations:
            noise = torch.randn_like(z)
            z_near = z + eps * noise
            z_near = SDPipeline.project_to_sphere(z_near, sphere_radius)

            x_pred = sd_pipe.model(z_near, sigma_tensor)
            f_near = -(z_near - x_pred) / (sigma ** 2)
            f_near_flat = f_near.view(-1)
            f_near_hat = f_near_flat / (f_near_flat.norm() + 1e-8)

            cosine = (f0_hat * f_near_hat).sum().item()
            norm_ratio = f_near_flat.norm().item() / (f0.norm().item() + 1e-8)
            print(f"  ε={eps}: cosine={cosine:.4f}, norm_ratio={norm_ratio:.4f}")

        print(f"  (cosine→1.0 and norm_ratio→1.0 means smooth field)")


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="NFSD Score Sanity Check")
    parser.add_argument("--img", type=str, default=None,
                        help="Path to real image (for tests B and C)")
    parser.add_argument("--prompt", type=str, default="a realistic photograph",
                        help="Text prompt")
    parser.add_argument("--noise_level", type=float, default=0.1,
                        help="Diffusion noise level")
    parser.add_argument("--sphere_radius", type=float, default=98.0,
                        help="Sphere radius for Test A (auto-computed for B/C)")
    parser.add_argument("--n_steps", type=int, default=20,
                        help="Number of score-following steps")
    parser.add_argument("--step_size", type=float, default=0.5,
                        help="Step size for gradient following")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("  NFSD Score Direction Sanity Check")
    print(f"  noise_level={args.noise_level}, prompt='{args.prompt}'")
    print(f"  output: {OUTPUT_DIR}")
    print("=" * 60)

    sd_pipe = SDPipeline.load(device=args.device)

    # Test A: always run (no image needed)
    test_A_random_to_realistic(
        sd_pipe, args.prompt, args.noise_level, args.sphere_radius,
        n_steps=args.n_steps, step_size=args.step_size,
    )

    # Tests B and C: need a real image
    if args.img and os.path.exists(args.img):
        input_image = Image.open(args.img).convert("RGB")

        test_B_perturb_and_recover(
            sd_pipe, input_image, args.prompt, args.noise_level,
            n_steps=args.n_steps, step_size=args.step_size,
        )

        test_C_score_consistency(
            sd_pipe, input_image, args.prompt, args.noise_level,
        )
    else:
        print("\n  (Skipping Tests B/C — no --img provided)")

    print("\n" + "=" * 60)
    print("  Done. Check visual outputs in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
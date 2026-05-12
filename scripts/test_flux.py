"""
Correctness tests for flux_wrapper.py

Tests (in order of increasing complexity):
  T1  preprocess / postprocess are exact inverses
  T2  VAE encode → decode round-trip  (save visual)
  T3  project_to_sphere  norm invariant
  T4  get_sigma_from_noise_level monotone ordering
  T5  FluxModel.forward ('denoise' mode)  x_pred numerically sane
  T6  FluxModel.forward ('nfsd' mode)     gradient not NaN/Inf
  T7  flow_backward ONLY (manual noising → denoise, no inversion)
  T8  flow_forward + flow_backward roundtrip (unconditional, cfg=0)
  T9  flow_forward + flow_backward roundtrip (with text inversion)
  T10 compute_force interface compatibility with GFM

Run:
    python test_flux_correctness.py [--img path/to/image.jpg] [--device cuda]

Key math differences from SD test script:
  - Noising: z_t = (1-t)·x₀ + t·ε       (rectified flow, NOT VP schedule)
  - No alphas_cumprod — the flow time t IS the noise level
  - x₀ = z_t - t·v                        (v-prediction denoising)
  - Force = -(z_t - x_pred) / t²          (t replaces σ²)
  - No VP↔EDM conversions anywhere

Interpretation:
  - T1-T6: Pure logic tests. Must all pass.
  - T7: Tests flow_backward in isolation. Uses exact linear noising.
         If this fails, flow_backward has a bug.
  - T8: Tests flow_forward+backward with cfg=0 (no CFG mismatch).
         Unconditional inversion should be near-exact up to discretization.
         If noise=0.3 fails, flow_forward has a bug.
  - T9: Full pipeline with text inversion. Only with --test_text_inv (slow).
  - T10: Tests that FluxModel output works correctly with GFM's compute_force.
"""

import argparse
import math
import os
import sys
import traceback

import numpy as np
import torch
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gfm.path.flux_wrapper import FluxAutoencoder, FluxModel, FluxGFMPipeline

OUTPUT_DIR = os.path.join(ROOT_DIR, "test_outputs", "flux_correctness")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================================== #
# Helpers                                                                      #
# =========================================================================== #

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""

    def ok(self, msg=""):
        self.passed = True
        self.message = msg
        status = "\033[32mPASS\033[0m"
        print(f"  [{status}] {self.name}" + (f": {msg}" if msg else ""))
        return self

    def fail(self, msg=""):
        self.passed = False
        self.message = msg
        status = "\033[31mFAIL\033[0m"
        print(f"  [{status}] {self.name}: {msg}")
        return self


def psnr(a: Image.Image, b: Image.Image) -> float:
    w = min(a.width, b.width)
    h = min(a.height, b.height)
    a = a.convert("RGB").resize((w, h), Image.LANCZOS)
    b = b.convert("RGB").resize((w, h), Image.LANCZOS)
    arr_a = np.array(a).astype(np.float32)
    arr_b = np.array(b).astype(np.float32)
    mse = np.mean((arr_a - arr_b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def make_comparison_strip(images: list, labels: list, out_path: str):
    assert len(images) == len(labels)
    W, H = 512, 512
    pad = 30
    strip_w = W * len(images)
    strip_h = H + pad
    strip = Image.new("RGB", (strip_w, strip_h), (240, 240, 240))
    draw = ImageDraw.Draw(strip)
    for i, (img, label) in enumerate(zip(images, labels)):
        img_rs = img.convert("RGB").resize((W, H))
        strip.paste(img_rs, (i * W, pad))
        draw.text((i * W + 5, 5), label, fill=(0, 0, 0))
    strip.save(out_path)
    print(f"    → saved: {out_path}")


def make_test_image(size=(512, 512)) -> Image.Image:
    """Synthetic gradient image with a white circle — no external dependencies."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            arr[y, x, 0] = int(255 * x / size[0])
            arr[y, x, 1] = int(255 * y / size[1])
            arr[y, x, 2] = int(255 * (x + y) / (size[0] + size[1]))
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    cx, cy, r = size[0] // 2, size[1] // 2, size[0] // 6
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
    return img


def flow_add_noise(x0: torch.Tensor, t: float) -> tuple:
    """
    Rectified flow noising: z_t = (1-t)·x₀ + t·ε

    This is the EXACT analogue of scheduler.add_noise in VP diffusion,
    but for flow matching there is no scheduler — it's just linear interp.

    Returns:
        z_t:   noisy latent
        noise: the ε that was used (for reference)
    """
    noise = torch.randn_like(x0)
    z_t = (1.0 - t) * x0 + t * noise
    return z_t, noise


# =========================================================================== #
# T1-T6: Pure logic tests
# =========================================================================== #

def test_T1_preprocess_postprocess(autoenc: FluxAutoencoder) -> TestResult:
    r = TestResult("T1  preprocess ↔ postprocess roundtrip")
    img = make_test_image()
    tensor = autoenc._preprocess(img)
    if tensor.min().item() < -1.05 or tensor.max().item() > 1.05:
        return r.fail(f"tensor out of [-1,1]: [{tensor.min():.3f}, {tensor.max():.3f}]")
    recovered = autoenc._postprocess(tensor)
    # Resize original to match autoencoder resolution for fair comparison
    img_resized = img.resize((autoenc.resolution, autoenc.resolution), Image.LANCZOS)
    err = np.abs(np.array(img_resized).astype(int) - np.array(recovered).astype(int))
    if err.max() > 2:
        return r.fail(f"max pixel error={err.max()}, mean={err.mean():.4f}")
    return r.ok(f"max_err={err.max()}, mean_err={err.mean():.4f}")


def test_T2_vae_roundtrip(autoenc: FluxAutoencoder, input_image: Image.Image) -> TestResult:
    r = TestResult("T2  VAE encode → decode roundtrip")
    with torch.no_grad():
        latent = autoenc.encode(input_image)
        reconstructed = autoenc.decode(latent)
    if not torch.isfinite(latent).all():
        return r.fail("latent contains NaN or Inf")
    latent_shape = latent.shape
    info_shape = f"latent_shape={list(latent_shape)}"
    # Flux VAE should produce 16-channel latents
    if latent_shape[1] != 16:
        return r.fail(f"expected 16 latent channels, got {latent_shape[1]}  ({info_shape})")
    score = psnr(input_image, reconstructed)
    make_comparison_strip(
        [input_image, reconstructed],
        ["original", f"VAE decoded  PSNR={score:.1f}dB"],
        os.path.join(OUTPUT_DIR, "T2_vae_roundtrip.png"),
    )
    if score < 20.0:
        return r.fail(f"PSNR={score:.1f}dB below 20dB  ({info_shape})")
    return r.ok(f"PSNR={score:.1f}dB  {info_shape}")


def test_T3_sphere_projection() -> TestResult:
    r = TestResult("T3  project_to_sphere norm invariant")
    # Flux latent: [B, 16, H/8, W/8]; for 1024×1024 → [B, 16, 128, 128]
    for shape, radius in [((1, 16, 128, 128), 15.0), ((3, 16, 128, 128), 100.0)]:
        x = torch.randn(*shape)
        projected = FluxGFMPipeline.project_to_sphere(x, radius)
        norms = projected.view(shape[0], -1).norm(dim=-1)
        err = (norms - radius).abs().max().item()
        if err > 1e-4:
            return r.fail(f"shape={shape} radius={radius}: err={err:.6f}")
    return r.ok("all within 1e-4")


def test_T4_sigma_ordering(flux_pipe: FluxGFMPipeline) -> TestResult:
    r = TestResult("T4  get_sigma_from_noise_level monotone ordering")
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    sigmas = [flux_pipe.get_sigma_from_noise_level(l) for l in levels]

    # In flow matching: higher noise_level → larger flow time t → more noise
    sg_ok = all(sigmas[i] <= sigmas[i + 1] for i in range(len(sigmas) - 1))
    info = f"σ(=t)={[f'{s:.4f}' for s in sigmas]}"

    # Sanity: all sigmas should be in (0, 1] for rectified flow
    range_ok = all(0 < s <= 1.0 for s in sigmas)

    if not sg_ok:
        return r.fail(f"sigmas not monotone  {info}")
    if not range_ok:
        return r.fail(f"sigmas outside (0,1]  {info}")
    return r.ok(info)


def test_T5_flux_model_denoise_mode(flux_pipe: FluxGFMPipeline,
                                     input_image: Image.Image) -> TestResult:
    r = TestResult("T5  FluxModel.forward (denoise mode)")
    original_mode = flux_pipe.model.mode
    flux_pipe.model.mode = "denoise"

    t_val = 0.4  # flow time
    embeds, pooled, tids = flux_pipe.encode_prompt("a photo")
    flux_pipe.model.set_conditioning(embeds, pooled, tids, t_val)

    with torch.no_grad():
        clean = flux_pipe.autoencoder.encode(input_image)
        # Rectified flow noising: z_t = (1-t)·x₀ + t·ε
        noisy, _ = flow_add_noise(clean, t_val)

        sigma_tensor = torch.tensor([t_val], device=flux_pipe.device)
        x_pred = flux_pipe.model(noisy, sigma_tensor)

    flux_pipe.model.mode = original_mode

    if not torch.isfinite(x_pred).all():
        return r.fail("x_pred NaN/Inf")
    ratio = x_pred.norm().item() / (clean.norm().item() + 1e-8)
    info = f"x_pred_norm={x_pred.norm():.2f}, clean_norm={clean.norm():.2f}, ratio={ratio:.2f}"
    if ratio > 20 or ratio < 0.05:
        return r.fail(f"scale suspicious  {info}")
    return r.ok(info)


def test_T6_flux_model_nfsd_mode(flux_pipe: FluxGFMPipeline,
                                  input_image: Image.Image) -> TestResult:
    r = TestResult("T6  FluxModel.forward (nfsd mode)")
    assert flux_pipe.model.mode == "nfsd"

    t_val = 0.6
    embeds, pooled, tids = flux_pipe.encode_prompt("a realistic photo")
    flux_pipe.model.set_conditioning(embeds, pooled, tids, t_val)

    with torch.no_grad():
        clean = flux_pipe.autoencoder.encode(input_image)
        noisy, _ = flow_add_noise(clean, t_val)

        sigma_tensor = torch.tensor([t_val], device=flux_pipe.device)
        x_pred = flux_pipe.model(noisy, sigma_tensor)

    if not torch.isfinite(x_pred).all():
        return r.fail("x_pred NaN/Inf")

    # Force in flow matching: -(z_t - x_pred) / t²
    force = -(noisy - x_pred) / (t_val ** 2 + 1e-8)
    force_norm = force.norm().item()
    if force_norm == 0:
        return r.fail("force is exactly zero")
    return r.ok(f"force_norm={force_norm:.4f}")


# =========================================================================== #
# T7: Flow Backward ONLY (no inversion — isolates denoise correctness)
# =========================================================================== #

def test_T7_flow_denoise_only(flux_pipe: FluxGFMPipeline,
                               input_image: Image.Image,
                               prompt: str) -> TestResult:
    """
    Add noise via exact flow noising z_t = (1-t)·x₀ + t·ε,
    then denoise via flow_backward.

    This isolates flow_backward from any inversion error.
    If this fails, flow_backward has a bug.

    NOTE: thresholds are more lenient than SD because:
      1. Flux's higher-dim latent (16ch) accumulates more discretization error
      2. Euler ODE integration is first-order
      3. The internal guidance embedding adds a bias vs pure unconditional
    """
    r = TestResult("T7  flow_backward only (exact noising → denoise)")

    if not prompt:
        prompt = "a photo"

    failures = []
    strip_images = [input_image]
    strip_labels = ["original"]

    configs = [
        # (noise_level, cfg_scale, prompt_str,  label,          threshold)
        (0.3, 0.0,  "",     "uncond t=0.3",  18.0),
        (0.6, 0.0,  "",     "uncond t=0.6",  12.0),
        (0.3, 0.0,  prompt, "cond t=0.3",    20.0),
        (0.6, 0.0,  prompt, "cond t=0.6",    16.0),
    ]

    with torch.no_grad():
        latent_clean = flux_pipe.autoencoder.encode(input_image)

        for noise_level, cfg, p, label, threshold in configs:
            embeds, pooled, tids = flux_pipe.encode_prompt(p if p else "")

            # Exact flow noising (no approximation)
            noisy, _ = flow_add_noise(latent_clean, noise_level)

            # Denoise via backward ODE
            latent_recon = flux_pipe.flow_backward(
                noisy, embeds, pooled, tids,
                noise_level, cfg_scale=cfg
            )
            recon_img = flux_pipe.autoencoder.decode(latent_recon)

            score = psnr(input_image, recon_img)
            strip_images.append(recon_img)
            strip_labels.append(f"{label}\nPSNR={score:.1f}dB")

            if score < threshold:
                failures.append(f"{label}: {score:.1f}dB < {threshold}dB")

    make_comparison_strip(
        strip_images, strip_labels,
        os.path.join(OUTPUT_DIR, "T7_flow_denoise_only.png"),
    )

    if failures:
        return r.fail(" | ".join(failures))
    return r.ok("all denoise-only tests passed")


# =========================================================================== #
# T8: Flow Forward + Backward Roundtrip (unconditional, cfg=0)
# =========================================================================== #

def test_T8_flow_inversion_uncond(flux_pipe: FluxGFMPipeline,
                                   input_image: Image.Image) -> TestResult:
    """
    Flow forward inversion then backward, both with cfg=0.

    With cfg=0 and Flux's internal guidance, forward/backward should be
    near-inverses since they integrate the SAME velocity field in
    opposite directions. Error comes from:
      1. Euler discretization (first-order)
      2. Karras schedule spacing
      3. Final projection x₀ = z_t - t·v at t_min > 0

    Expected (lenient — Euler + 16ch latent):
      noise=0.2: >25dB
      noise=0.4: >20dB
      noise=0.6: >14dB
    """
    r = TestResult("T8  flow inversion roundtrip (uncond, cfg=0)")

    embeds_uncond, pooled_uncond, tids_uncond = flux_pipe.encode_prompt("")
    failures = []
    strip_images = [input_image]
    strip_labels = ["original"]

    configs = [
        (0.2, 25.0),
        (0.4, 20.0),
        (0.6, 14.0),
    ]

    with torch.no_grad():
        latent_clean = flux_pipe.autoencoder.encode(input_image)

        for noise_level, threshold in configs:
            latent_noisy = flux_pipe.flow_forward(
                latent_clean, embeds_uncond, pooled_uncond, tids_uncond,
                noise_level, cfg_scale=0.0
            )
            latent_recon = flux_pipe.flow_backward(
                latent_noisy, embeds_uncond, pooled_uncond, tids_uncond,
                noise_level, cfg_scale=0.0
            )
            recon_img = flux_pipe.autoencoder.decode(latent_recon)

            score = psnr(input_image, recon_img)
            lat_err = (latent_recon - latent_clean).norm() / (latent_clean.norm() + 1e-8)

            strip_images.append(recon_img)
            strip_labels.append(
                f"uncond t={noise_level}\n"
                f"PSNR={score:.1f}dB\n"
                f"lat_err={lat_err:.3f}"
            )

            if score < threshold:
                failures.append(
                    f"t={noise_level}: {score:.1f}dB < {threshold}dB "
                    f"(lat_err={lat_err:.3f})"
                )

    make_comparison_strip(
        strip_images, strip_labels,
        os.path.join(OUTPUT_DIR, "T8_flow_inversion_uncond.png"),
    )

    if failures:
        return r.fail(" | ".join(failures))
    return r.ok("all unconditional roundtrips passed")


# =========================================================================== #
# T9: Flow Roundtrip with Text Inversion
# =========================================================================== #

def test_T9_flow_with_text_inversion(flux_pipe: FluxGFMPipeline,
                                      input_image: Image.Image,
                                      prompt: str) -> TestResult:
    """
    Full pipeline: text inversion → flow forward → flow backward.

    Text inversion optimizes T5 embeddings to faithfully reconstruct
    this specific image under the flow matching loss:
        L = ||v_θ(z_t, t; e_opt) - (ε - x₀)||²

    SLOW: ~120s for 500 steps (Flux transformer is larger than SD UNet).
    """
    r = TestResult("T9  flow roundtrip with text inversion")

    if not prompt:
        return r.fail("need --prompt for T9")

    failures = []
    strip_images = [input_image]
    strip_labels = ["original"]

    with torch.no_grad():
        latent_clean = flux_pipe.autoencoder.encode(input_image)

    # Text inversion (requires grad, returns tuple)
    print("    Running text inversion (500 steps, Flux — this is slow)...")
    embed_inv, pooled_inv, tid_inv = flux_pipe.text_inversion(
        prompt, latent_clean, steps=500, lr=0.005
    )

    configs = [
        # (noise_level, cfg_scale, threshold)
        (0.3, 0.0, 22.0),
        (0.6, 0.0, 18.0),
    ]

    with torch.no_grad():
        for noise_level, cfg, threshold in configs:
            latent_noisy = flux_pipe.flow_forward(
                latent_clean, embed_inv, pooled_inv, tid_inv,
                noise_level, cfg_scale=cfg
            )
            latent_recon = flux_pipe.flow_backward(
                latent_noisy, embed_inv, pooled_inv, tid_inv,
                noise_level, cfg_scale=cfg
            )
            recon_img = flux_pipe.autoencoder.decode(latent_recon)

            score = psnr(input_image, recon_img)
            strip_images.append(recon_img)
            strip_labels.append(f"text_inv t={noise_level}\nPSNR={score:.1f}dB")

            if score < threshold:
                failures.append(f"t={noise_level}: {score:.1f}dB < {threshold}dB")

    make_comparison_strip(
        strip_images, strip_labels,
        os.path.join(OUTPUT_DIR, "T9_flow_text_inversion.png"),
    )

    if failures:
        return r.fail(" | ".join(failures))
    return r.ok("text inversion roundtrips passed")


# =========================================================================== #
# T10: compute_force interface test (GFM compatibility)
# =========================================================================== #

def test_T10_compute_force_interface(flux_pipe: FluxGFMPipeline,
                                      input_image: Image.Image) -> TestResult:
    """
    Verify that FluxModel works correctly with GFM's compute_force pattern:
        x_pred = model(latent, sigma_tensor, class_labels)
        force  = -(latent - x_pred) / sigma²

    In flow matching, sigma ≡ t, so:
        force = -(z_t - x_pred) / t²
              = v_θ / t                    (when mode='denoise')
              = NFSD_grad / t              (when mode='nfsd')

    Checks:
      1. force is finite and non-zero
      2. force has the same shape as input
      3. force projected to tangent space of sphere is non-zero
    """
    r = TestResult("T10 compute_force GFM interface")

    t_val = 0.6
    embeds, pooled, tids = flux_pipe.encode_prompt("a photo")
    flux_pipe.model.set_conditioning(embeds, pooled, tids, t_val)

    with torch.no_grad():
        clean = flux_pipe.autoencoder.encode(input_image)
        noisy, _ = flow_add_noise(clean, t_val)

        # Simulate what BaseInterpolator.compute_force does
        sigma_tensor = torch.full((noisy.shape[0],), t_val, device=flux_pipe.device)
        x_pred = flux_pipe.model(noisy, sigma_tensor, None)
        force = -(noisy - x_pred) / (t_val ** 2)

    # Check 1: finite and non-zero
    if not torch.isfinite(force).all():
        return r.fail("force contains NaN/Inf")
    if force.norm().item() == 0:
        return r.fail("force is exactly zero")

    # Check 2: shape matches
    if force.shape != noisy.shape:
        return r.fail(f"shape mismatch: force={force.shape} vs input={noisy.shape}")

    # Check 3: tangent component exists (for sphere constraint in GFM)
    x_flat = noisy.view(1, -1)
    f_flat = force.view(1, -1)
    x_hat = x_flat / (x_flat.norm(dim=-1, keepdim=True) + 1e-8)
    radial = (f_flat * x_hat).sum(dim=-1, keepdim=True) * x_hat
    tangent = f_flat - radial
    tangent_norm = tangent.norm().item()
    radial_norm = radial.norm().item()
    total_norm = f_flat.norm().item()

    if tangent_norm < 1e-10:
        return r.fail("force is purely radial — no tangent component for GFM")

    info = (f"force_norm={total_norm:.4f}, "
            f"tangent={tangent_norm:.4f}, radial={radial_norm:.4f}, "
            f"tangent_frac={tangent_norm / total_norm:.3f}")
    return r.ok(info)


# =========================================================================== #
# T11: Latent packing / unpacking roundtrip (Flux-specific)
# =========================================================================== #

def test_T11_pack_unpack_roundtrip() -> TestResult:
    """
    Flux packs [B,C,H,W] → [B, (H/2)(W/2), C*4] for the transformer.
    Verify this is a perfect bijection.
    """
    r = TestResult("T11 pack ↔ unpack roundtrip (Flux-specific)")

    for B, C, H, W in [(1, 16, 128, 128), (2, 16, 64, 64), (1, 16, 96, 96)]:
        x = torch.randn(B, C, H, W)
        packed = FluxModel._pack_latents(x)
        unpacked = FluxModel._unpack_latents(packed, H, W)

        if packed.shape != (B, (H // 2) * (W // 2), C * 4):
            return r.fail(
                f"packed shape wrong: expected "
                f"{(B, (H // 2) * (W // 2), C * 4)}, got {packed.shape}"
            )
        err = (unpacked - x).abs().max().item()
        if err > 1e-6:
            return r.fail(f"shape ({B},{C},{H},{W}): roundtrip err={err:.8f}")

    return r.ok("all shapes roundtrip exactly")


# =========================================================================== #
# T12: VAE normalization consistency (shift_factor + scaling_factor)
# =========================================================================== #

def test_T12_vae_normalization(autoenc: FluxAutoencoder,
                                input_image: Image.Image) -> TestResult:
    """
    Verify encode/decode normalization is self-consistent:
        encode: lat = (raw - shift) * scale
        decode: raw = lat / scale + shift
    If these are mismatched, VAE roundtrip degrades silently.
    """
    r = TestResult("T12 VAE normalization (shift + scale consistency)")

    with torch.no_grad():
        img_tensor = autoenc._preprocess(input_image)
        raw = autoenc.vae.encode(img_tensor)['latent_dist'].mean

        # Manual encode path
        lat_manual = (raw - autoenc.shift_factor) * autoenc.scaling_factor

        # Via autoencoder.encode
        lat_api = autoenc.encode(input_image)

        encode_err = (lat_manual - lat_api).abs().max().item()
        if encode_err > 1e-4:
            return r.fail(f"encode mismatch: max_err={encode_err:.6f}")

        # Manual decode path
        raw_back = lat_api / autoenc.scaling_factor + autoenc.shift_factor
        img_manual = autoenc.vae.decode(raw_back)['sample']

        # Via autoencoder.decode
        img_api = autoenc.decode(lat_api)
        img_api_tensor = autoenc._preprocess(img_api)  # re-preprocess for comparison

        # Compare at tensor level (before uint8 quantization)
        decode_err = (img_manual - img_api_tensor).abs().mean().item()

    info = (f"encode_err={encode_err:.6f}, decode_err={decode_err:.6f}, "
            f"shift={autoenc.shift_factor}, scale={autoenc.scaling_factor}")

    # decode_err will be nonzero due to uint8 quantization in _postprocess→_preprocess
    # but should be small
    if decode_err > 0.02:
        return r.fail(info)
    return r.ok(info)


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="Flux wrapper correctness tests")
    parser.add_argument("--img", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for the image (used in T7 cond, T9)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_id", type=str,
                        default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution (Flux native: 1024)")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--test_text_inv", action="store_true",
                        help="Include T9 (text inversion, slow ~120s)")
    args = parser.parse_args()

    print("=" * 65)
    print("  Flux Wrapper Correctness Tests")
    print(f"  output dir: {OUTPUT_DIR}")
    print("=" * 65)

    if args.img and os.path.exists(args.img):
        input_image = Image.open(args.img).convert("RGB")
        print(f"\nUsing real image: {args.img}")
    else:
        input_image = make_test_image()
        print("\nNo real image — using synthetic test image.")
        input_image.save(os.path.join(OUTPUT_DIR, "synthetic_input.png"))

    # --- Run offline tests first (no model needed) ---
    print("\n--- Offline tests (no model) ---")
    offline_results = []

    print("\nRunning T3…")
    offline_results.append(test_T3_sphere_projection())

    print("\nRunning T11…")
    offline_results.append(test_T11_pack_unpack_roundtrip())

    # --- Load model ---
    print(f"\nLoading Flux pipeline ({args.model_id}, device={args.device})…")
    print("  (This may take a while — Flux is ~24GB)")
    try:
        flux_pipe = FluxGFMPipeline.load(
            model_id=args.model_id,
            device=args.device,
            cache_dir=args.cache_dir,
            load_blip=False,
            resolution=args.resolution,
        )
        print("Pipeline loaded.\n")
    except Exception:
        traceback.print_exc()
        print("\nFailed to load Flux. Offline test results:")
        for res in offline_results:
            status = "\033[32m✓\033[0m" if res.passed else "\033[31m✗\033[0m"
            print(f"  {status}  {res.name}: {res.message}")
        sys.exit(1)

    autoenc = flux_pipe.autoencoder

    # Build test list (model-dependent tests)
    tests = [
        *[("offline", r) for r in offline_results],  # include offline results
        ("T1",  lambda: test_T1_preprocess_postprocess(autoenc)),
        ("T2",  lambda: test_T2_vae_roundtrip(autoenc, input_image)),
        ("T4",  lambda: test_T4_sigma_ordering(flux_pipe)),
        ("T5",  lambda: test_T5_flux_model_denoise_mode(flux_pipe, input_image)),
        ("T6",  lambda: test_T6_flux_model_nfsd_mode(flux_pipe, input_image)),
        ("T7",  lambda: test_T7_flow_denoise_only(flux_pipe, input_image, args.prompt)),
        ("T8",  lambda: test_T8_flow_inversion_uncond(flux_pipe, input_image)),
        ("T10", lambda: test_T10_compute_force_interface(flux_pipe, input_image)),
        ("T12", lambda: test_T12_vae_normalization(autoenc, input_image)),
    ]

    if args.test_text_inv:
        tests.append(
            ("T9", lambda: test_T9_flow_with_text_inversion(
                flux_pipe, input_image, args.prompt))
        )
    else:
        print("  (Skipping T9 — pass --test_text_inv to include)")

    results = []
    for name, fn_or_result in tests:
        if name == "offline":
            # Already ran; just collect the result
            results.append(fn_or_result)
            continue
        print(f"\nRunning {name}…")
        try:
            res = fn_or_result()
        except Exception as e:
            res = TestResult(name)
            res.fail(f"EXCEPTION: {e}")
            traceback.print_exc()
        results.append(res)

    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    for res in results:
        status = "\033[32m✓\033[0m" if res.passed else "\033[31m✗\033[0m"
        print(f"  {status}  {res.name}: {res.message}")
    print(f"\n  {n_pass}/{len(results)} passed, {n_fail} failed")
    print(f"  Visuals: {OUTPUT_DIR}")

    if n_fail > 0:
        print("\n  Diagnostic guide:")
        print("    T1 fail  → preprocess/postprocess mismatch (resolution?)")
        print("    T2 fail  → VAE encode/decode broken or wrong channel count")
        print("    T5 fail  → v-prediction → x₀ formula wrong (x₀ = z_t - t·v)")
        print("    T6 fail  → NFSD gradient computation wrong (σ² → t mapping)")
        print("    T7 fail  → flow_backward ODE integration bug")
        print("    T8 fail at t=0.2/0.4 → flow_forward ODE integration bug")
        print("    T8 fail only at t=0.6 → expected Euler discretization error")
        print("    T9 fail  → text inversion loss or embedding interface wrong")
        print("    T10 fail → FluxModel interface incompatible with GFM")
        print("    T11 fail → latent packing/unpacking bijection broken")
        print("    T12 fail → VAE shift_factor/scaling_factor encode↔decode mismatch")

    print("=" * 65)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
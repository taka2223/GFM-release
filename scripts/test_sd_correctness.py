"""
Correctness tests for gfm/path/sd_wrapper.py

Tests (in order of increasing complexity):
  T1  preprocess / postprocess are exact inverses
  T2  VAE encode → decode round-trip  (save visual)
  T3  project_to_sphere  norm invariant
  T4  get_timestep + get_sigma  monotone ordering
  T5  SDModel.forward ('denoise' mode)  x_pred numerically sane
  T6  SDModel.forward ('nfsd' mode)     gradient not NaN/Inf
  T7  DDIM backward ONLY (add_noise → denoise, no inversion)
  T8  DDIM forward+backward roundtrip (unconditioned, cfg=0)
  T9  DDIM forward+backward roundtrip (with text inversion)
  T10 compute_force interface compatibility with GFM

Run:
    python scripts/test_sd_correctness.py [--img path/to/image.jpg] [--device cuda]

Interpretation:
  - T1-T6: Pure logic tests. Must all pass.
  - T7: Tests DDIM backward in isolation. Uses exact add_noise (no approx).
         If this fails, ddim_backward has a bug.
  - T8: Tests DDIM forward+backward with cfg=0 (no CFG mismatch).
         Unconditional inversion should be mathematically exact up to
         discretization. If noise=0.3 fails here, ddim_forward has a bug.
         noise=0.6 may degrade due to long inversion chain — threshold is lenient.
  - T9: Tests full pipeline with text inversion. This is the "realistic"
         test. Only run if --test_text_inv is passed (slow: ~60s per image).
  - T10: Tests that SDModel output works correctly with GFM's compute_force.
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

from gfm.path.sd_wrapper_gemini import SDAutoencoder, SDModel, SDPipeline

OUTPUT_DIR = os.path.join(ROOT_DIR, "test_outputs", "sd_correctness")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================================== #
# Helpers                                                                       #
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


# =========================================================================== #
# T1-T6: Pure logic tests (unchanged)
# =========================================================================== #

def test_T1_preprocess_postprocess(autoenc: SDAutoencoder) -> TestResult:
    r = TestResult("T1  preprocess ↔ postprocess roundtrip")
    img = make_test_image()
    tensor = autoenc._preprocess(img)
    if tensor.min().item() < -1.05 or tensor.max().item() > 1.05:
        return r.fail(f"tensor out of [-1,1]: [{tensor.min():.3f}, {tensor.max():.3f}]")
    recovered = autoenc._postprocess(tensor)
    err = np.abs(np.array(img).astype(int) - np.array(recovered).astype(int))
    if err.max() > 2:
        return r.fail(f"max pixel error={err.max()}, mean={err.mean():.4f}")
    return r.ok(f"max_err={err.max()}, mean_err={err.mean():.4f}")


def test_T2_vae_roundtrip(autoenc: SDAutoencoder, input_image: Image.Image) -> TestResult:
    r = TestResult("T2  VAE encode → decode roundtrip")
    with torch.no_grad():
        latent = autoenc.encode(input_image)
        reconstructed = autoenc.decode(latent)
    if not torch.isfinite(latent).all():
        return r.fail("latent contains NaN or Inf")
    score = psnr(input_image, reconstructed)
    make_comparison_strip(
        [input_image, reconstructed],
        ["original", f"VAE decoded  PSNR={score:.1f}dB"],
        os.path.join(OUTPUT_DIR, "T2_vae_roundtrip.png"),
    )
    if score < 20.0:
        return r.fail(f"PSNR={score:.1f}dB below 20dB")
    return r.ok(f"PSNR={score:.1f}dB")


def test_T3_sphere_projection() -> TestResult:
    r = TestResult("T3  project_to_sphere norm invariant")
    for shape, radius in [((1, 4, 64, 64), 15.0), ((3, 4, 64, 64), 100.0)]:
        x = torch.randn(*shape)
        projected = SDPipeline.project_to_sphere(x, radius)
        norms = projected.view(shape[0], -1).norm(dim=-1)
        err = (norms - radius).abs().max().item()
        if err > 1e-4:
            return r.fail(f"shape={shape} radius={radius}: err={err:.6f}")
    return r.ok("all within 1e-4")


def test_T4_timestep_sigma_ordering(sd_pipe: SDPipeline) -> TestResult:
    r = TestResult("T4  get_timestep + get_sigma monotone ordering")
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    timesteps = [sd_pipe.get_timestep(l) for l in levels]
    sigmas = [sd_pipe.get_sigma_from_timestep(t) for t in timesteps]
    ts_ok = all(timesteps[i] <= timesteps[i + 1] for i in range(len(timesteps) - 1))
    sg_ok = all(sigmas[i] <= sigmas[i + 1] for i in range(len(sigmas) - 1))
    info = f"t={timesteps}  σ={[f'{s:.3f}' for s in sigmas]}"
    if not ts_ok:
        return r.fail(f"timesteps not monotone  {info}")
    if not sg_ok:
        return r.fail(f"sigmas not monotone  {info}")
    return r.ok(info)


def test_T5_sdmodel_denoise_mode(sd_pipe: SDPipeline, input_image: Image.Image) -> TestResult:
    r = TestResult("T5  SDModel.forward (denoise mode)")
    original_mode = sd_pipe.model.mode
    sd_pipe.model.mode = "denoise"
    t = sd_pipe.get_timestep(0.4)
    sigma = sd_pipe.get_sigma_from_timestep(t)
    embed = sd_pipe.encode_prompt("a photo")
    sd_pipe.model.set_conditioning(embed, t)
    with torch.no_grad():
        clean = sd_pipe.autoencoder.encode(input_image)
        noise = torch.randn_like(clean)
        alpha_t = sd_pipe.pipe.scheduler.alphas_cumprod[t]
        noisy = alpha_t.sqrt() * clean + (1 - alpha_t).sqrt() * noise
        x_pred = sd_pipe.model(noisy, torch.tensor([sigma], device=sd_pipe.device))
    sd_pipe.model.mode = original_mode
    if not torch.isfinite(x_pred).all():
        return r.fail("x_pred NaN/Inf")
    ratio = x_pred.norm().item() / (clean.norm().item() + 1e-8)
    info = f"x_pred_norm={x_pred.norm():.2f}, clean_norm={clean.norm():.2f}, ratio={ratio:.2f}"
    if ratio > 20 or ratio < 0.05:
        return r.fail(f"scale suspicious  {info}")
    return r.ok(info)


def test_T6_sdmodel_nfsd_mode(sd_pipe: SDPipeline, input_image: Image.Image) -> TestResult:
    r = TestResult("T6  SDModel.forward (nfsd mode)")
    t = sd_pipe.get_timestep(0.6)
    sigma = sd_pipe.get_sigma_from_timestep(t)
    embed = sd_pipe.encode_prompt("a realistic photo")
    sd_pipe.model.set_conditioning(embed, t)
    with torch.no_grad():
        clean = sd_pipe.autoencoder.encode(input_image)
        alpha_t = sd_pipe.pipe.scheduler.alphas_cumprod[t]
        noise = torch.randn_like(clean)
        noisy = alpha_t.sqrt() * clean + (1 - alpha_t).sqrt() * noise
        x_pred = sd_pipe.model(noisy, torch.tensor([sigma], device=sd_pipe.device))
    if not torch.isfinite(x_pred).all():
        return r.fail("x_pred NaN/Inf")
    force = -(noisy - x_pred) / (sigma ** 2 + 1e-8)
    force_norm = force.norm().item()
    if force_norm == 0:
        return r.fail("force is exactly zero")
    return r.ok(f"force_norm={force_norm:.4f}")


# =========================================================================== #
# T7: DDIM Backward ONLY (no inversion — isolates denoise correctness)
# =========================================================================== #

def test_T7_ddim_denoise_only(sd_pipe: SDPipeline, input_image: Image.Image,
                               prompt: str) -> TestResult:
    """
    Add noise via scheduler.add_noise (mathematically exact),
    then denoise via ddim_backward.
    
    This isolates ddim_backward from any inversion error.
    If this fails, ddim_backward has a bug.
    
    Expected:
      noise=0.3 with prompt: >25dB  (moderate noise, good prompt → easy)
      noise=0.6 with prompt: >20dB  (heavy noise, but UNet should recover structure)
      noise=0.3 uncond:      >22dB  (moderate noise, no guidance → harder)
      noise=0.6 uncond:      >14dB  (heavy noise, no guidance → just above random)
    """
    r = TestResult("T7  DDIM backward only (add_noise → denoise)")
    
    if not prompt:
        prompt = "a photo"  # minimal fallback
    
    failures = []
    strip_images = [input_image]
    strip_labels = ["original"]
    
    configs = [
        # (noise_level, cfg_scale, embed_prompt, label, threshold)
        (0.3, 0.0,  "",     "uncond n=0.3", 22.0),
        (0.6, 0.0,  "",     "uncond n=0.6", 14.0),
        (0.3, 1.0,  prompt, "cond n=0.3",   25.0),
        (0.6, 1.0,  prompt, "cond n=0.6",   20.0),
    ]
    
    with torch.no_grad():
        latent_clean = sd_pipe.autoencoder.encode(input_image)
        
        for noise_level, cfg, p, label, threshold in configs:
            embed = sd_pipe.encode_prompt(p) if p else sd_pipe.encode_prompt("")
            t = sd_pipe.get_timestep(noise_level)
            
            # Exact noising (no approximation, no inversion)
            noise = torch.randn_like(latent_clean)
            latent_noisy = sd_pipe.pipe.scheduler.add_noise(latent_clean, noise, t)
            
            # Denoise
            latent_recon = sd_pipe.ddim_backward(latent_noisy, embed,
                                                  noise_level, cfg_scale=cfg)
            recon_img = sd_pipe.autoencoder.decode(latent_recon)
            
            score = psnr(input_image, recon_img)
            strip_images.append(recon_img)
            strip_labels.append(f"{label}\nPSNR={score:.1f}dB")
            
            if score < threshold:
                failures.append(f"{label}: {score:.1f}dB < {threshold}dB")
    
    make_comparison_strip(
        strip_images, strip_labels,
        os.path.join(OUTPUT_DIR, "T7_ddim_denoise_only.png"),
    )
    
    if failures:
        return r.fail(" | ".join(failures))
    return r.ok("all denoise-only tests passed")


# =========================================================================== #
# T8: DDIM Forward+Backward Roundtrip (unconditional, cfg=0)
# =========================================================================== #

def test_T8_ddim_inversion_uncond(sd_pipe: SDPipeline,
                                   input_image: Image.Image) -> TestResult:
    """
    DDIM forward inversion then backward, both UNCONDITIONAL (cfg=0).
    
    With cfg=0, DDIM inversion should be mathematically the exact inverse
    of DDIM backward (same UNet call, same alpha schedule, reversed order).
    The only error source is floating-point discretization.
    
    Expected:
      noise=0.2: >30dB (few steps, minimal accumulation)
      noise=0.4: >25dB (moderate steps)
      noise=0.6: >18dB (many steps, some float accumulation — lenient)
    """
    r = TestResult("T8  DDIM inversion roundtrip (uncond, cfg=0)")
    
    embed_uncond = sd_pipe.encode_prompt("")
    failures = []
    strip_images = [input_image]
    strip_labels = ["original"]
    
    configs = [
        (0.2, 30.0),
        (0.4, 25.0),
        (0.6, 18.0),
    ]
    
    with torch.no_grad():
        latent_clean = sd_pipe.autoencoder.encode(input_image)
        
        for noise_level, threshold in configs:
            latent_noisy = sd_pipe.ddim_forward(latent_clean, embed_uncond,
                                                 noise_level, cfg_scale=0.0)
            latent_recon = sd_pipe.ddim_backward(latent_noisy, embed_uncond,
                                                  noise_level, cfg_scale=0.0)
            recon_img = sd_pipe.autoencoder.decode(latent_recon)
            
            score = psnr(input_image, recon_img)
            lat_err = (latent_recon - latent_clean).norm() / (latent_clean.norm() + 1e-8)
            
            strip_images.append(recon_img)
            strip_labels.append(f"uncond n={noise_level}\n"
                               f"PSNR={score:.1f}dB\n"
                               f"lat_err={lat_err:.3f}")
            
            if score < threshold:
                failures.append(f"n={noise_level}: {score:.1f}dB < {threshold}dB "
                              f"(lat_err={lat_err:.3f})")
    
    make_comparison_strip(
        strip_images, strip_labels,
        os.path.join(OUTPUT_DIR, "T8_ddim_inversion_uncond.png"),
    )
    
    if failures:
        return r.fail(" | ".join(failures))
    return r.ok("all unconditional roundtrips passed")


# =========================================================================== #
# T9: DDIM Roundtrip with Text Inversion (realistic pipeline test)
# =========================================================================== #

def test_T9_ddim_with_text_inversion(sd_pipe: SDPipeline,
                                      input_image: Image.Image,
                                      prompt: str) -> TestResult:
    """
    Full pipeline test: text inversion → DDIM forward → DDIM backward.
    
    Text inversion optimizes the embedding to faithfully reconstruct
    this specific image. This is how Yu et al. actually use the pipeline.
    
    Expected:
      noise=0.6 with text_inv: >22dB (this is the realistic operating point)
    
    SLOW: ~60s for 500-step text inversion.
    """
    r = TestResult("T9  DDIM roundtrip with text inversion")
    
    if not prompt:
        return r.fail("need --prompt for T9")
    
    failures = []
    strip_images = [input_image]
    strip_labels = ["original"]
    
    with torch.no_grad():
        latent_clean = sd_pipe.autoencoder.encode(input_image)
    
    # Text inversion (requires grad)
    print("    Running text inversion (500 steps)...")
    embed_inv = sd_pipe.text_inversion(prompt, latent_clean, steps=500, lr=0.005)
    
    configs = [
        (0.3, 0.5, 26.0),
        (0.6, 0.5, 22.0),
    ]
    
    with torch.no_grad():
        for noise_level, cfg, threshold in configs:
            latent_noisy = sd_pipe.ddim_forward(latent_clean, embed_inv,
                                                 noise_level, cfg_scale=cfg)
            latent_recon = sd_pipe.ddim_backward(latent_noisy, embed_inv,
                                                  noise_level, cfg_scale=cfg)
            recon_img = sd_pipe.autoencoder.decode(latent_recon)
            
            score = psnr(input_image, recon_img)
            strip_images.append(recon_img)
            strip_labels.append(f"text_inv n={noise_level}\nPSNR={score:.1f}dB")
            
            if score < threshold:
                failures.append(f"n={noise_level}: {score:.1f}dB < {threshold}dB")
    
    make_comparison_strip(
        strip_images, strip_labels,
        os.path.join(OUTPUT_DIR, "T9_ddim_text_inversion.png"),
    )
    
    if failures:
        return r.fail(" | ".join(failures))
    return r.ok("text inversion roundtrips passed")


# =========================================================================== #
# T10: compute_force interface test (GFM compatibility)
# =========================================================================== #

def test_T10_compute_force_interface(sd_pipe: SDPipeline,
                                      input_image: Image.Image) -> TestResult:
    """
    Verify that SDModel works correctly with GFM's compute_force pattern:
        x_pred = model(latent, sigma_tensor, class_labels)
        force = -(latent - x_pred) / sigma^2
    
    Checks:
      1. force is finite and non-zero
      2. force has the same shape as input
      3. force projected to tangent space (sphere) is non-zero
         (i.e., it has a component that's not purely radial)
    """
    r = TestResult("T10 compute_force GFM interface")
    
    t = sd_pipe.get_timestep(0.6)
    sigma = sd_pipe.get_sigma_from_timestep(t)
    embed = sd_pipe.encode_prompt("a photo")
    sd_pipe.model.set_conditioning(embed, t)
    
    with torch.no_grad():
        clean = sd_pipe.autoencoder.encode(input_image)
        alpha_t = sd_pipe.pipe.scheduler.alphas_cumprod[t]
        noise = torch.randn_like(clean)
        noisy = alpha_t.sqrt() * clean + (1 - alpha_t).sqrt() * noise
        
        # Simulate what BaseInterpolator.compute_force does
        sigma_tensor = torch.full((noisy.shape[0],), sigma, device=sd_pipe.device)
        x_pred = sd_pipe.model(noisy, sigma_tensor, None)
        force = -(noisy - x_pred) / (sigma ** 2)
    
    # Check 1: finite and non-zero
    if not torch.isfinite(force).all():
        return r.fail("force contains NaN/Inf")
    if force.norm().item() == 0:
        return r.fail("force is exactly zero")
    
    # Check 2: shape matches
    if force.shape != noisy.shape:
        return r.fail(f"shape mismatch: force={force.shape} vs input={noisy.shape}")
    
    # Check 3: tangent component exists (for sphere constraint)
    # Project force onto tangent space of sphere at noisy
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
            f"tangent_frac={tangent_norm/total_norm:.3f}")
    return r.ok(info)


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="SD wrapper correctness tests")
    parser.add_argument("--img", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for the image (used in T7 cond, T9)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_text_inv", action="store_true",
                        help="Include T9 (text inversion test, slow ~60s)")
    args = parser.parse_args()

    print("=" * 65)
    print("  SD Wrapper Correctness Tests")
    print(f"  output dir: {OUTPUT_DIR}")
    print("=" * 65)

    if args.img and os.path.exists(args.img):
        input_image = Image.open(args.img).convert("RGB")
        print(f"\nUsing real image: {args.img}")
    else:
        input_image = make_test_image()
        print("\nNo real image — using synthetic test image.")
        input_image.save(os.path.join(OUTPUT_DIR, "synthetic_input.png"))

    print(f"\nLoading SD pipeline (device={args.device})…")
    try:
        sd_pipe = SDPipeline.load(device=args.device)
        print("Pipeline loaded.\n")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    autoenc = sd_pipe.autoencoder

    # Build test list
    tests = [
        ("T1",  lambda: test_T1_preprocess_postprocess(autoenc)),
        ("T2",  lambda: test_T2_vae_roundtrip(autoenc, input_image)),
        ("T3",  lambda: test_T3_sphere_projection()),
        ("T4",  lambda: test_T4_timestep_sigma_ordering(sd_pipe)),
        ("T5",  lambda: test_T5_sdmodel_denoise_mode(sd_pipe, input_image)),
        ("T6",  lambda: test_T6_sdmodel_nfsd_mode(sd_pipe, input_image)),
        ("T7",  lambda: test_T7_ddim_denoise_only(sd_pipe, input_image, args.prompt)),
        ("T8",  lambda: test_T8_ddim_inversion_uncond(sd_pipe, input_image)),
        ("T10", lambda: test_T10_compute_force_interface(sd_pipe, input_image)),
    ]
    
    if args.test_text_inv:
        tests.append(
            ("T9", lambda: test_T9_ddim_with_text_inversion(
                sd_pipe, input_image, args.prompt))
        )
    else:
        print("  (Skipping T9 — pass --test_text_inv to include)")

    results = []
    for name, fn in tests:
        print(f"\nRunning {name}…")
        try:
            res = fn()
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
        print("    T7 fail → ddim_backward has a bug")
        print("    T8 fail at n=0.2/0.4 → ddim_forward has a bug")
        print("    T8 fail only at n=0.6 → expected discretization error (lenient)")
        print("    T9 fail → text inversion not helping enough (check lr/steps)")
        print("    T10 fail → SDModel interface incompatible with GFM")
    
    print("=" * 65)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
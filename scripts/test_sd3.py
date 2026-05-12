"""
Correctness tests for sd3_wrapper.py

Tests:
  T1  preprocess / postprocess roundtrip
  T2  VAE encode → decode roundtrip
  T3  project_to_sphere norm invariant
  T4  get_sigma_from_noise_level monotone / identity
  T5  SD3Model.forward (denoise mode)
  T6  SD3Model.forward (nfsd mode)
  T7  flow_backward only (exact noising → denoise)
  T8  flow_forward + flow_backward roundtrip (uncond, cfg=0)
  T9  flow_forward + flow_backward with text inversion
  T10 compute_force GFM interface
  T11 VAE normalization (shift + scale consistency)

Run:
    python test_sd3_correctness.py [--img path.jpg] [--device cuda]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse, math, os, sys, traceback
import numpy as np
import torch
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gfm.path.sd3_wrapper import SD3Autoencoder, SD3Model, SD3GFMPipeline

OUTPUT_DIR = os.path.join(ROOT_DIR, "test_outputs", "sd3_correctness")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================================== #
# Helpers
# =========================================================================== #

class TestResult:
    def __init__(self, name):
        self.name, self.passed, self.message = name, False, ""
    def ok(self, msg=""):
        self.passed, self.message = True, msg
        print(f"  [\033[32mPASS\033[0m] {self.name}" + (f": {msg}" if msg else ""))
        return self
    def fail(self, msg=""):
        self.passed, self.message = False, msg
        print(f"  [\033[31mFAIL\033[0m] {self.name}: {msg}")
        return self

def psnr(a, b):
    w, h = min(a.width, b.width), min(a.height, b.height)
    a = np.array(a.convert("RGB").resize((w, h), Image.LANCZOS), dtype=np.float32)
    b = np.array(b.convert("RGB").resize((w, h), Image.LANCZOS), dtype=np.float32)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))

def make_strip(images, labels, path):
    W, H, pad = 512, 512, 30
    strip = Image.new("RGB", (W * len(images), H + pad), (240, 240, 240))
    draw = ImageDraw.Draw(strip)
    for i, (img, lbl) in enumerate(zip(images, labels)):
        strip.paste(img.convert("RGB").resize((W, H)), (i * W, pad))
        draw.text((i * W + 5, 5), lbl, fill=(0, 0, 0))
    strip.save(path)
    print(f"    → {path}")

def make_test_image(size=(512, 512)):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            arr[y, x] = [int(255*x/size[0]), int(255*y/size[1]),
                         int(255*(x+y)/(size[0]+size[1]))]
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    cx, cy, r = size[0]//2, size[1]//2, size[0]//6
    d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 255, 255))
    return img

def flow_add_noise(x0, t):
    noise = torch.randn_like(x0)
    return (1.0 - t) * x0 + t * noise, noise


# =========================================================================== #
# Tests
# =========================================================================== #

def test_T1(autoenc):
    r = TestResult("T1  preprocess ↔ postprocess roundtrip")
    img = make_test_image()
    tensor = autoenc._preprocess(img)
    if tensor.min().item() < -1.05 or tensor.max().item() > 1.05:
        return r.fail(f"range [{tensor.min():.3f}, {tensor.max():.3f}]")
    recovered = autoenc._postprocess(tensor)
    img_resized = img.resize((autoenc.resolution, autoenc.resolution), Image.LANCZOS)
    err = np.abs(np.array(img_resized).astype(int) - np.array(recovered).astype(int))
    if err.max() > 2:
        return r.fail(f"max_err={err.max()}")
    return r.ok(f"max_err={err.max()}, mean={err.mean():.4f}")

def test_T2(autoenc, img):
    r = TestResult("T2  VAE encode → decode roundtrip")
    with torch.no_grad():
        lat = autoenc.encode(img)
        recon = autoenc.decode(lat)
    if not torch.isfinite(lat).all():
        return r.fail("NaN/Inf in latent")
    if lat.shape[1] != 16:
        return r.fail(f"expected 16 channels, got {lat.shape[1]}")
    score = psnr(img, recon)
    make_strip([img, recon], ["original", f"decoded PSNR={score:.1f}dB"],
               os.path.join(OUTPUT_DIR, "T2_vae.png"))
    return r.fail(f"PSNR={score:.1f}dB < 20") if score < 20 else r.ok(f"PSNR={score:.1f}dB, shape={list(lat.shape)}")

def test_T3():
    r = TestResult("T3  project_to_sphere")
    for shape, radius in [((1, 16, 128, 128), 15.0), ((3, 16, 128, 128), 100.0)]:
        x = torch.randn(*shape)
        proj = SD3GFMPipeline.project_to_sphere(x, radius)
        err = (proj.view(shape[0], -1).norm(dim=-1) - radius).abs().max().item()
        if err > 1e-4:
            return r.fail(f"err={err:.6f}")
    return r.ok("all within 1e-4")

def test_T4(pipe):
    r = TestResult("T4  noise_level → sigma identity mapping")
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    sigmas = [pipe.get_sigma_from_noise_level(l) for l in levels]
    # Should be identity: sigma == noise_level
    identity_ok = all(abs(s - l) < 1e-6 for s, l in zip(sigmas, levels))
    mono_ok = all(sigmas[i] <= sigmas[i+1] for i in range(len(sigmas)-1))
    info = f"σ={[f'{s:.4f}' for s in sigmas]}"
    if not identity_ok:
        return r.fail(f"not identity  {info}")
    if not mono_ok:
        return r.fail(f"not monotone  {info}")
    return r.ok(info)

def test_T5(pipe, img):
    r = TestResult("T5  SD3Model.forward (denoise mode)")
    orig = pipe.model.mode
    pipe.model.mode = "denoise"
    t_val = 0.4
    embeds, pooled = pipe.encode_prompt("a photo")
    pipe.model.set_conditioning(embeds, pooled, t_val)
    with torch.no_grad():
        clean = pipe.autoencoder.encode(img)
        noisy, _ = flow_add_noise(clean, t_val)
        x_pred = pipe.model(noisy, torch.tensor([t_val], device=pipe.device))
    pipe.model.mode = orig
    if not torch.isfinite(x_pred).all():
        return r.fail("NaN/Inf")
    ratio = x_pred.norm().item() / (clean.norm().item() + 1e-8)
    info = f"ratio={ratio:.2f}, x_pred_norm={x_pred.norm():.1f}"
    return r.fail(f"scale suspicious {info}") if ratio > 20 or ratio < 0.05 else r.ok(info)

def test_T6(pipe, img):
    r = TestResult("T6  SD3Model.forward (nfsd mode)")
    t_val = 0.6
    embeds, pooled = pipe.encode_prompt("a realistic photo")
    pipe.model.set_conditioning(embeds, pooled, t_val)
    with torch.no_grad():
        clean = pipe.autoencoder.encode(img)
        noisy, _ = flow_add_noise(clean, t_val)
        x_pred = pipe.model(noisy, torch.tensor([t_val], device=pipe.device))
    if not torch.isfinite(x_pred).all():
        return r.fail("NaN/Inf")
    force_norm = (-(noisy - x_pred) / (t_val**2 + 1e-8)).norm().item()
    return r.fail("zero force") if force_norm == 0 else r.ok(f"force_norm={force_norm:.4f}")

def test_T7(pipe, img, prompt):
    r = TestResult("T7  flow_backward only (exact noising → denoise)")
    if not prompt: prompt = "a photo"
    failures, strip_imgs, strip_lbls = [], [img], ["original"]
    configs = [
        (0.3, 0.0, "",     "uncond t=0.3", 18.0),
        (0.6, 0.0, "",     "uncond t=0.6", 12.0),
        (0.3, 0.0, prompt, "cond t=0.3",   20.0),
        (0.6, 0.0, prompt, "cond t=0.6",   16.0),
    ]
    with torch.no_grad():
        lat_clean = pipe.autoencoder.encode(img)
        for nl, cfg, p, label, thresh in configs:
            embeds, pooled = pipe.encode_prompt(p if p else "")
            noisy, _ = flow_add_noise(lat_clean, nl)
            recon = pipe.flow_backward(noisy, embeds, pooled, nl, cfg)
            recon_img = pipe.autoencoder.decode(recon)
            score = psnr(img, recon_img)
            strip_imgs.append(recon_img)
            strip_lbls.append(f"{label}\n{score:.1f}dB")
            if score < thresh:
                failures.append(f"{label}: {score:.1f}<{thresh}")
    make_strip(strip_imgs, strip_lbls, os.path.join(OUTPUT_DIR, "T7_denoise.png"))
    return r.fail(" | ".join(failures)) if failures else r.ok("all passed")

def test_T8(pipe, img):
    r = TestResult("T8  flow inversion roundtrip (uncond, cfg=0)")
    embeds, pooled = pipe.encode_prompt("")
    failures, strip_imgs, strip_lbls = [], [img], ["original"]
    for nl, thresh in [(0.2, 25.0), (0.4, 20.0), (0.6, 14.0)]:
        with torch.no_grad():
            lat = pipe.autoencoder.encode(img)
            noisy = pipe.flow_forward(lat, embeds, pooled, nl, 0.0)
            recon = pipe.flow_backward(noisy, embeds, pooled, nl, 0.0)
            recon_img = pipe.autoencoder.decode(recon)
        score = psnr(img, recon_img)
        lat_err = (recon - lat).norm() / (lat.norm() + 1e-8)
        strip_imgs.append(recon_img)
        strip_lbls.append(f"t={nl}\n{score:.1f}dB\nerr={lat_err:.3f}")
        if score < thresh:
            failures.append(f"t={nl}: {score:.1f}<{thresh}")
    make_strip(strip_imgs, strip_lbls, os.path.join(OUTPUT_DIR, "T8_inversion.png"))
    return r.fail(" | ".join(failures)) if failures else r.ok("all passed")

def test_T9(pipe, img, prompt):
    r = TestResult("T9  flow roundtrip with text inversion")
    if not prompt: return r.fail("need --prompt")
    with torch.no_grad():
        lat = pipe.autoencoder.encode(img)
    print("    Text inversion (500 steps)...")
    embed_inv, pooled_inv = pipe.text_inversion(prompt, lat, steps=500, lr=0.005)
    failures, strip_imgs, strip_lbls = [], [img], ["original"]
    for nl, cfg, thresh in [(0.3, 0.0, 22.0), (0.6, 0.0, 18.0)]:
        with torch.no_grad():
            noisy = pipe.flow_forward(lat, embed_inv, pooled_inv, nl, cfg)
            recon = pipe.flow_backward(noisy, embed_inv, pooled_inv, nl, cfg)
            recon_img = pipe.autoencoder.decode(recon)
        score = psnr(img, recon_img)
        strip_imgs.append(recon_img)
        strip_lbls.append(f"ti t={nl}\n{score:.1f}dB")
        if score < thresh:
            failures.append(f"t={nl}: {score:.1f}<{thresh}")
    make_strip(strip_imgs, strip_lbls, os.path.join(OUTPUT_DIR, "T9_textinv.png"))
    return r.fail(" | ".join(failures)) if failures else r.ok("passed")

def test_T10(pipe, img):
    r = TestResult("T10 compute_force GFM interface")
    t_val = 0.6
    embeds, pooled = pipe.encode_prompt("a photo")
    pipe.model.set_conditioning(embeds, pooled, t_val)
    with torch.no_grad():
        clean = pipe.autoencoder.encode(img)
        noisy, _ = flow_add_noise(clean, t_val)
        sigma_t = torch.full((noisy.shape[0],), t_val, device=pipe.device)
        x_pred = pipe.model(noisy, sigma_t, None)
        force = -(noisy - x_pred) / (t_val ** 2)
    if not torch.isfinite(force).all(): return r.fail("NaN/Inf")
    if force.norm().item() == 0: return r.fail("zero force")
    if force.shape != noisy.shape: return r.fail(f"shape mismatch")
    x_flat = noisy.view(1, -1)
    f_flat = force.view(1, -1)
    x_hat = x_flat / (x_flat.norm(dim=-1, keepdim=True) + 1e-8)
    radial = (f_flat * x_hat).sum(dim=-1, keepdim=True) * x_hat
    tangent = (f_flat - radial).norm().item()
    total = f_flat.norm().item()
    if tangent < 1e-10: return r.fail("purely radial")
    return r.ok(f"total={total:.4f}, tangent_frac={tangent/total:.3f}")

def test_T11(autoenc, img):
    r = TestResult("T11 VAE normalization consistency")
    with torch.no_grad():
        tensor = autoenc._preprocess(img)
        raw = autoenc.vae.encode(tensor)['latent_dist'].mean
        lat_manual = (raw - autoenc.shift_factor) * autoenc.scaling_factor
        lat_api = autoenc.encode(img)
        enc_err = (lat_manual - lat_api).abs().max().item()
        if enc_err > 1e-4: return r.fail(f"encode err={enc_err:.6f}")
        raw_back = lat_api / autoenc.scaling_factor + autoenc.shift_factor
        img_manual = autoenc.vae.decode(raw_back)['sample']
        img_api = autoenc.decode(lat_api)
        img_api_t = autoenc._preprocess(img_api)
        dec_err = (img_manual - img_api_t).abs().mean().item()
    info = f"enc_err={enc_err:.6f}, dec_err={dec_err:.6f}, shift={autoenc.shift_factor}, scale={autoenc.scaling_factor}"
    return r.fail(info) if dec_err > 0.02 else r.ok(info)


# =========================================================================== #
# Main
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_id", type=str,
                        default="stabilityai/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--cache_dir", type=str,
                        default="/cns/USERS/zzhixuan/weights")
    parser.add_argument("--keep_t5", action="store_true")
    parser.add_argument("--test_text_inv", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  SD3 Wrapper Correctness Tests")
    print(f"  output: {OUTPUT_DIR}")
    print("=" * 60)

    if args.img and os.path.exists(args.img):
        img = Image.open(args.img).convert("RGB")
        print(f"\nUsing: {args.img}")
    else:
        img = make_test_image()
        print("\nUsing synthetic image.")
        img.save(os.path.join(OUTPUT_DIR, "synthetic.png"))

    # Offline test
    print("\n--- Offline tests ---")
    offline = [test_T3()]

    print(f"\nLoading SD3 ({args.model_id})...")
    try:
        pipe = SD3GFMPipeline.load(
            model_id=args.model_id, device=args.device,
            cache_dir=args.cache_dir, load_blip=False,
            resolution=args.resolution, drop_t5=not args.keep_t5,
        )
        print("Loaded.\n")
    except Exception:
        traceback.print_exc()
        for r in offline:
            s = "\033[32m✓\033[0m" if r.passed else "\033[31m✗\033[0m"
            print(f"  {s}  {r.name}: {r.message}")
        sys.exit(1)

    tests = [
        *[("off", r) for r in offline],
        ("T1",  lambda: test_T1(pipe.autoencoder)),
        ("T2",  lambda: test_T2(pipe.autoencoder, img)),
        ("T4",  lambda: test_T4(pipe)),
        ("T5",  lambda: test_T5(pipe, img)),
        ("T6",  lambda: test_T6(pipe, img)),
        ("T7",  lambda: test_T7(pipe, img, args.prompt)),
        ("T8",  lambda: test_T8(pipe, img)),
        ("T10", lambda: test_T10(pipe, img)),
        ("T11", lambda: test_T11(pipe.autoencoder, img)),
    ]
    if args.test_text_inv:
        tests.append(("T9", lambda: test_T9(pipe, img, args.prompt)))
    else:
        print("  (Skipping T9 — pass --test_text_inv)")

    results = []
    for name, fn_or_r in tests:
        if name == "off":
            results.append(fn_or_r); continue
        print(f"\nRunning {name}…")
        try:
            results.append(fn_or_r())
        except Exception as e:
            r = TestResult(name); r.fail(f"EXCEPTION: {e}")
            traceback.print_exc(); results.append(r)

    print("\n" + "=" * 60)
    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass
    for r in results:
        s = "\033[32m✓\033[0m" if r.passed else "\033[31m✗\033[0m"
        print(f"  {s}  {r.name}: {r.message}")
    print(f"\n  {n_pass}/{len(results)} passed, {n_fail} failed")
    print(f"  Visuals: {OUTPUT_DIR}")
    if n_fail:
        print("\n  Diagnostic:")
        print("    T5 fail  → x₀ = z_t - t·v formula wrong")
        print("    T6 fail  → NFSD grad_v computation wrong")
        print("    T7 fail  → flow_backward ODE bug")
        print("    T8 fail  → flow_forward ODE bug")
        print("    T10 fail → GFM interface mismatch")
        print("    T11 fail → VAE shift/scale encode↔decode mismatch")
    print("=" * 60)
    sys.exit(0 if n_fail == 0 else 1)

if __name__ == "__main__":
    main()
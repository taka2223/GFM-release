"""Batch-run SD3 GFM interpolation on MorphBench.

Loads SD3 + BLIP-large ONCE, then iterates over all <name>_0.png / <name>_1.png
pairs in the given subset directories. Frames + strip + opt_curves are saved
to <out_root>/<subset>/<name>/. Already-finished pairs are skipped (resume).

Usage:
    python scripts/run_morphbench.py
"""

import os
import sys
import glob
import json
import argparse
import torch
from PIL import Image

# Make `gfm` importable when running from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gfm.path.sd3_wrapper import (
    SD3GFMPipeline,
    run_gfm_interpolation,
    save_interpolation_strip,
    save_frames,
    save_optimization_curves,
)


def discover_pairs(subset_dir: str):
    """Find <name>_0.png / <name>_1.png pairs."""
    files = sorted(os.listdir(subset_dir))
    bases = set()
    for f in files:
        if f.endswith("_0.png"):
            base = f[:-len("_0.png")]
            if f"{base}_1.png" in files:
                bases.add(base)
    pairs = []
    for base in sorted(bases):
        pairs.append(
            (
                base,
                os.path.join(subset_dir, f"{base}_0.png"),
                os.path.join(subset_dir, f"{base}_1.png"),
            )
        )
    return pairs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_root", type=str, default="/cns/USERS/zzhixuan/data/MorphBench"
    )
    p.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["Metamorphosis", "Animation"],
    )
    p.add_argument("--out_root", type=str, default="./output/morphbench_sd3")
    p.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    p.add_argument("--cache_dir", type=str, default="/cns/USERS/zzhixuan/weights")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--noise_level", type=float, default=0.6)
    p.add_argument("--cfg_scale", type=float, default=0.0)
    p.add_argument("--lam", type=float, default=10.0)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--max_iters", type=int, default=400)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--keep_t5", action="store_true")
    p.add_argument(
        "--method",
        type=str,
        default="gfm",
        choices=["gfm", "slerp"],
        help="gfm = full optimization; slerp = baseline (no optimization)",
    )
    p.add_argument(
        "--limit", type=int, default=0, help="Stop after N pairs (debug)"
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    # --- collect pairs ---
    all_pairs = []
    for subset in args.subsets:
        sd = os.path.join(args.data_root, subset)
        if not os.path.isdir(sd):
            print(f"[WARN] {sd} not found, skipping")
            continue
        pairs = discover_pairs(sd)
        print(f"  {subset}: {len(pairs)} pairs")
        for name, a, b in pairs:
            all_pairs.append((subset, name, a, b))
    if args.limit > 0:
        all_pairs = all_pairs[: args.limit]
    print(f"Total: {len(all_pairs)} pairs")

    # --- load SD3 + BLIP ONCE ---
    print(f"Loading SD3 from {args.model_id} (BLIP-large captioner)...")
    pipe = SD3GFMPipeline.load(
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
        load_blip=True,
        resolution=args.resolution,
        drop_t5=not args.keep_t5,
    )

    # The captioner gets unloaded inside run_gfm_interpolation after use.
    # To reuse it across pairs we re-create per pair (cheap if model stays).
    # Easier: keep a reference and re-attach after each run.
    captioner = pipe.captioner

    for i, (subset, name, imgA_path, imgB_path) in enumerate(all_pairs):
        out_dir = os.path.join(args.out_root, subset, name)
        strip_path = os.path.join(out_dir, "strip.png")
        if os.path.exists(strip_path) and not args.overwrite:
            print(f"[{i+1}/{len(all_pairs)}] SKIP {subset}/{name} (done)")
            continue

        print(f"\n[{i+1}/{len(all_pairs)}] {subset}/{name}")
        os.makedirs(out_dir, exist_ok=True)

        imgA = Image.open(imgA_path)
        imgB = Image.open(imgB_path)

        # Re-attach captioner since previous run may have moved it to CPU.
        pipe.captioner = captioner
        if captioner is not None:
            captioner.model.to(args.device)

        results = run_gfm_interpolation(
            pipe,
            imgA,
            imgB,
            promptA=None,
            promptB=None,
            noise_level=args.noise_level,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            lam=args.lam,
            lr=args.lr,
            max_iters=args.max_iters,
            mode="nfsd",
            method=args.method,
        )

        save_interpolation_strip(
            results["images"], os.path.join(out_dir, "strip.png")
        )
        save_frames(results["images"], os.path.join(out_dir, "frames"))
        save_optimization_curves(
            losses=results["losses"],
            acc_norms=results["acc_norms"],
            f_norms=results["f_norms"],
            lam=args.lam,
            output_dir=out_dir,
        )
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "promptA": results["promptA"],
                    "promptB": results["promptB"],
                    "sphere_radius": results["sphere_radius"],
                    "sigma_eff": results["sigma_eff"],
                    "final_loss": (results["losses"] or [None])[-1],
                    "args": {
                        k: v for k, v in vars(args).items() if k != "out_root"
                    },
                },
                f,
                indent=2,
            )

    print(f"\nDone. Outputs at {args.out_root}")


if __name__ == "__main__":
    main()

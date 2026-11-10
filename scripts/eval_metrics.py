"""Evaluate MorphBench path length and transition uniformity with LPIPS.

For adjacent-frame LPIPS distances d_i, this script records:

    PPL      = sum_i d_i
    PDV-var  = population variance_i(d_i)
    PDV-std  = population standard deviation_i(d_i)

Both PDV conventions are retained because image-morphing papers have described
PDV as variance while some implementations report its square root.  Outputs
are written to ``<out_root>/lpips_per_pair.csv`` and
``<out_root>/lpips_summary.json``.
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="./output/morphbench_sd3")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gfm", "slerp"],
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["Metamorphosis", "Animation"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips_backbone", choices=["alex", "vgg"], default="alex")
    parser.add_argument("--expected_frames", type=int, default=17)
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def load_frames(frame_dir: str, device: str) -> torch.Tensor:
    paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(TF.to_tensor(image.convert("RGB")))
    if not frames:
        raise ValueError(f"No frames in {frame_dir}")
    # The lpips package expects RGB values in [-1, 1].
    return (torch.stack(frames).to(device) * 2.0) - 1.0


def summarize(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def paired_bootstrap(
    values_a: List[float],
    values_b: List[float],
    repeats: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    differences = np.asarray(values_a) - np.asarray(values_b)
    result = {
        "n": int(differences.size),
        "mean_difference_method_a_minus_method_b": float(differences.mean()),
    }
    if not differences.size or repeats <= 0:
        return result
    indices = rng.integers(0, differences.size, size=(repeats, differences.size))
    estimates = differences[indices].mean(axis=1)
    result["bootstrap_95_ci"] = [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]
    return result


def main():
    args = parse_args()
    import lpips

    if len(args.methods) < 2:
        raise ValueError("Paired evaluation requires at least two methods")
    method_a, method_b = args.methods[:2]

    print(f"Loading LPIPS/{args.lpips_backbone} on {args.device}...", flush=True)
    metric = lpips.LPIPS(net=args.lpips_backbone).eval().to(args.device)

    rows = []
    for method in args.methods:
        for subset in args.subsets:
            subset_dir = os.path.join(args.out_root, method, subset)
            if not os.path.isdir(subset_dir):
                print(f"[WARN] Missing {subset_dir}", flush=True)
                continue
            names = sorted(
                name
                for name in os.listdir(subset_dir)
                if os.path.isdir(os.path.join(subset_dir, name, "frames"))
            )
            for name in names:
                frame_dir = os.path.join(subset_dir, name, "frames")
                frames = load_frames(frame_dir, args.device)
                if frames.shape[0] != args.expected_frames:
                    print(
                        f"[SKIP] {method}/{subset}/{name}: "
                        f"{frames.shape[0]} frames, expected {args.expected_frames}",
                        flush=True,
                    )
                    continue
                with torch.inference_mode():
                    adjacent = (
                        metric(frames[:-1], frames[1:])
                        .reshape(-1)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                row = {
                    "method": method,
                    "subset": subset,
                    "name": name,
                    "n_frames": int(frames.shape[0]),
                    "ppl": float(adjacent.sum()),
                    "pdv_var": float(adjacent.var(ddof=0)),
                    "pdv_std": float(adjacent.std(ddof=0)),
                    "adjacent_lpips": adjacent.tolist(),
                }
                rows.append(row)
                print(
                    f"{method:5s} {subset:13s} {name:30s} "
                    f"PPL={row['ppl']:.6f} PDV-var={row['pdv_var']:.8f}",
                    flush=True,
                )

    csv_path = os.path.join(args.out_root, "lpips_per_pair.csv")
    os.makedirs(args.out_root, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "subset",
                "name",
                "n_frames",
                "PPL",
                "PDV_variance",
                "PDV_std",
                "adjacent_LPIPS",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["method"],
                    row["subset"],
                    row["name"],
                    row["n_frames"],
                    row["ppl"],
                    row["pdv_var"],
                    row["pdv_std"],
                    json.dumps(row["adjacent_lpips"]),
                ]
            )

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric_name in ("ppl", "pdv_var", "pdv_std"):
            grouped[(row["method"], row["subset"])][metric_name].append(
                row[metric_name]
            )
            grouped[(row["method"], "ALL")][metric_name].append(row[metric_name])

    summary = {
        "definitions": {
            "PPL": "sum of adjacent-frame LPIPS distances",
            "PDV_variance": "population variance of adjacent-frame LPIPS distances",
            "PDV_std": "population standard deviation of adjacent-frame LPIPS distances",
        },
        "expected_frames": args.expected_frames,
        "lpips_backbone": args.lpips_backbone,
        "groups": {},
        "paired_comparison": {
            "method_a": method_a,
            "method_b": method_b,
            "difference": "method_a minus method_b",
            "groups": {},
        },
    }
    for (method, subset), values_by_metric in sorted(grouped.items()):
        key = f"{method}/{subset}"
        summary["groups"][key] = {
            metric_name: summarize(values)
            for metric_name, values in values_by_metric.items()
    }

    rng = np.random.default_rng(args.seed)
    rows_by_key = {
        (row["method"], row["subset"], row["name"]): row for row in rows
    }
    for subset in [*args.subsets, "ALL"]:
        if subset == "ALL":
            pair_keys = sorted(
                (row["subset"], row["name"])
                for row in rows
                if row["method"] == method_a
                and (method_b, row["subset"], row["name"]) in rows_by_key
            )
        else:
            pair_keys = sorted(
                (row["subset"], row["name"])
                for row in rows
                if row["method"] == method_a
                and row["subset"] == subset
                and (method_b, row["subset"], row["name"]) in rows_by_key
            )
        summary["paired_comparison"]["groups"][subset] = {}
        for metric_name in ("ppl", "pdv_var", "pdv_std"):
            values_a = [
                rows_by_key[(method_a, pair_subset, name)][metric_name]
                for pair_subset, name in pair_keys
            ]
            values_b = [
                rows_by_key[(method_b, pair_subset, name)][metric_name]
                for pair_subset, name in pair_keys
            ]
            summary["paired_comparison"]["groups"][subset][metric_name] = (
                paired_bootstrap(
                    values_a,
                    values_b,
                    args.bootstrap_repeats,
                    rng,
                )
                if pair_keys
                else {"n": 0}
            )

    summary_path = os.path.join(args.out_root, "lpips_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved {csv_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()

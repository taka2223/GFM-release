"""Compute PPL / PDV / TOPIQ over MorphBench GFM outputs.

For each <out_root>/<subset>/<name>/frames/ directory containing 10 frame_*.png,
compute:
  PPL  = sum_i LPIPS(x_i, x_{i+1})           (path length)
  PDV  = std_i  LPIPS(x_i, x_{i+1})           (rate consistency)
  TOPIQ = mean_i TOPIQ_NR(x_i)                (per-frame quality)

Outputs:
  <out_root>/metrics.csv     (per-pair rows + per-subset means + global mean)

Usage:
  pip install pyiqa
  python scripts/eval_metrics.py --out_root ./output/morphbench_sd3
"""

import os
import argparse
import csv
import glob
import statistics
from typing import List

import torch
from PIL import Image
import torchvision.transforms.functional as TF


def load_frames(frame_dir: str, device: str) -> torch.Tensor:
    paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not paths:
        return None
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        t = TF.to_tensor(img)  # [3, H, W] in [0, 1]
        imgs.append(t)
    return torch.stack(imgs).to(device)  # [N, 3, H, W]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, default="./output/morphbench_sd3")
    p.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=["Metamorphosis", "Animation"],
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--topiq",
        type=str,
        default="topiq_nr",
        choices=["topiq_nr", "topiq_fr"],
        help="topiq_nr: no-reference; topiq_fr: full-reference",
    )
    p.add_argument(
        "--lpips_backbone",
        type=str,
        default="alex",
        choices=["alex", "vgg"],
    )
    return p.parse_args()


def main():
    args = parse_args()
    import pyiqa

    print(f"Loading LPIPS (backbone={args.lpips_backbone}) and {args.topiq}...")
    lpips = pyiqa.create_metric(
        "lpips", net=args.lpips_backbone, as_loss=False
    ).to(args.device)
    topiq = pyiqa.create_metric(args.topiq).to(args.device)

    rows = []
    per_subset = {s: {"ppl": [], "pdv": [], "topiq": []} for s in args.subsets}

    for subset in args.subsets:
        subset_dir = os.path.join(args.out_root, subset)
        if not os.path.isdir(subset_dir):
            print(f"[WARN] {subset_dir} not found")
            continue
        names = sorted(
            d
            for d in os.listdir(subset_dir)
            if os.path.isdir(os.path.join(subset_dir, d, "frames"))
        )
        for name in names:
            frame_dir = os.path.join(subset_dir, name, "frames")
            frames = load_frames(frame_dir, args.device)
            if frames is None or frames.shape[0] < 2:
                print(f"[SKIP] {subset}/{name}: <2 frames")
                continue

            with torch.no_grad():
                adj = []
                for i in range(frames.shape[0] - 1):
                    d = lpips(frames[i : i + 1], frames[i + 1 : i + 2]).item()
                    adj.append(d)
                ppl = float(sum(adj))
                pdv = float(statistics.stdev(adj)) if len(adj) > 1 else 0.0

                topiq_scores = []
                for i in range(frames.shape[0]):
                    topiq_scores.append(topiq(frames[i : i + 1]).item())
                topiq_mean = float(sum(topiq_scores) / len(topiq_scores))

            rows.append(
                {
                    "subset": subset,
                    "name": name,
                    "ppl": ppl,
                    "pdv": pdv,
                    "topiq": topiq_mean,
                    "n_frames": int(frames.shape[0]),
                }
            )
            per_subset[subset]["ppl"].append(ppl)
            per_subset[subset]["pdv"].append(pdv)
            per_subset[subset]["topiq"].append(topiq_mean)
            print(
                f"  {subset}/{name:30s}  PPL={ppl:.3f}  PDV={pdv:.4f}  TOPIQ={topiq_mean:.3f}"
            )

    csv_path = os.path.join(args.out_root, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "name", "PPL", "PDV", "TOPIQ", "n_frames"])
        for r in rows:
            w.writerow([r["subset"], r["name"], r["ppl"], r["pdv"], r["topiq"], r["n_frames"]])
        w.writerow([])
        w.writerow(["--- per-subset mean ---"])
        all_ppl, all_pdv, all_topiq = [], [], []
        for subset, agg in per_subset.items():
            if not agg["ppl"]:
                continue
            ppl_m = sum(agg["ppl"]) / len(agg["ppl"])
            pdv_m = sum(agg["pdv"]) / len(agg["pdv"])
            topiq_m = sum(agg["topiq"]) / len(agg["topiq"])
            w.writerow([subset, f"(n={len(agg['ppl'])})", ppl_m, pdv_m, topiq_m, ""])
            all_ppl.extend(agg["ppl"])
            all_pdv.extend(agg["pdv"])
            all_topiq.extend(agg["topiq"])
            print(
                f"\n{subset} mean over {len(agg['ppl'])} pairs: "
                f"PPL={ppl_m:.3f}  PDV={pdv_m:.4f}  TOPIQ={topiq_m:.3f}"
            )
        if all_ppl:
            w.writerow(
                [
                    "ALL",
                    f"(n={len(all_ppl)})",
                    sum(all_ppl) / len(all_ppl),
                    sum(all_pdv) / len(all_pdv),
                    sum(all_topiq) / len(all_topiq),
                    "",
                ]
            )
            print(
                f"\nALL  mean over {len(all_ppl)} pairs: "
                f"PPL={sum(all_ppl)/len(all_ppl):.3f}  "
                f"PDV={sum(all_pdv)/len(all_pdv):.4f}  "
                f"TOPIQ={sum(all_topiq)/len(all_topiq):.3f}"
            )

    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()

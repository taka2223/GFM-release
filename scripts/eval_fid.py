"""Compute FID(generated middle frames || endpoint images).

For each <out_root>/<subset>/<name>/frames/:
  reference   <- frame_000.png + frame_009.png   (the input endpoints, passed through)
  generated   <- frame_001.png ... frame_008.png (8 inner interpolated frames)

Reports FID per subset and global.

Usage:
    python scripts/eval_fid.py --out_root ./output/morphbench_sd3
"""

import os
import sys
import argparse
import glob
import shutil
import tempfile
from typing import List, Tuple


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
    return p.parse_args()


def collect_frames(out_root: str, subsets: List[str]) -> List[Tuple[str, str, List[str], List[str]]]:
    """For each (subset, pair), return (ref_paths, gen_paths)."""
    out = []
    for subset in subsets:
        sd = os.path.join(out_root, subset)
        if not os.path.isdir(sd):
            continue
        for name in sorted(os.listdir(sd)):
            fdir = os.path.join(sd, name, "frames")
            if not os.path.isdir(fdir):
                continue
            paths = sorted(glob.glob(os.path.join(fdir, "frame_*.png")))
            if len(paths) < 3:
                continue
            ref = [paths[0], paths[-1]]
            gen = paths[1:-1]
            out.append((subset, name, ref, gen))
    return out


def stage(paths: List[str], dst_dir: str, prefix: str):
    os.makedirs(dst_dir, exist_ok=True)
    for i, p in enumerate(paths):
        link = os.path.join(dst_dir, f"{prefix}_{i:05d}.png")
        if not os.path.exists(link):
            os.symlink(os.path.abspath(p), link)


def main():
    args = parse_args()
    import pyiqa

    print(f"Loading FID metric...")
    fid = pyiqa.create_metric("fid").to(args.device)

    pairs = collect_frames(args.out_root, args.subsets)
    print(f"Found {len(pairs)} pairs total")
    if not pairs:
        return

    base_tmp = tempfile.mkdtemp(prefix="fid_stage_")
    print(f"Staging frames in {base_tmp}")

    by_subset = {}
    for subset, name, ref, gen in pairs:
        by_subset.setdefault(subset, []).append((name, ref, gen))

    results = {}
    for subset, items in by_subset.items():
        ref_dir = os.path.join(base_tmp, f"{subset}_ref")
        gen_dir = os.path.join(base_tmp, f"{subset}_gen")
        for name, ref, gen in items:
            stage(ref, ref_dir, f"{name}_ref")
            stage(gen, gen_dir, f"{name}_gen")
        n_ref = len(glob.glob(os.path.join(ref_dir, "*.png")))
        n_gen = len(glob.glob(os.path.join(gen_dir, "*.png")))
        print(f"[{subset}] computing FID over {n_gen} gen vs {n_ref} ref ...")
        score = fid(gen_dir, ref_dir).item()
        print(f"  {subset}  FID = {score:.3f}")
        results[subset] = (score, n_gen, n_ref)

    # global
    all_ref = os.path.join(base_tmp, "ALL_ref")
    all_gen = os.path.join(base_tmp, "ALL_gen")
    for subset, items in by_subset.items():
        for name, ref, gen in items:
            stage(ref, all_ref, f"{subset}_{name}_ref")
            stage(gen, all_gen, f"{subset}_{name}_gen")
    n_ref = len(glob.glob(os.path.join(all_ref, "*.png")))
    n_gen = len(glob.glob(os.path.join(all_gen, "*.png")))
    print(f"[ALL] computing FID over {n_gen} gen vs {n_ref} ref ...")
    score_all = fid(all_gen, all_ref).item()
    print(f"  ALL  FID = {score_all:.3f}")
    results["ALL"] = (score_all, n_gen, n_ref)

    csv_path = os.path.join(args.out_root, "fid.csv")
    with open(csv_path, "w") as f:
        f.write("subset,FID,n_gen,n_ref\n")
        for k, (s, ng, nr) in results.items():
            f.write(f"{k},{s:.4f},{ng},{nr}\n")
    print(f"\nSaved {csv_path}")

    # cleanup staging
    try:
        shutil.rmtree(base_tmp)
    except Exception:
        pass


if __name__ == "__main__":
    main()

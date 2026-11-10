"""No-reference image-quality evaluation for MorphBench paths.

Scores are computed for every interior frame.  Statistical comparisons first
aggregate frames within a pair, then bootstrap pairs, so correlated frames from
one interpolation sequence are never treated as independent observations.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor


Pair = Tuple[str, str]


@dataclass(frozen=True)
class FrameRecord:
    method: str
    subset: str
    name: str
    frame_index: int
    path: str


class InteriorFrameDataset(Dataset):
    def __init__(self, records: Sequence[FrameRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = pil_to_tensor(image.convert("RGB")).float().div_(255.0)
        return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_root",
        default="./output/morphbench_c2f_sigma0p6_lambda20",
    )
    parser.add_argument("--methods", nargs="+", default=["gfm_c2f", "slerp"])
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["Metamorphosis", "Animation"],
    )
    parser.add_argument("--metric", default="musiq")
    parser.add_argument("--expected_frames", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--pair_names", nargs="*", default=None)
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Defaults to the metric name, e.g. musiq_per_frame.csv.",
    )
    return parser.parse_args()


def discover(
    out_root: str,
    methods: Sequence[str],
    subsets: Sequence[str],
    expected_frames: int,
    pair_names: Sequence[str] | None,
) -> Tuple[List[Pair], List[FrameRecord]]:
    requested = set(pair_names) if pair_names else None
    frames_by_key: Dict[Tuple[str, str, str], Tuple[str, ...]] = {}
    for method in methods:
        for subset in subsets:
            subset_dir = os.path.join(out_root, method, subset)
            if not os.path.isdir(subset_dir):
                continue
            for name in sorted(os.listdir(subset_dir)):
                if requested is not None and name not in requested:
                    continue
                frames = tuple(
                    sorted(
                        glob.glob(
                            os.path.join(
                                subset_dir,
                                name,
                                "frames",
                                "frame_*.png",
                            )
                        )
                    )
                )
                if len(frames) == expected_frames:
                    frames_by_key[(method, subset, name)] = frames

    common_pairs = sorted(
        {
            (subset, name)
            for _method, subset, name in frames_by_key
            if all(
                (method, subset, name) in frames_by_key for method in methods
            )
        }
    )
    if not common_pairs:
        raise RuntimeError("No complete pairs are shared by all methods")

    records = []
    for method in methods:
        for subset, name in common_pairs:
            frames = frames_by_key[(method, subset, name)]
            records.extend(
                FrameRecord(method, subset, name, index, frames[index])
                for index in range(1, expected_frames - 1)
            )
    return common_pairs, records


def extract_scores(
    records: Sequence[FrameRecord],
    metric_name: str,
    device: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, bool]:
    import pyiqa

    metric = pyiqa.create_metric(metric_name, device=device)
    loader = DataLoader(
        InteriorFrameDataset(records),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    chunks = []
    with torch.inference_mode():
        for batch_index, images in enumerate(loader):
            images = images.to(device, non_blocking=True)
            scores = metric(images).reshape(-1)
            chunks.append(scores.float().cpu().numpy())
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
                completed = min((batch_index + 1) * batch_size, len(records))
                print(
                    f"{metric_name}: {completed}/{len(records)} frames",
                    flush=True,
                )
    return np.concatenate(chunks), bool(metric.lower_better)


def aggregate_pairs(
    records: Sequence[FrameRecord],
    scores: np.ndarray,
    expected_frames: int,
) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Tuple[int, float]]] = {}
    for record, score in zip(records, scores):
        grouped.setdefault(
            (record.method, record.subset, record.name),
            [],
        ).append((record.frame_index, float(score)))

    center_low = int(np.ceil(0.25 * (expected_frames - 1)))
    center_high = int(np.floor(0.75 * (expected_frames - 1)))
    rows = []
    for (method, subset, name), indexed_scores in sorted(grouped.items()):
        indexed_scores.sort()
        all_values = np.asarray([score for _index, score in indexed_scores])
        center_values = np.asarray(
            [
                score
                for index, score in indexed_scores
                if center_low <= index <= center_high
            ]
        )
        rows.append(
            {
                "method": method,
                "subset": subset,
                "name": name,
                "interior_mean": float(all_values.mean()),
                "interior_median": float(np.median(all_values)),
                "center_mean": float(center_values.mean()),
                "center_median": float(np.median(center_values)),
            }
        )
    return rows


def summarize(
    pair_rows: Sequence[Mapping[str, object]],
    methods: Sequence[str],
    subsets: Sequence[str],
    lower_better: bool,
    bootstrap_repeats: int,
    seed: int,
) -> Dict[str, object]:
    metric_names = [
        "interior_mean",
        "interior_median",
        "center_mean",
        "center_median",
    ]
    rng = np.random.default_rng(seed)
    summary: Dict[str, object] = {}
    for scope in [*subsets, "ALL"]:
        rows = [
            row
            for row in pair_rows
            if scope == "ALL" or row["subset"] == scope
        ]
        if not rows:
            continue
        scope_summary: Dict[str, object] = {"methods": {}, "paired": {}}
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            scope_summary["methods"][method] = {
                "n_pairs": len(method_rows),
                "metrics": {
                    metric: {
                        "mean": float(
                            np.mean([float(row[metric]) for row in method_rows])
                        ),
                        "median": float(
                            np.median(
                                [float(row[metric]) for row in method_rows]
                            )
                        ),
                    }
                    for metric in metric_names
                },
            }

        if len(methods) == 2:
            left, right = methods
            keyed = {
                (str(row["method"]), str(row["subset"]), str(row["name"])): row
                for row in rows
            }
            pairs = sorted(
                {
                    (str(row["subset"]), str(row["name"]))
                    for row in rows
                    if (
                        (left, str(row["subset"]), str(row["name"])) in keyed
                        and (right, str(row["subset"]), str(row["name"])) in keyed
                    )
                }
            )
            for metric in metric_names:
                differences = np.asarray(
                    [
                        float(keyed[(left, subset, name)][metric])
                        - float(keyed[(right, subset, name)][metric])
                        for subset, name in pairs
                    ],
                    dtype=np.float64,
                )
                sampled_indices = rng.integers(
                    0,
                    differences.size,
                    size=(bootstrap_repeats, differences.size),
                )
                bootstrap_means = differences[sampled_indices].mean(axis=1)
                ci_low, ci_high = np.quantile(
                    bootstrap_means,
                    [0.025, 0.975],
                )
                if lower_better:
                    preferred_win_rate = np.mean(differences < 0.0)
                else:
                    preferred_win_rate = np.mean(differences > 0.0)
                scope_summary["paired"][metric] = {
                    "difference": f"{left} - {right}",
                    "mean_difference": float(differences.mean()),
                    "ci95": [float(ci_low), float(ci_high)],
                    "preferred_direction": (
                        "lower" if lower_better else "higher"
                    ),
                    "left_preferred_win_rate": float(preferred_win_rate),
                }
        summary[scope] = scope_summary
    return summary


def main() -> None:
    args = parse_args()
    common_pairs, records = discover(
        args.out_root,
        args.methods,
        args.subsets,
        args.expected_frames,
        args.pair_names,
    )
    print(
        f"Discovered {len(common_pairs)} pairs and "
        f"{len(records)} interior frames.",
        flush=True,
    )
    scores, lower_better = extract_scores(
        records,
        args.metric,
        args.device,
        args.batch_size,
        args.num_workers,
    )

    prefix = args.output_prefix or args.metric.replace("+", "plus")
    frame_path = os.path.join(args.out_root, f"{prefix}_per_frame.csv")
    with open(frame_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "subset", "name", "frame_index", "score"])
        for record, score in zip(records, scores):
            writer.writerow(
                [
                    record.method,
                    record.subset,
                    record.name,
                    record.frame_index,
                    float(score),
                ]
            )

    pair_rows = aggregate_pairs(records, scores, args.expected_frames)
    pair_path = os.path.join(args.out_root, f"{prefix}_per_pair.csv")
    with open(pair_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    summary = {
        "protocol": {
            "metric": args.metric,
            "lower_better": lower_better,
            "frames": "all interior frames",
            "center_region": "frame indices spanning t in [0.25, 0.75]",
            "bootstrap_unit": "pair",
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
            "n_common_pairs": len(common_pairs),
        },
        "scores": summarize(
            pair_rows,
            args.methods,
            args.subsets,
            lower_better,
            args.bootstrap_repeats,
            args.seed,
        ),
    }
    summary_path = os.path.join(args.out_root, f"{prefix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    print(f"Saved {frame_path}", flush=True)
    print(f"Saved {pair_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()

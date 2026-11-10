"""Pair-aware DINOv3 evaluation for MorphBench interpolation paths.

This evaluation deliberately does not compare the aggregate interior-frame
distribution with an aggregate endpoint distribution.  Instead, every frame
is evaluated relative to its own two endpoints in DINOv3 feature space.

The script reports:

* progress correlation and monotonicity violations;
* endpoint-triangle excess (a diagnostic for feature-space detours);
* feature path stretch, velocity variation, and second-order jerk;
* paired sharpness diagnostics computed on the same interior frames.

Per-pair values are written to CSV.  Summary means and pair-level bootstrap
confidence intervals for method differences are written to JSON.
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
from PIL import Image


Pair = Tuple[str, str]
FeatureKey = Tuple[str, str, str]

METRIC_DIRECTIONS = {
    "endpoint_angle": "descriptive",
    "progress_spearman": "higher",
    "reversal_count": "lower",
    "monotonic_violation": "lower",
    "raw_triangle_excess": "lower",
    "normalized_triangle_excess": "lower",
    "midpoint_balance": "lower",
    "path_length": "descriptive",
    "path_stretch": "descriptive",
    "velocity_mean": "descriptive",
    "velocity_cv": "lower",
    "feature_jerk": "lower",
    "laplacian_variance_median": "higher",
    "tenengrad_median": "higher",
}


@dataclass(frozen=True)
class SequenceRecord:
    method: str
    subset: str
    name: str
    frames: Tuple[str, ...]


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
    parser.add_argument(
        "--model_path",
        default=(
            "facebook/"
            "dinov3-vitl16-pretrain-lvd1689m"
        ),
    )
    parser.add_argument("--expected_frames", type=int, default=17)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--reversal_epsilon", type=float, default=0.01)
    parser.add_argument("--bootstrap_repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pair_names", nargs="*", default=None)
    parser.add_argument(
        "--cache_path",
        default=None,
        help="Defaults to <out_root>/dinov3_features.npz.",
    )
    parser.add_argument(
        "--force_features",
        action="store_true",
        help="Ignore an existing feature cache.",
    )
    return parser.parse_args()


def discover_sequences(
    out_root: str,
    methods: Sequence[str],
    subsets: Sequence[str],
    expected_frames: int,
    pair_names: Sequence[str] | None,
) -> Tuple[List[Pair], List[SequenceRecord]]:
    requested = set(pair_names) if pair_names else None
    by_key: Dict[FeatureKey, Tuple[str, ...]] = {}
    for method in methods:
        for subset in subsets:
            subset_dir = os.path.join(out_root, method, subset)
            if not os.path.isdir(subset_dir):
                continue
            for name in sorted(os.listdir(subset_dir)):
                if requested is not None and name not in requested:
                    continue
                frame_dir = os.path.join(subset_dir, name, "frames")
                frames = tuple(
                    sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
                )
                if len(frames) == expected_frames:
                    by_key[(method, subset, name)] = frames

    common_pairs = sorted(
        {
            (subset, name)
            for _method, subset, name in by_key
            if all((method, subset, name) in by_key for method in methods)
        }
    )
    if not common_pairs:
        raise RuntimeError("No complete pairs are shared by all requested methods")

    records = [
        SequenceRecord(method, subset, name, by_key[(method, subset, name)])
        for method in methods
        for subset, name in common_pairs
    ]
    return common_pairs, records


class FrameDataset:
    def __init__(self, paths: Sequence[str]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return image.convert("RGB")


def load_or_extract_features(
    records: Sequence[SequenceRecord],
    model_path: str,
    cache_path: str,
    batch_size: int,
    num_workers: int,
    device: str,
    dtype_name: str,
    force: bool,
) -> Dict[FeatureKey, np.ndarray]:
    expected_keys = [
        f"{record.method}\t{record.subset}\t{record.name}" for record in records
    ]
    if os.path.isfile(cache_path) and not force:
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_keys = cache["keys"].astype(str).tolist()
            if (
                cached_keys == expected_keys
                and str(cache["model_path"].item()) == os.path.abspath(model_path)
            ):
                stacked = cache["features"].astype(np.float32, copy=False)
                print(f"Loaded cached DINOv3 features from {cache_path}", flush=True)
                return {
                    tuple(key.split("\t")): stacked[index]
                    for index, key in enumerate(cached_keys)
                }
        print("Feature cache metadata differs; extracting again.", flush=True)

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoImageProcessor, AutoModel

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]
    processor = AutoImageProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
    ).to(device)
    model.eval()

    all_paths = [path for record in records for path in record.frames]
    dataset = FrameDataset(all_paths)

    def collate(images):
        return processor(images=images, return_tensors="pt")["pixel_values"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate,
    )
    chunks = []
    with torch.inference_mode():
        for batch_index, pixels in enumerate(loader):
            pixels = pixels.to(device=device, dtype=dtype, non_blocking=True)
            output = model(pixel_values=pixels)
            pooled = output.pooler_output
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            chunks.append(pooled.cpu().numpy())
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
                completed = min((batch_index + 1) * batch_size, len(dataset))
                print(
                    f"DINOv3 features: {completed}/{len(dataset)} frames",
                    flush=True,
                )

    flat = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    frame_count = len(records[0].frames)
    stacked = flat.reshape(len(records), frame_count, flat.shape[-1])
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(
        cache_path,
        keys=np.asarray(expected_keys),
        features=stacked,
        model_path=np.asarray(os.path.abspath(model_path)),
    )
    print(f"Saved DINOv3 feature cache to {cache_path}", flush=True)
    return {
        (record.method, record.subset, record.name): stacked[index]
        for index, record in enumerate(records)
    }


def angular_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    cosine = np.sum(left * right, axis=-1)
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(values: np.ndarray) -> float:
    time = np.arange(values.size, dtype=np.float64)
    ranked = rankdata(values.astype(np.float64))
    if np.std(ranked) == 0.0:
        return 0.0
    return float(np.corrcoef(time, ranked)[0, 1])


def sharpness_metrics(frame_paths: Sequence[str]) -> Tuple[float, float]:
    laplacian_variances = []
    tenengrad_values = []
    for path in frame_paths[1:-1]:
        with Image.open(path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32)

        laplacian = -4.0 * gray
        laplacian[1:, :] += gray[:-1, :]
        laplacian[:-1, :] += gray[1:, :]
        laplacian[:, 1:] += gray[:, :-1]
        laplacian[:, :-1] += gray[:, 1:]
        laplacian_variances.append(float(laplacian[1:-1, 1:-1].var()))

        grad_x = gray[:, 2:] - gray[:, :-2]
        grad_y = gray[2:, :] - gray[:-2, :]
        tenengrad = 0.5 * (
            np.mean(np.square(grad_x)) + np.mean(np.square(grad_y))
        )
        tenengrad_values.append(float(tenengrad))

    return (
        float(np.median(laplacian_variances)),
        float(np.median(tenengrad_values)),
    )


def sequence_metrics(
    features: np.ndarray,
    frame_paths: Sequence[str],
    reversal_epsilon: float,
) -> Dict[str, float]:
    features = features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True),
        1e-12,
    )
    first = np.broadcast_to(features[0], features.shape)
    last = np.broadcast_to(features[-1], features.shape)
    from_first = angular_distance(first, features)
    to_last = angular_distance(features, last)
    endpoint_angle = float(from_first[-1])
    denominator = np.maximum(from_first + to_last, 1e-12)
    progress = from_first / denominator

    progress_steps = np.diff(progress)
    monotonic_violation = float(np.mean(np.maximum(-progress_steps, 0.0)))
    reversal_count = int(np.sum(progress_steps < -reversal_epsilon))

    interior_excess = (
        from_first[1:-1] + to_last[1:-1] - endpoint_angle
    )
    mean_raw_excess = float(np.mean(np.maximum(interior_excess, 0.0)))
    normalized_excess = mean_raw_excess / max(endpoint_angle, 1e-6)

    velocities = angular_distance(features[:-1], features[1:])
    path_length = float(np.sum(velocities))
    path_stretch = path_length / max(endpoint_angle, 1e-6)
    velocity_mean = float(np.mean(velocities))
    velocity_cv = float(np.std(velocities) / max(velocity_mean, 1e-12))
    jerk = float(
        np.mean(
            np.linalg.norm(
                features[2:] - 2.0 * features[1:-1] + features[:-2],
                axis=1,
            )
        )
    )

    midpoint = len(features) // 2
    midpoint_balance = float(
        abs(from_first[midpoint] - to_last[midpoint])
        / max(from_first[midpoint] + to_last[midpoint], 1e-12)
    )
    laplacian, tenengrad = sharpness_metrics(frame_paths)
    return {
        "endpoint_angle": endpoint_angle,
        "progress_spearman": spearman(progress),
        "reversal_count": float(reversal_count),
        "monotonic_violation": monotonic_violation,
        "raw_triangle_excess": mean_raw_excess,
        "normalized_triangle_excess": normalized_excess,
        "midpoint_balance": midpoint_balance,
        "path_length": path_length,
        "path_stretch": path_stretch,
        "velocity_mean": velocity_mean,
        "velocity_cv": velocity_cv,
        "feature_jerk": jerk,
        "laplacian_variance_median": laplacian,
        "tenengrad_median": tenengrad,
    }


def percentile_interval(samples: np.ndarray) -> Tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def summarize(
    rows: Sequence[Mapping[str, object]],
    methods: Sequence[str],
    subsets: Sequence[str],
    metric_names: Sequence[str],
    bootstrap_repeats: int,
    seed: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    result: Dict[str, object] = {}
    scopes = [*subsets, "ALL"]
    for scope in scopes:
        scope_rows = [
            row for row in rows if scope == "ALL" or row["subset"] == scope
        ]
        if not scope_rows:
            continue
        scope_result: Dict[str, object] = {"methods": {}, "paired": {}}
        for method in methods:
            method_rows = [row for row in scope_rows if row["method"] == method]
            method_result = {}
            for metric in metric_names:
                values = np.asarray(
                    [float(row[metric]) for row in method_rows],
                    dtype=np.float64,
                )
                method_result[metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "median": float(np.median(values)),
                }
            scope_result["methods"][method] = {
                "n_pairs": len(method_rows),
                "metrics": method_result,
            }

        if len(methods) == 2:
            left, right = methods
            keyed = {
                (str(row["method"]), str(row["subset"]), str(row["name"])): row
                for row in scope_rows
            }
            pairs = sorted(
                {
                    (str(row["subset"]), str(row["name"]))
                    for row in scope_rows
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
                indices = rng.integers(
                    0,
                    differences.size,
                    size=(bootstrap_repeats, differences.size),
                )
                bootstrap_means = differences[indices].mean(axis=1)
                ci_low, ci_high = percentile_interval(bootstrap_means)
                direction = METRIC_DIRECTIONS[metric]
                if direction == "higher":
                    preferred_win_rate = float(np.mean(differences > 0.0))
                elif direction == "lower":
                    preferred_win_rate = float(np.mean(differences < 0.0))
                else:
                    preferred_win_rate = None
                scope_result["paired"][metric] = {
                    "difference": f"{left} - {right}",
                    "mean_difference": float(differences.mean()),
                    "ci95": [ci_low, ci_high],
                    "preferred_direction": direction,
                    "left_preferred_win_rate": preferred_win_rate,
                    "positive_difference_rate": float(
                        np.mean(differences > 0.0)
                    ),
                }
        result[scope] = scope_result
    return result


def main() -> None:
    args = parse_args()
    if len(args.methods) < 1:
        raise ValueError("At least one method is required")
    if args.expected_frames < 3:
        raise ValueError("--expected_frames must include an interior frame")

    common_pairs, records = discover_sequences(
        args.out_root,
        args.methods,
        args.subsets,
        args.expected_frames,
        args.pair_names,
    )
    print(
        f"Discovered {len(common_pairs)} common pairs and "
        f"{len(records)} method sequences.",
        flush=True,
    )

    cache_path = args.cache_path or os.path.join(
        args.out_root,
        "dinov3_features.npz",
    )
    features = load_or_extract_features(
        records,
        args.model_path,
        cache_path,
        args.batch_size,
        args.num_workers,
        args.device,
        args.dtype,
        args.force_features,
    )

    rows: List[Dict[str, object]] = []
    for record in records:
        metrics = sequence_metrics(
            features[(record.method, record.subset, record.name)],
            record.frames,
            args.reversal_epsilon,
        )
        rows.append(
            {
                "method": record.method,
                "subset": record.subset,
                "name": record.name,
                **metrics,
            }
        )

    metric_names = [
        key
        for key in rows[0]
        if key not in {"method", "subset", "name"}
    ]
    per_pair_path = os.path.join(args.out_root, "dinov3_per_pair.csv")
    with open(per_pair_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "subset", "name", *metric_names],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": {
            "model_path": os.path.abspath(args.model_path),
            "feature": "L2-normalized DINOv3 pooled output",
            "distance": "angular distance in radians",
            "expected_frames": args.expected_frames,
            "reversal_epsilon": args.reversal_epsilon,
            "bootstrap_unit": "pair",
            "bootstrap_repeats": args.bootstrap_repeats,
            "seed": args.seed,
            "methods": args.methods,
            "subsets": args.subsets,
            "n_common_pairs": len(common_pairs),
        },
        "scores": summarize(
            rows,
            args.methods,
            args.subsets,
            metric_names,
            args.bootstrap_repeats,
            args.seed,
        ),
    }
    summary_path = os.path.join(args.out_root, "dinov3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    print(f"Saved {per_pair_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)


if __name__ == "__main__":
    main()

"""Compute MorphBench FID for GFM and SLERP on identical completed pairs.

The reference distribution contains the two endpoint images from every pair;
following the MorphBench protocol, two distinct interior frames per sequence
are sampled in each of ten repeats.  Inception features are extracted once and
reused for every repeat, subset, and method.  The same sampled frame indices
are used for GFM and SLERP.  Results are saved to
``<out_root>/fid_summary.json``.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", default="./output/morphbench_sd3")
    parser.add_argument("--methods", nargs="+", default=["gfm", "slerp"])
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["Metamorphosis", "Animation"],
    )
    parser.add_argument("--expected_frames", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    parser.add_argument("--fid_repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def discover(
    out_root: str,
    methods: List[str],
    subsets: List[str],
    expected_frames: int,
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str, str], List[str]]]:
    frames_by_key: Dict[Tuple[str, str, str], List[str]] = {}
    for method in methods:
        for subset in subsets:
            subset_dir = os.path.join(out_root, method, subset)
            if not os.path.isdir(subset_dir):
                continue
            for name in sorted(os.listdir(subset_dir)):
                frame_dir = os.path.join(subset_dir, name, "frames")
                paths = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
                if len(paths) == expected_frames:
                    frames_by_key[(method, subset, name)] = paths

    common_pairs = sorted(
        {
            (subset, name)
            for method, subset, name in frames_by_key
            if all((candidate, subset, name) in frames_by_key for candidate in methods)
        }
    )
    return common_pairs, frames_by_key


def frechet_distance_from_features(
    reference: np.ndarray,
    generated: np.ndarray,
) -> float:
    """Exact empirical FID without a dense feature-by-feature square root.

    If A and B contain centered samples scaled by their covariance
    denominators, the Bures covariance cross term is the nuclear norm of
    A @ B.T.  MorphBench has far fewer samples than Inception dimensions, so
    this sample-space form replaces a 2048x2048 matrix square root with an
    SVD whose largest dimension is the number of images.
    """
    if reference.shape[0] < 2 or generated.shape[0] < 2:
        raise ValueError("FID requires at least two samples per distribution")
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    mean_difference = reference.mean(axis=0) - generated.mean(axis=0)
    centered_reference = (
        reference - reference.mean(axis=0, keepdims=True)
    ) / np.sqrt(reference.shape[0] - 1)
    centered_generated = (
        generated - generated.mean(axis=0, keepdims=True)
    ) / np.sqrt(generated.shape[0] - 1)
    covariance_trace = (
        np.square(centered_reference).sum()
        + np.square(centered_generated).sum()
    )
    cross = centered_reference @ centered_generated.T
    cross_trace = np.linalg.svd(cross, compute_uv=False).sum()
    distance = (
        np.dot(mean_difference, mean_difference)
        + covariance_trace
        - 2.0 * cross_trace
    )
    return float(max(distance, 0.0))


def main():
    args = parse_args()
    from pytorch_fid.fid_score import get_activations
    from pytorch_fid.inception import InceptionV3

    common_pairs, frames_by_key = discover(
        args.out_root,
        args.methods,
        args.subsets,
        args.expected_frames,
    )
    if not common_pairs:
        raise RuntimeError("No complete pairs shared by all requested methods")
    print(f"Common complete pairs: {len(common_pairs)}", flush=True)

    # Endpoint frames are copied verbatim by the pipeline, so one method is a
    # sufficient source for the shared reference distribution.
    endpoint_files = []
    endpoint_pair_indices = []
    for pair_index, (subset, name) in enumerate(common_pairs):
        paths = frames_by_key[(args.methods[0], subset, name)]
        endpoint_files.extend([paths[0], paths[-1]])
        endpoint_pair_indices.extend([pair_index, pair_index])

    interior_files_by_method = {}
    interior_pair_indices_by_method = {}
    for method in args.methods:
        files = []
        pair_indices = []
        for pair_index, (subset, name) in enumerate(common_pairs):
            interior = frames_by_key[(method, subset, name)][1:-1]
            files.extend(interior)
            pair_indices.extend([pair_index] * len(interior))
        interior_files_by_method[method] = files
        interior_pair_indices_by_method[method] = np.asarray(pair_indices)

    block_index = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_index]).to(args.device).eval()
    print(f"Extracting {len(endpoint_files)} endpoint activations...", flush=True)
    endpoint_features = get_activations(
        endpoint_files,
        model,
        args.batch_size,
        args.dims,
        args.device,
        args.num_workers,
    )
    features_by_method = {}
    for method in args.methods:
        files = interior_files_by_method[method]
        print(f"Extracting {len(files)} interior activations for {method}...", flush=True)
        features_by_method[method] = get_activations(
            files,
            model,
            args.batch_size,
            args.dims,
            args.device,
            args.num_workers,
        )

    endpoint_pair_indices = np.asarray(endpoint_pair_indices)
    n_interior = args.expected_frames - 2
    if n_interior < 2:
        raise ValueError("--expected_frames must leave at least two interior frames")
    rng = np.random.default_rng(args.seed)
    # Argsort gives two different positions without a Python loop.  Sampling
    # positions are shared across methods for a paired comparison.
    sampled_positions = np.argsort(
        rng.random((args.fid_repeats, len(common_pairs), n_interior)),
        axis=2,
    )[:, :, :2]
    results = {
        "protocol": {
            "reference": "both endpoint frames from every complete pair",
            "generated": (
                "two distinct interior frames sampled per sequence in each repeat"
            ),
            "fid_repeats": args.fid_repeats,
            "random_seed": args.seed,
            "same_sampled_positions_across_methods": True,
            "expected_frames_per_sequence": args.expected_frames,
            "inception_dimensions": args.dims,
            "common_pairs_only": True,
        },
        "scores": {},
    }
    scopes = [*args.subsets, "ALL"]
    for scope in scopes:
        selected_pairs = np.asarray(
            [
                index
                for index, (subset, _name) in enumerate(common_pairs)
                if scope == "ALL" or subset == scope
            ],
            dtype=np.int64,
        )
        if not selected_pairs.size:
            continue
        endpoint_mask = np.isin(endpoint_pair_indices, selected_pairs)
        reference_features = endpoint_features[endpoint_mask]
        results["scores"][scope] = {}
        for method in args.methods:
            method_features = features_by_method[method].reshape(
                len(common_pairs), n_interior, args.dims
            )
            repeat_scores = []
            for repeat in range(args.fid_repeats):
                positions = sampled_positions[repeat, selected_pairs]
                generated = method_features[
                    selected_pairs[:, None],
                    positions,
                ].reshape(-1, args.dims)
                repeat_scores.append(
                    frechet_distance_from_features(
                        reference_features,
                        generated,
                    )
                )
            scores = np.asarray(repeat_scores, dtype=np.float64)
            results["scores"][scope][method] = {
                "fid_mean": float(scores.mean()),
                "fid_std": float(scores.std(ddof=1)) if scores.size > 1 else 0.0,
                "fid_repeats": repeat_scores,
                "n_pairs": int(selected_pairs.size),
                "n_reference": int(endpoint_mask.sum()),
                "n_generated_per_repeat": int(2 * selected_pairs.size),
            }
            print(
                f"{scope:13s} {method:5s}: "
                f"FID={scores.mean():.4f} +/- {scores.std(ddof=1):.4f} "
                f"(ref={endpoint_mask.sum()}, generated/repeat={2 * selected_pairs.size})",
                flush=True,
            )

    output_path = os.path.join(args.out_root, "fid_summary.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()

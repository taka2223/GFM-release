"""Run the paired ShapeNet rebuttal experiments (E1 and E2).

The runner is intentionally idempotent:

* endpoint and noisy-endpoint caches are immutable once created;
* each optimization batch is written atomically;
* completed batches are skipped on resume;
* E2 summaries are derived directly from the lambda=1 E1 runs.

This stage produces optimized and denoised latent paths plus solver diagnostics.
Mesh decoding and distributional metric evaluation are separate, downstream
stages because they are substantially more expensive than path optimization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gfm.helper.help import load_model_from_path
from gfm.models import models_class_cond
from gfm.path.geodesic_interpolation import ELInterpolator


DEFAULT_CATEGORIES = [5, 47]
DEFAULT_SIGMAS = [0.1, 0.3]
DEFAULT_LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
DEFAULT_SEEDS = [99995, 199995, 299995]
FORMAT_VERSION = 1


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def float_slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )


def atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def atomic_json_dump(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


def append_jsonl(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def cache_metadata(args: argparse.Namespace, category: int, seed: int) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "category": category,
        "seed": seed,
        "num_pairs": args.num_pairs,
        "dm_name": args.dm_name,
        "dm_path": str(Path(args.dm_path).resolve()),
    }


def validate_metadata(
    actual: dict[str, Any], expected: dict[str, Any], source: Path
) -> None:
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Cache metadata mismatch in {source}: {mismatches}")


def load_diffusion_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    log(f"Loading diffusion model from {args.dm_path}")
    return load_model_from_path(
        model_name=args.dm_name,
        ckpt_path=args.dm_path,
        registry=models_class_cond.__dict__,
        device=device,
    )


def sample_latents(
    dm: torch.nn.Module,
    category: int,
    sample_seeds: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for start in range(0, sample_seeds.numel(), batch_size):
        seeds = sample_seeds[start : start + batch_size].to(device)
        labels = torch.full(
            (seeds.numel(),), category, dtype=torch.long, device=device
        )
        with torch.no_grad():
            chunk = dm.sample(cond=labels, batch_seeds=seeds).float().cpu()
        chunks.append(chunk)
        log(
            f"Sampled category={category}: "
            f"{min(start + seeds.numel(), sample_seeds.numel())}/{sample_seeds.numel()}"
        )
    return torch.cat(chunks, dim=0)


def endpoint_path(args: argparse.Namespace, category: int, seed: int) -> Path:
    return (
        args.output_root
        / "cache"
        / "endpoints"
        / f"category{category:02d}_seed{seed}_pairs{args.num_pairs}.pt"
    )


def noise_path(
    args: argparse.Namespace, category: int, seed: int, sigma: float
) -> Path:
    return (
        args.output_root
        / "cache"
        / "noisy_endpoints"
        / (
            f"category{category:02d}_seed{seed}_pairs{args.num_pairs}"
            f"_sigma{float_slug(sigma)}.pt"
        )
    )


def ensure_endpoint_cache(
    args: argparse.Namespace,
    dm: torch.nn.Module,
    category: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    destination = endpoint_path(args, category, seed)
    expected = cache_metadata(args, category, seed)
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=False)
        validate_metadata(payload["metadata"], expected, destination)
        return payload

    sample_seeds = torch.arange(
        seed, seed + 2 * args.num_pairs, dtype=torch.long
    )
    clean = sample_latents(
        dm, category, sample_seeds, args.sample_batch_size, device
    )
    payload = {
        "metadata": expected,
        "sample_seeds": sample_seeds,
        "start": clean[0::2].contiguous(),
        "end": clean[1::2].contiguous(),
    }
    atomic_torch_save(payload, destination)
    log(f"Saved endpoint cache: {destination}")
    return payload


def ensure_noise_cache(
    args: argparse.Namespace,
    endpoints: dict[str, Any],
    category: int,
    seed: int,
    sigma: float,
) -> dict[str, Any]:
    destination = noise_path(args, category, seed, sigma)
    expected = {
        **cache_metadata(args, category, seed),
        "sigma": sigma,
        "noise_seed": stable_seed("endpoint-noise", category, seed, sigma),
    }
    if destination.exists():
        payload = torch.load(destination, map_location="cpu", weights_only=False)
        validate_metadata(payload["metadata"], expected, destination)
        return payload

    generator = torch.Generator(device="cpu")
    generator.manual_seed(expected["noise_seed"])
    start = endpoints["start"].float()
    end = endpoints["end"].float()
    start_noise = torch.randn(
        start.shape, generator=generator, dtype=start.dtype
    )
    end_noise = torch.randn(end.shape, generator=generator, dtype=end.dtype)
    payload = {
        "metadata": expected,
        "start": (start + sigma * start_noise).contiguous(),
        "end": (end + sigma * end_noise).contiguous(),
    }
    atomic_torch_save(payload, destination)
    log(f"Saved noisy endpoint cache: {destination}")
    return payload


def prepare_caches(
    args: argparse.Namespace, dm: torch.nn.Module, device: torch.device
) -> None:
    for category in args.categories:
        for seed in args.seeds:
            endpoints = ensure_endpoint_cache(args, dm, category, seed, device)
            for sigma in args.sigmas:
                ensure_noise_cache(args, endpoints, category, seed, sigma)


def condition_dir(
    args: argparse.Namespace, category: int, sigma: float, lam: float, seed: int
) -> Path:
    return (
        args.output_root
        / "runs"
        / f"category{category:02d}"
        / f"sigma{float_slug(sigma)}"
        / f"lambda{float_slug(lam)}"
        / f"seed{seed}"
    )


def batch_path(destination: Path, start: int, end: int) -> Path:
    return destination / "batches" / f"pairs{start:03d}-{end - 1:03d}.pt"


def run_metadata(
    args: argparse.Namespace,
    category: int,
    sigma: float,
    lam: float,
    seed: int,
) -> dict[str, Any]:
    return {
        **cache_metadata(args, category, seed),
        "sigma": sigma,
        "lambda": lam,
        "num_waypoints": args.num_waypoints,
        "max_iters": args.max_iters,
        "lr": args.lr,
        "num_denoise_steps": args.num_denoise_steps,
        "initialization": "SLERP",
    }


def optimize_batch(
    args: argparse.Namespace,
    interpolator: ELInterpolator,
    noisy: dict[str, Any],
    category: int,
    sigma: float,
    lam: float,
    seed: int,
    start: int,
    end: int,
    device: torch.device,
) -> dict[str, Any]:
    started = time.time()
    batch_labels = torch.full(
        (end - start,), category, dtype=torch.long, device=device
    )
    noisy_start = noisy["start"][start:end].to(device)
    noisy_end = noisy["end"][start:end].to(device)

    optimized, info = interpolator.optimize_path(
        start_latent=noisy_start,
        end_latent=noisy_end,
        num_steps=args.num_waypoints,
        lr=args.lr,
        sigma=sigma,
        class_label=batch_labels,
        max_iters=args.max_iters,
        lam=lam,
        verbose=args.verbose_optimizer,
        record_per_path_residuals=True,
        fail_on_nonfinite=True,
    )
    optimized_finite = torch.isfinite(optimized).flatten(start_dim=1).all(dim=1)
    if not optimized_finite.all():
        bad = (~optimized_finite).nonzero(as_tuple=False).flatten().tolist()
        raise FloatingPointError(f"Non-finite optimized paths at batch indices {bad}")

    denoised = interpolator.denoise_path(
        optimized,
        sigma_start=sigma,
        class_label=batch_labels,
        num_denoise_steps=args.num_denoise_steps,
    )
    denoised_finite = torch.isfinite(denoised).flatten(start_dim=1).all(dim=1)
    residuals = info["per_path_residuals"]
    residual_finite = torch.isfinite(residuals).all(dim=1)

    return {
        "metadata": {
            **run_metadata(args, category, sigma, lam, seed),
            "pair_start": start,
            "pair_end": end,
            "elapsed_seconds": time.time() - started,
        },
        "pair_indices": torch.arange(start, end, dtype=torch.long),
        "optimized_paths": optimized.detach().cpu(),
        "denoised_paths": denoised.detach().cpu(),
        "residual_curves": residuals,
        "optimized_finite": optimized_finite.cpu(),
        "denoised_finite": denoised_finite.cpu(),
        "residual_finite": residual_finite.cpu(),
        "status": ["success"] * (end - start),
    }


def failure_records(
    metadata: dict[str, Any],
    start: int,
    end: int,
    error: BaseException,
) -> list[dict[str, Any]]:
    status = "numerical_error" if isinstance(error, FloatingPointError) else "runtime_error"
    return [
        {
            **metadata,
            "pair_index": pair_index,
            "status": status,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        for pair_index in range(start, end)
    ]


def run_condition(
    args: argparse.Namespace,
    interpolator: ELInterpolator,
    category: int,
    sigma: float,
    lam: float,
    seed: int,
    device: torch.device,
) -> None:
    destination = condition_dir(args, category, sigma, lam, seed)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = run_metadata(args, category, sigma, lam, seed)
    atomic_json_dump(metadata, destination / "metadata.json")
    noisy = torch.load(
        noise_path(args, category, seed, sigma),
        map_location="cpu",
        weights_only=False,
    )

    covered_pairs: set[int] = set()
    for existing in iter_batch_payloads(destination):
        validate_metadata(existing["metadata"], metadata, destination)
        covered_pairs.update(int(value) for value in existing["pair_indices"])

    complete = len(covered_pairs)
    failed = 0
    for start in range(0, args.num_pairs, args.batch_size):
        end = min(start + args.batch_size, args.num_pairs)
        missing_pairs = [
            pair_index
            for pair_index in range(start, end)
            if pair_index not in covered_pairs
        ]
        if not missing_pairs:
            continue

        output = batch_path(destination, start, end)
        log(
            f"E1 category={category} sigma={sigma:g} lambda={lam:g} seed={seed} "
            f"pairs={missing_pairs[0]}-{missing_pairs[-1]}"
        )
        try:
            if len(missing_pairs) == end - start:
                payload = optimize_batch(
                    args,
                    interpolator,
                    noisy,
                    category,
                    sigma,
                    lam,
                    seed,
                    start,
                    end,
                    device,
                )
                atomic_torch_save(payload, output)
                covered_pairs.update(range(start, end))
                complete += end - start
            else:
                raise RuntimeError("partial_batch_resume")
        except Exception as error:
            # If a multi-path batch fails, retry paths independently so failures
            # remain path-level rather than being attributed to the whole batch.
            if str(error) != "partial_batch_resume":
                log(
                    f"Batch failed ({type(error).__name__}: {error}); "
                    "retrying singly"
                )
            if len(missing_pairs) > 1 or str(error) == "partial_batch_resume":
                for pair_index in missing_pairs:
                    single_output = batch_path(
                        destination, pair_index, pair_index + 1
                    )
                    try:
                        payload = optimize_batch(
                            args,
                            interpolator,
                            noisy,
                            category,
                            sigma,
                            lam,
                            seed,
                            pair_index,
                            pair_index + 1,
                            device,
                        )
                        atomic_torch_save(payload, single_output)
                        covered_pairs.add(pair_index)
                        complete += 1
                    except Exception as single_error:
                        failed += 1
                        for record in failure_records(
                            metadata,
                            pair_index,
                            pair_index + 1,
                            single_error,
                        ):
                            append_jsonl(record, destination / "failures.jsonl")
            else:
                failed += 1
                for record in failure_records(
                    metadata,
                    missing_pairs[0],
                    missing_pairs[0] + 1,
                    error,
                ):
                    append_jsonl(record, destination / "failures.jsonl")
            torch.cuda.empty_cache()

        atomic_json_dump(
            {
                **metadata,
                "completed_paths": complete,
                "failed_paths": failed,
                "expected_paths": args.num_pairs,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            destination / "status.json",
        )


def run_sweep(
    args: argparse.Namespace, dm: torch.nn.Module, device: torch.device
) -> None:
    dummy_autoencoder = torch.nn.Identity().to(device).eval()
    interpolator = ELInterpolator(dm, dummy_autoencoder, device=device)
    # Run lambda=1 first so E2 becomes available as early as possible.
    lambda_order = sorted(args.lambdas, key=lambda value: (value != 1.0, value))
    for category in args.categories:
        for sigma in args.sigmas:
            for lam in lambda_order:
                for seed in args.seeds:
                    run_condition(
                        args, interpolator, category, sigma, lam, seed, device
                    )


def iter_batch_payloads(destination: Path) -> Iterable[dict[str, Any]]:
    for path in sorted((destination / "batches").glob("pairs*.pt")):
        yield torch.load(path, map_location="cpu", weights_only=False)


def scalar_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "p95": None,
        }
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "q25": float(np.percentile(finite, 25)),
        "q75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
    }


def summarize_reliability(args: argparse.Namespace) -> None:
    summary_dir = args.output_root / "summaries" / "e2_reliability"
    summary_dir.mkdir(parents=True, exist_ok=True)
    table_rows: list[dict[str, Any]] = []
    convergence_panels: dict[tuple[int, float], dict[str, np.ndarray]] = {}

    for category in args.categories:
        for sigma in args.sigmas:
            curves: list[np.ndarray] = []
            optimized_finite: list[np.ndarray] = []
            denoised_finite: list[np.ndarray] = []
            explicit_failures: list[dict[str, Any]] = []
            for seed in args.seeds:
                destination = condition_dir(args, category, sigma, 1.0, seed)
                for payload in iter_batch_payloads(destination):
                    curves.append(payload["residual_curves"].numpy())
                    optimized_finite.append(payload["optimized_finite"].numpy())
                    denoised_finite.append(payload["denoised_finite"].numpy())
                failure_path = destination / "failures.jsonl"
                if failure_path.exists():
                    explicit_failures.extend(
                        json.loads(line)
                        for line in failure_path.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    )

            if curves:
                curve = np.concatenate(curves, axis=0)
                initial = curve[:, 0]
                final = curve[:, -1]
                relative = (initial - final) / np.maximum(initial, 1e-12)
                convergence = {
                    "median": np.nanmedian(curve, axis=0),
                    "p10": np.nanpercentile(curve, 10, axis=0),
                    "p90": np.nanpercentile(curve, 90, axis=0),
                }
            else:
                curve = np.empty((0, args.max_iters + 1), dtype=np.float32)
                initial = final = relative = np.empty((0,), dtype=np.float32)
                convergence = {
                    key: np.full((args.max_iters + 1,), np.nan)
                    for key in ("median", "p10", "p90")
                }
            convergence_panels[(category, sigma)] = convergence

            finite_opt = (
                np.concatenate(optimized_finite)
                if optimized_finite
                else np.empty((0,), dtype=bool)
            )
            finite_denoised = (
                np.concatenate(denoised_finite)
                if denoised_finite
                else np.empty((0,), dtype=bool)
            )
            row = {
                "category": category,
                "sigma": sigma,
                "expected_runs": args.num_pairs * len(args.seeds),
                "completed_runs": int(curve.shape[0]),
                "explicit_failure_count": len(explicit_failures),
                "nonfinite_optimized_count": int((~finite_opt).sum()),
                "nonfinite_denoised_count": int((~finite_denoised).sum()),
                "initial_residual": scalar_summary(initial),
                "final_residual": scalar_summary(final),
                "relative_decrease": scalar_summary(relative),
            }
            table_rows.append(row)
            atomic_json_dump(
                {
                    **row,
                    "explicit_failures": explicit_failures,
                },
                summary_dir
                / f"category{category:02d}_sigma{float_slug(sigma)}.json",
            )
            np.savez_compressed(
                summary_dir
                / f"category{category:02d}_sigma{float_slug(sigma)}_convergence.npz",
                residual_curves=curve,
                median=convergence["median"],
                p10=convergence["p10"],
                p90=convergence["p90"],
            )

    atomic_json_dump(table_rows, summary_dir / "summary.json")
    csv_path = summary_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "category",
            "sigma",
            "expected_runs",
            "completed_runs",
            "explicit_failure_count",
            "nonfinite_optimized_count",
            "nonfinite_denoised_count",
            "initial_median",
            "initial_q25",
            "initial_q75",
            "initial_p95",
            "final_median",
            "final_q25",
            "final_q75",
            "final_p95",
            "relative_median",
            "relative_q25",
            "relative_q75",
            "relative_p95",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in table_rows:
            writer.writerow(
                {
                    "category": row["category"],
                    "sigma": row["sigma"],
                    "expected_runs": row["expected_runs"],
                    "completed_runs": row["completed_runs"],
                    "explicit_failure_count": row["explicit_failure_count"],
                    "nonfinite_optimized_count": row[
                        "nonfinite_optimized_count"
                    ],
                    "nonfinite_denoised_count": row[
                        "nonfinite_denoised_count"
                    ],
                    **{
                        f"{prefix}_{name}": row[key][name]
                        for prefix, key in (
                            ("initial", "initial_residual"),
                            ("final", "final_residual"),
                            ("relative", "relative_decrease"),
                        )
                        for name in ("median", "q25", "q75", "p95")
                    },
                }
            )

    category_names = {5: "Bed", 47: "Sofa"}
    markdown = [
        "# E2 Optimization Reliability",
        "",
        "| Category | σ | Runs | Failures | R0 median [IQR] | "
        "Rf median [IQR] | Relative decrease median [IQR] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        initial = row["initial_residual"]
        final = row["final_residual"]
        relative = row["relative_decrease"]
        failures = (
            row["explicit_failure_count"]
            + row["nonfinite_optimized_count"]
            + row["nonfinite_denoised_count"]
        )
        markdown.append(
            f"| {category_names.get(row['category'], row['category'])} "
            f"| {row['sigma']:g} "
            f"| {row['completed_runs']}/{row['expected_runs']} "
            f"| {failures} "
            f"| {initial['median']:.4g} [{initial['q25']:.4g}, "
            f"{initial['q75']:.4g}] "
            f"| {final['median']:.4g} [{final['q25']:.4g}, "
            f"{final['q75']:.4g}] "
            f"| {relative['median']:.4g} [{relative['q25']:.4g}, "
            f"{relative['q75']:.4g}] |"
        )
    (summary_dir / "summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    if convergence_panels:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(
            len(args.categories),
            len(args.sigmas),
            figsize=(4.2 * len(args.sigmas), 3.3 * len(args.categories)),
            squeeze=False,
            sharex=True,
        )
        for row_index, category in enumerate(args.categories):
            for column_index, sigma in enumerate(args.sigmas):
                axis = axes[row_index, column_index]
                panel = convergence_panels[(category, sigma)]
                iterations = np.arange(panel["median"].shape[0])
                axis.fill_between(
                    iterations,
                    panel["p10"],
                    panel["p90"],
                    color="#4C78A8",
                    alpha=0.22,
                    linewidth=0,
                    label="10–90%",
                )
                axis.plot(
                    iterations,
                    panel["median"],
                    color="#1F5A94",
                    linewidth=1.7,
                    label="Median",
                )
                axis.set_yscale("log")
                axis.set_title(
                    f"{category_names.get(category, category)}, σ={sigma:g}"
                )
                axis.grid(alpha=0.2)
                if row_index == len(args.categories) - 1:
                    axis.set_xlabel("Optimization iteration")
                if column_index == 0:
                    axis.set_ylabel("Euler–Lagrange residual")
                if row_index == 0 and column_index == 0:
                    axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(summary_dir / "convergence.png", dpi=220)
        figure.savefig(summary_dir / "convergence.pdf")
        plt.close(figure)
    log(f"Wrote E2 reliability summaries to {summary_dir}")


def write_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "format_version": FORMAT_VERSION,
        "categories": args.categories,
        "sigmas": args.sigmas,
        "lambdas": args.lambdas,
        "seeds": args.seeds,
        "num_pairs": args.num_pairs,
        "batch_size": args.batch_size,
        "sample_batch_size": args.sample_batch_size,
        "num_waypoints": args.num_waypoints,
        "max_iters": args.max_iters,
        "num_denoise_steps": args.num_denoise_steps,
        "lr": args.lr,
        "dm_name": args.dm_name,
        "dm_path": str(Path(args.dm_path).resolve()),
        "paired_design": (
            "Within category/sigma/seed, every lambda uses identical clean "
            "endpoints, endpoint noise, and SLERP initialization."
        ),
    }
    is_full_design = (
        args.categories == DEFAULT_CATEGORIES
        and args.sigmas == DEFAULT_SIGMAS
        and args.lambdas == DEFAULT_LAMBDAS
        and args.seeds == DEFAULT_SEEDS
        and args.num_pairs == 50
        and args.num_waypoints == 10
        and args.max_iters == 400
        and args.num_denoise_steps == 5
    )
    if is_full_design:
        destination = args.output_root / "manifest.json"
    else:
        shard = (
            "categories-" + "-".join(str(value) for value in args.categories)
            + "_sigmas-" + "-".join(float_slug(value) for value in args.sigmas)
            + "_lambdas-" + "-".join(float_slug(value) for value in args.lambdas)
            + "_seeds-" + "-".join(str(value) for value in args.seeds)
        )
        destination = args.output_root / "manifests" / f"{shard}.json"
    atomic_json_dump(manifest, destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["all", "prepare", "sweep", "reliability"],
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--categories", type=int, nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    parser.add_argument("--lambdas", type=float, nargs="+", default=DEFAULT_LAMBDAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--num-pairs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--sample-batch-size", type=int, default=16)
    parser.add_argument("--num-waypoints", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=400)
    parser.add_argument("--num-denoise-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--dm-name", default="kl_d512_m512_l8_d24_edm"
    )
    parser.add_argument(
        "--dm-path",
        type=Path,
        default=ROOT
        / "class_cond_dm"
        / "kl_d512_m512_l8_d24_edm"
        / "checkpoint-499.pth",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "rebuttal",
    )
    parser.add_argument("--verbose-optimizer", action="store_true")
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.categories = [5]
    args.sigmas = [0.1]
    args.lambdas = [1.0]
    args.seeds = [99995]
    args.num_pairs = 1
    args.batch_size = 1
    args.sample_batch_size = 2
    args.max_iters = 2
    args.num_denoise_steps = 2
    args.output_root = args.output_root / "smoke"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    apply_smoke_overrides(args)
    args.output_root = args.output_root.resolve()
    args.dm_path = args.dm_path.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_manifest(args)

    if not torch.cuda.is_available() and args.phase != "reliability":
        raise RuntimeError("CUDA is required for endpoint generation and path optimization")
    device = torch.device("cuda:0")

    dm: torch.nn.Module | None = None
    if args.phase in {"all", "prepare", "sweep"}:
        dm = load_diffusion_model(args, device)
    if args.phase in {"all", "prepare"}:
        assert dm is not None
        prepare_caches(args, dm, device)
    if args.phase in {"all", "sweep"}:
        assert dm is not None
        # Sweep can be invoked separately after caches were prepared.
        if args.phase == "sweep":
            for category in args.categories:
                for seed in args.seeds:
                    endpoint = endpoint_path(args, category, seed)
                    if not endpoint.exists():
                        endpoints = ensure_endpoint_cache(
                            args, dm, category, seed, device
                        )
                    else:
                        endpoints = torch.load(
                            endpoint, map_location="cpu", weights_only=False
                        )
                    for sigma in args.sigmas:
                        ensure_noise_cache(
                            args, endpoints, category, seed, sigma
                        )
        run_sweep(args, dm, device)
    if args.phase in {"all", "reliability"}:
        summarize_reliability(args)

    log("Requested rebuttal stages completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted; completed caches and batches remain resumable")
        raise
    except Exception:
        traceback.print_exc()
        raise

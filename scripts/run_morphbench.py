"""Run paired SD3/SD2.1 GFM and SLERP interpolation on MorphBench.

The output layout is

    <out_root>/<method>/<subset>/<pair>/frames/frame_*.png
    <out_root>/captions/<subset>/<pair>.json

Captions are generated once and shared by every method.  Work can be split
across GPUs with ``--num_shards``/``--shard_id`` and resumed safely.
"""

import argparse
import json
import os
import sys
import time
import traceback
from types import ModuleType
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

# Make `gfm` importable when running from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Pair = Tuple[str, str, str, str]
DEFAULT_MODEL_IDS = {
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd21": "sd2-community/stable-diffusion-2-1-base",
}
DEFAULT_MODEL_REVISIONS = {
    "sd21": "4e63672c03103b6c636b8fb4119ba982469b2955",
}


def load_backend_api(backbone: str) -> Tuple[ModuleType, Any]:
    """Import only the selected diffusion implementation."""
    if backbone == "sd3":
        from gfm.path import sd3_wrapper as api

        return api, api.SD3GFMPipeline
    if backbone == "sd21":
        from gfm.path import sd_wrapper as api

        return api, api.SD21GFMPipeline
    raise ValueError(f"Unsupported backbone: {backbone}")


def discover_pairs(subset_dir: str) -> List[Tuple[str, str, str]]:
    """Find complete <name>_0.png / <name>_1.png pairs."""
    files = set(os.listdir(subset_dir))
    bases = sorted(
        filename[: -len("_0.png")]
        for filename in files
        if filename.endswith("_0.png")
        and f"{filename[:-len('_0.png')]}_1.png" in files
    )
    return [
        (
            base,
            os.path.join(subset_dir, f"{base}_0.png"),
            os.path.join(subset_dir, f"{base}_1.png"),
        )
        for base in bases
    ]


def write_json(path: str, payload: Dict) -> None:
    """Write JSON through a same-directory temporary file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        choices=["sd3", "sd21"],
        default="sd3",
        help="Diffusion backbone. The default preserves the existing SD3 protocol.",
    )
    parser.add_argument("--data_root", default="./data/MorphBench")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["Metamorphosis", "Animation"],
    )
    parser.add_argument("--out_root", default="./output/morphbench_sd3")
    parser.add_argument(
        "--caption_root",
        default=None,
        help=(
            "Optional shared caption directory. Defaults to "
            "<out_root>/captions."
        ),
    )
    parser.add_argument(
        "--model_id",
        default=None,
        help="Override the official default checkpoint for the selected backbone.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face model revision (recorded in metadata).",
    )
    parser.add_argument(
        "--caption_model_id",
        default="./BLIP_large",
        help="BLIP model used to create the shared endpoint captions.",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise_level", type=float, default=0.6)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument(
        "--cond_seed",
        type=int,
        default=20260727,
        help="Shared Gaussian-noise seed for the cond_interp baseline.",
    )
    parser.add_argument("--lam", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_iters", type=int, default=400)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop after this many iterations without sufficient improvement; 0 disables.",
    )
    parser.add_argument(
        "--early_stop_rel_delta",
        type=float,
        default=1e-4,
        help="Minimum relative loss improvement that resets early-stop patience.",
    )
    parser.add_argument(
        "--early_stop_warmup",
        type=int,
        default=0,
        help="Minimum number of iterations before early stopping is allowed.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=17,
        help="Number of output frames, including both endpoints.",
    )
    parser.add_argument(
        "--opt_num_steps",
        type=int,
        default=0,
        help=(
            "Number of path control points optimized before spherical "
            "resampling; 0 uses --num_steps."
        ),
    )
    parser.add_argument(
        "--c2f_steps",
        type=int,
        nargs="+",
        default=None,
        help="Increasing control-point counts for gfm_c2f.",
    )
    parser.add_argument(
        "--c2f_iters",
        type=int,
        nargs="+",
        default=None,
        help="Optimization iterations for each gfm_c2f stage.",
    )
    parser.add_argument(
        "--c2f_lrs",
        type=float,
        nargs="+",
        default=None,
        help="Learning rates for each gfm_c2f stage.",
    )
    parser.add_argument(
        "--c2f_freeze_anchors",
        action="store_true",
        help=(
            "At each refined C2F stage, freeze the previous grid points and "
            "optimize only newly inserted SLERP-initialized points."
        ),
    )
    parser.add_argument(
        "--n_inference_steps",
        type=int,
        default=50,
        help="Forward/backward integration steps for the selected backbone.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--keep_t5", action="store_true")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gfm", "slerp"],
        choices=["gfm", "gfm_c2f", "gfm_sphere", "slerp", "cond_interp"],
        help=(
            "gfm uses the existing ambient EL2 solver; gfm_c2f progressively "
            "upsamples and refines its control path; gfm_sphere uses tangent "
            "projection plus per-update sphere retraction; cond_interp generates "
            "from linearly interpolated condition embeddings and fixed noise."
        ),
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only the first N pairs globally (before sharding); 0 means all.",
    )
    parser.add_argument(
        "--pair_names",
        nargs="+",
        default=None,
        help="Run only these exact MorphBench pair names.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def collect_pairs(args) -> List[Pair]:
    all_pairs: List[Pair] = []
    for subset in args.subsets:
        subset_dir = os.path.join(args.data_root, subset)
        if not os.path.isdir(subset_dir):
            print(f"[WARN] {subset_dir} not found, skipping", flush=True)
            continue
        pairs = discover_pairs(subset_dir)
        print(f"{subset}: {len(pairs)} pairs", flush=True)
        all_pairs.extend(
            (subset, name, path_a, path_b)
            for name, path_a, path_b in pairs
            if args.pair_names is None or name in set(args.pair_names)
        )

    if args.limit > 0:
        all_pairs = all_pairs[: args.limit]
    return all_pairs


def load_or_create_captions(
    pipe,
    captioner,
    args,
    subset: str,
    name: str,
    image_a: Image.Image,
    image_b: Image.Image,
) -> Dict[str, str]:
    caption_root = args.caption_root or os.path.join(args.out_root, "captions")
    caption_path = os.path.join(caption_root, subset, f"{name}.json")
    if os.path.isfile(caption_path) and not args.overwrite:
        with open(caption_path, encoding="utf-8") as handle:
            captions = json.load(handle)
        if captions.get("promptA") and captions.get("promptB"):
            pipe.unload_captioner()
            return captions

    pipe.captioner = captioner
    captioner.model.to(args.device)
    captions = {
        "promptA": pipe.auto_caption(image_a),
        "promptB": pipe.auto_caption(image_b),
        "model_id": args.caption_model_id,
        "prefix": captioner.prefix,
    }
    pipe.unload_captioner()
    write_json(caption_path, captions)
    print(f'  Caption A: "{captions["promptA"]}"', flush=True)
    print(f'  Caption B: "{captions["promptB"]}"', flush=True)
    return captions


def run_method(
    api: ModuleType,
    pipe,
    args,
    method: str,
    subset: str,
    name: str,
    image_a: Image.Image,
    image_b: Image.Image,
    captions: Dict[str, str],
) -> None:
    out_dir = os.path.join(args.out_root, method, subset, name)
    strip_path = os.path.join(out_dir, "strip.png")
    if os.path.isfile(strip_path) and not args.overwrite:
        print(f"  SKIP {method} (done)", flush=True)
        return

    os.makedirs(out_dir, exist_ok=True)
    started = time.monotonic()
    print(f"  RUN {method}", flush=True)
    try:
        results = api.run_gfm_interpolation(
            pipe,
            image_a,
            image_b,
            promptA=captions["promptA"],
            promptB=captions["promptB"],
            noise_level=args.noise_level,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            lam=args.lam,
            lr=args.lr,
            max_iters=args.max_iters,
            mode="nfsd",
            method=method,
            cond_seed=args.cond_seed,
            opt_num_steps=args.opt_num_steps or None,
            early_stop_patience=args.early_stop_patience,
            early_stop_rel_delta=args.early_stop_rel_delta,
            early_stop_warmup=args.early_stop_warmup,
            c2f_steps=args.c2f_steps,
            c2f_iters=args.c2f_iters,
            c2f_lrs=args.c2f_lrs,
            c2f_freeze_anchors=args.c2f_freeze_anchors,
        )
        api.save_frames(results["images"], os.path.join(out_dir, "frames"))
        api.save_interpolation_strip(results["images"], strip_path)
        api.save_optimization_curves(
            losses=results["losses"],
            acc_norms=results["acc_norms"],
            f_norms=results["f_norms"],
            lam=args.lam,
            output_dir=out_dir,
        )
        elapsed = time.monotonic() - started
        write_json(
            os.path.join(out_dir, "meta.json"),
            {
                "status": "complete",
                "method": method,
                "runtime_seconds": elapsed,
                "promptA": results["promptA"],
                "promptB": results["promptB"],
                "sphere_radius": results["sphere_radius"],
                "sigma_eff": results["sigma_eff"],
                "alpha_t": results.get("alpha_t"),
                "timestep": results.get("timestep"),
                "final_loss": (
                    results.get("best_loss")
                    if results.get("best_loss") is not None
                    else (results["losses"] or [None])[-1]
                ),
                "best_loss": results.get("best_loss"),
                "best_iteration": results.get("best_iteration"),
                "iterations_run": results.get("iterations_run"),
                "early_stopped": results.get("early_stopped", False),
                "c2f_stages": results.get("c2f_stages"),
                "c2f_freeze_anchors": results.get(
                    "c2f_freeze_anchors",
                    False,
                ),
                "model": results.get(
                    "model",
                    {
                        "backbone": args.backbone,
                        "model_id": args.model_id,
                        "requested_revision": args.revision,
                        "resolved_revision": getattr(
                            getattr(pipe, "pipe", None),
                            "_commit_hash",
                            args.revision,
                        ),
                        "resolution": args.resolution,
                        "n_inference_steps": args.n_inference_steps,
                    },
                ),
                "conditioning_protocol": results.get("conditioning_protocol"),
                "args": {
                    key: value
                    for key, value in vars(args).items()
                    if key not in {"out_root", "overwrite"}
                },
            },
        )
        error_path = os.path.join(out_dir, "error.json")
        if os.path.exists(error_path):
            os.unlink(error_path)
        print(f"  DONE {method} in {elapsed / 60:.1f} min", flush=True)
    except Exception as exc:
        elapsed = time.monotonic() - started
        write_json(
            os.path.join(out_dir, "error.json"),
            {
                "status": "failed",
                "method": method,
                "runtime_seconds": elapsed,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "model": getattr(pipe, "model_metadata", None),
                "args": {
                    key: value
                    for key, value in vars(args).items()
                    if key not in {"out_root", "overwrite"}
                },
            },
        )
        print(f"  FAILED {method}: {exc!r}", flush=True)
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    args.model_id = args.model_id or DEFAULT_MODEL_IDS[args.backbone]
    if args.backbone in DEFAULT_MODEL_REVISIONS and args.revision is None:
        args.revision = DEFAULT_MODEL_REVISIONS[args.backbone]
    if args.backbone == "sd21" and args.resolution != 512:
        raise ValueError("--backbone sd21 requires --resolution 512")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    args.methods = list(dict.fromkeys(args.methods))
    if args.backbone == "sd21":
        unsupported = set(args.methods) - {"gfm", "gfm_c2f", "slerp"}
        if unsupported:
            raise ValueError(
                "SD2.1 supports gfm, gfm_c2f, and slerp only; unsupported: "
                + ", ".join(sorted(unsupported))
            )
    os.makedirs(args.out_root, exist_ok=True)

    all_pairs = collect_pairs(args)
    shard_pairs = [
        pair for index, pair in enumerate(all_pairs) if index % args.num_shards == args.shard_id
    ]
    print(
        f"Global total: {len(all_pairs)}; shard {args.shard_id}/{args.num_shards}: "
        f"{len(shard_pairs)} pairs",
        flush=True,
    )
    if not shard_pairs:
        return

    api, pipeline_class = load_backend_api(args.backbone)
    caption_root = args.caption_root or os.path.join(args.out_root, "captions")
    captions_complete = all(
        os.path.isfile(os.path.join(caption_root, subset, f"{name}.json"))
        for subset, name, _path_a, _path_b in shard_pairs
    )
    print(
        f"Loading {args.backbone} from {args.model_id}"
        f" at revision {args.revision or 'default'}"
        f" (BLIP {'not needed' if captions_complete else 'enabled'})...",
        flush=True,
    )
    load_kwargs = dict(
        model_id=args.model_id,
        device=args.device,
        cache_dir=args.cache_dir,
        load_blip=not captions_complete,
        caption_model_id=args.caption_model_id,
        resolution=args.resolution,
        drop_t5=not args.keep_t5,
        n_inference_steps=args.n_inference_steps,
    )
    if args.backbone == "sd21":
        load_kwargs["revision"] = args.revision
    pipe = pipeline_class.load(**load_kwargs)
    captioner = pipe.captioner

    for index, (subset, name, path_a, path_b) in enumerate(shard_pairs, start=1):
        print(
            f"\n[{index}/{len(shard_pairs)}; shard {args.shard_id}] {subset}/{name}",
            flush=True,
        )
        with Image.open(path_a) as source_a, Image.open(path_b) as source_b:
            image_a = source_a.convert("RGB").copy()
            image_b = source_b.convert("RGB").copy()
        captions = load_or_create_captions(
            pipe, captioner, args, subset, name, image_a, image_b
        )
        for method in args.methods:
            run_method(
                api,
                pipe,
                args,
                method,
                subset,
                name,
                image_a,
                image_b,
                captions,
            )

    print(f"\nShard complete. Outputs at {args.out_root}", flush=True)


if __name__ == "__main__":
    main()

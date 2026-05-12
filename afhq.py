"""
AFHQ / CelebA-HQ Real Image Interpolation with EDM2.

Loads image pairs from HuggingFace datasets API, runs GFM interpolation.

Usage:
    # Dog → Cat interpolation
    python edm2_afhq_interp.py --dataset afhq --catA dog --catB cat --num_pairs 5

    # Cat → Wild interpolation  
    python edm2_afhq_interp.py --dataset afhq --catA cat --catB wild --num_pairs 3

    # CelebA face → face
    python edm2_afhq_interp.py --dataset celeba --num_pairs 5

    # Specific image indices
    python edm2_afhq_interp.py --dataset afhq --catA dog --catB cat --idxA 0 --idxB 0
"""

import os
import sys
import argparse
import torch
from PIL import Image

# EDM2 repo on path
EDM2_REPO = os.environ.get('EDM2_REPO', '/export_home/zzhixuan/code/edm2')
sys.path.insert(0, EDM2_REPO)

from gfm.path.edm import EDM2GFMPipeline, save_interpolation_strip, save_frames
from img_interp import run_real_image_interpolation


# ============================================================
# Dataset loading
# ============================================================

# AFHQ label mapping
AFHQ_LABEL_MAP = {'cat': 0, 'dog': 1, 'wild': 2}

# Best-matching ImageNet classes for EDM2 conditioning
IMAGENET_CLASS_MAP = {
    # AFHQ
    'dog': 207,      # golden retriever
    'cat': 281,      # tabby cat
    'wild': 292,     # tiger (closest wild animal)
    # CelebA - no perfect match, use low guidance
    'face': 0,       # placeholder; rely on low guidance
}


def load_afhq(category: str, cache_dir: str = None):
    """Load AFHQ split by category. Returns list of PIL Images."""
    from datasets import load_dataset
    
    print(f"Loading AFHQ dataset...")
    ds = load_dataset("huggan/AFHQ", cache_dir=cache_dir, trust_remote_code=True)
    
    label = AFHQ_LABEL_MAP[category]
    filtered = ds['train'].filter(lambda x: x['label'] == label)
    print(f"  {category}: {len(filtered)} images")
    return filtered


def load_celeba(cache_dir: str = None):
    """Load CelebA-HQ. Returns dataset object."""
    from datasets import load_dataset
    
    print(f"Loading CelebA-HQ dataset...")
    ds = load_dataset("flwrlabs/celeba", cache_dir=cache_dir, trust_remote_code=True)
    print(f"  {len(ds['train'])} images")
    return ds['train']


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AFHQ/CelebA GFM Interpolation")

    # Dataset
    parser.add_argument("--dataset", type=str, default="afhq",
                        choices=["afhq", "celeba"])
    parser.add_argument("--catA", type=str, default="dog",
                        help="Category for endpoint A (afhq: cat/dog/wild)")
    parser.add_argument("--catB", type=str, default="cat",
                        help="Category for endpoint B")
    parser.add_argument("--idxA", type=int, default=None,
                        help="Specific image index for A (None=random)")
    parser.add_argument("--idxB", type=int, default=None,
                        help="Specific image index for B (None=random)")
    parser.add_argument("--num_pairs", type=int, default=1,
                        help="Number of random pairs to interpolate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_cache", type=str, default=None,
                        help="HF datasets cache directory")

    # Model
    parser.add_argument("--preset", type=str, default="edm2-img512-s-guid-dino")
    parser.add_argument("--net", type=str, default=None)
    parser.add_argument("--gnet", type=str, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--vae", type=str, default="mse", choices=["mse", "ema"])

    # Manually override ImageNet class labels
    parser.add_argument("--classA", type=int, default=None,
                        help="Override ImageNet class for A")
    parser.add_argument("--classB", type=int, default=None,
                        help="Override ImageNet class for B")

    # GFM
    parser.add_argument("--noise_level", type=float, default=0.5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=800)
    parser.add_argument("--interpolator", type=str, default="el",
                        choices=["spherical", "el", "seq_el"])
    parser.add_argument("--num_denoise_steps", type=int, default=18)
    parser.add_argument("--resolution", type=int, default=512)

    # Output
    parser.add_argument("--output_dir", type=str, default="./output/edm2_afhq")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # ---- Resolve ImageNet class labels ----
    classA = args.classA if args.classA is not None else IMAGENET_CLASS_MAP.get(args.catA, 0)
    classB = args.classB if args.classB is not None else IMAGENET_CLASS_MAP.get(args.catB, 0)
    print(f"ImageNet conditioning: classA={classA} ({args.catA}), classB={classB} ({args.catB})")

    # ---- Load dataset ----
    import numpy as np
    rng = np.random.RandomState(args.seed)

    if args.dataset == "afhq":
        dsA = load_afhq(args.catA, cache_dir=args.data_cache)
        dsB = load_afhq(args.catB, cache_dir=args.data_cache)
    elif args.dataset == "celeba":
        ds_all = load_celeba(cache_dir=args.data_cache)
        dsA = ds_all
        dsB = ds_all
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ---- Generate pairs ----
    pairs = []
    if args.idxA is not None and args.idxB is not None:
        pairs.append((args.idxA, args.idxB))
    else:
        for _ in range(args.num_pairs):
            ia = rng.randint(len(dsA))
            ib = rng.randint(len(dsB))
            pairs.append((ia, ib))

    print(f"\nWill interpolate {len(pairs)} pair(s)")

    # ---- Load EDM2 pipeline ----
    extra = {}
    if args.guidance is not None:
        extra['guidance'] = args.guidance

    print(f"\nLoading EDM2 pipeline (preset={args.preset})...")
    pipe = EDM2GFMPipeline.load(
        preset=args.preset,
        net_path=args.net,
        gnet_path=args.gnet,
        vae_type=args.vae,
        device=args.device,
        **extra,
    )

    # ---- Run interpolation for each pair ----
    for pair_idx, (ia, ib) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"Pair {pair_idx+1}/{len(pairs)}: {args.catA}[{ia}] → {args.catB}[{ib}]")
        print(f"{'='*60}")

        imgA = dsA[ia]['image']
        imgB = dsB[ib]['image']

        # Ensure PIL RGB
        if not isinstance(imgA, Image.Image):
            imgA = Image.fromarray(imgA)
        if not isinstance(imgB, Image.Image):
            imgB = Image.fromarray(imgB)

        pair_dir = os.path.join(args.output_dir, f"pair_{pair_idx:03d}_{args.catA}{ia}_{args.catB}{ib}")

        results = run_real_image_interpolation(
            pipe,
            imgA=imgA,
            imgB=imgB,
            classA=classA,
            classB=classB,
            noise_level=args.noise_level,
            num_steps=args.num_steps,
            lam=args.lam,
            lr=args.lr,
            max_iters=args.max_iters,
            interpolator_type=args.interpolator,
            num_denoise_steps=args.num_denoise_steps,
            resolution=args.resolution,
        )

        # Save
        os.makedirs(pair_dir, exist_ok=True)
        save_interpolation_strip(
            results["images"],
            os.path.join(pair_dir, "strip.png"),
            size=(256, 256),
        )
        save_frames(results["images"], os.path.join(pair_dir, "frames"))

        # Save input images and reconstructions
        imgA.save(os.path.join(pair_dir, "input_A.png"))
        imgB.save(os.path.join(pair_dir, "input_B.png"))
        results["reconA"].save(os.path.join(pair_dir, "recon_A.png"))
        results["reconB"].save(os.path.join(pair_dir, "recon_B.png"))

        if results["losses"]:
            print(f"  Final GFM loss: {results['losses'][-1]:.6f}")

    print(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
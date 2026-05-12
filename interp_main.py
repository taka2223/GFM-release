"""
Unified CLI for EDM2 GFM Interpolation.
Supports three modes: 
 1. class: Random Generation Cross-class Interpolation
 2. real: Single Real Image Pair Interpolation
 3. dataset: Bulk Dataset (AFHQ/CelebA) Interpolation
"""
import os
import sys
import argparse
import torch
from PIL import Image

# 保证本地仓库在路径里
sys.path.insert(0, os.environ.get('EDM2_REPO', './edm2'))
from gfm.path.edm2 import EDM2GFMPipeline, run_gfm_interpolation, save_results

# 数据集和类别的简易映射
AFHQ_LABEL_MAP = {'cat': 0, 'dog': 1, 'wild': 2}
IMAGENET_CLASS_MAP = {'dog': 207, 'cat': 281, 'wild': 292, 'face': 0}

def get_args():
    parser = argparse.ArgumentParser(description="Unified GFM Interpolation with EDM2")
    parser.add_argument("--mode", choices=["class", "real", "dataset"], help="Interpolation mode")
    
    # Model Args
    parser.add_argument("--preset", type=str, default="edm2-img512-xxl-guid-dino")
    parser.add_argument("--device", type=str, default="cuda")
    
    # GFM Args
    parser.add_argument("--noise_level", type=float, default=0.5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=400)
    parser.add_argument("--interpolator", type=str, default="el", choices=["spherical", "el", "seq_el"])
    parser.add_argument("--num_denoise_steps", type=int, default=18)
    parser.add_argument("--output_dir", type=str, default="./output/edm2")

    # [Mode: class] Args
    parser.add_argument("--classA", type=int, default=207)
    parser.add_argument("--classB", type=int, default=281)
    parser.add_argument("--seedA", type=int, default=0)
    parser.add_argument("--seedB", type=int, default=1)

    # [Mode: real] Args
    parser.add_argument("--imgA", type=str, default=None)
    parser.add_argument("--imgB", type=str, default=None)

    # [Mode: dataset] Args
    parser.add_argument("--dataset", type=str, default="afhq", choices=["afhq", "celeba"])
    parser.add_argument("--catA", type=str, default="dog")
    parser.add_argument("--catB", type=str, default="cat")
    parser.add_argument("--num_pairs", type=int, default=1)

    return parser.parse_args()

def main():
    args = get_args()
    pipe = EDM2GFMPipeline.load(preset=args.preset, device=args.device)

    if args.mode == "class":
        print(f"--- Class Mode: {args.classA} -> {args.classB} ---")
        latA, imgA = pipe.model.sample(torch.tensor([args.classA], device=args.device), batch_seeds=[args.seedA]), None
        latB, imgB = pipe.model.sample(torch.tensor([args.classB], device=args.device), batch_seeds=[args.seedB]), None
        
        # 将潜在变量包裹给 pipeline 生成图像用于头尾展示
        imgA = pipe.autoencoder.decode(latA)
        imgB = pipe.autoencoder.decode(latB)

        res = run_gfm_interpolation(pipe, latA, latB, args.classA, args.classB, imgA, imgB, args)
        save_results(res['images'], args.output_dir, prefix="class_")

    elif args.mode == "real":
        print(f"--- Real Image Mode: {args.imgA} -> {args.imgB} ---")
        imgA, imgB = Image.open(args.imgA).convert("RGB"), Image.open(args.imgB).convert("RGB")
        latA = pipe.autoencoder.encode(imgA)
        latB = pipe.autoencoder.encode(imgB)
        
        res = run_gfm_interpolation(pipe, latA, latB, args.classA, args.classB, imgA, imgB, args)
        save_results(res['images'], args.output_dir, prefix="real_")

    elif args.mode == "dataset":
        print(f"--- Dataset Mode: {args.dataset.upper()} ({args.catA} -> {args.catB}) ---")
        from datasets import load_dataset
        import numpy as np
        
        if args.dataset == "afhq":
            ds = load_dataset("huggan/AFHQ", cache_dir='/cns/USERS/zzhixuan/data')['train']
            dsA = ds.filter(lambda x: x['label'] == AFHQ_LABEL_MAP[args.catA])
            dsB = ds.filter(lambda x: x['label'] == AFHQ_LABEL_MAP[args.catB])
        else:
            dsA = dsB = load_dataset("flwrlabs/celeba", trust_remote_code=True)['train']

        classA = IMAGENET_CLASS_MAP.get(args.catA, 0)
        classB = IMAGENET_CLASS_MAP.get(args.catB, 0)
        rng = np.random.RandomState(42)

        for i in range(args.num_pairs):
            print(f"\nProcessing Pair {i+1}/{args.num_pairs}...")
            idxA, idxB = rng.randint(len(dsA)), rng.randint(len(dsB))
            imgA = Image.fromarray(dsA[idxA]['image']) if not isinstance(dsA[idxA]['image'], Image.Image) else dsA[idxA]['image']
            imgB = Image.fromarray(dsB[idxB]['image']) if not isinstance(dsB[idxB]['image'], Image.Image) else dsB[idxB]['image']
            
            latA, latB = pipe.autoencoder.encode(imgA), pipe.autoencoder.encode(imgB)
            res = run_gfm_interpolation(pipe, latA, latB, classA, classB, imgA, imgB, args)
            
            pair_dir = os.path.join(args.output_dir, f"pair_{i:02d}_{args.catA}_{args.catB}")
            save_results(res['images'], pair_dir)

if __name__ == "__main__":
    main()
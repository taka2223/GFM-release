"""
Minimal test for noise-denoise consistency using GeodesicInterpolator.
"""
from ast import arg
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import numpy as np
import mcubes
import torch
import trimesh
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from gfm.models import models_class_cond, models_ae
from gfm.path.geodesic_interpolation import GeodesicInterpolator,RBFKernelInterpolator, LandInterpolator, ScoreBasedInterpolator, ELInterpolator, SteinScoreInterpolator
from gfm.helper.help import decode_and_save, load_model_from_path

def main(args):
    SIGMA = args.noise_level
    category = args.category
    seed = args.seed
    num_pairs = args.num_pairs
    batch_size = args.batch_size  # 新增
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    root = os.path.join(args.output_dir, args.dataset, args.approximator)
    os.makedirs(root, exist_ok=True)
    output_dir = os.path.join(root,
        f"test_denoise_sigma{SIGMA}_category{category}_seed{seed}_pairs{num_pairs}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    ae_name = args.ae_name
    ae_path = args.ae_path
    ae = load_model_from_path(
        model_name=ae_name,
        ckpt_path=ae_path,
        registry=models_ae.__dict__,
        device=device
    )
    
    dm_name = args.dm_name
    dm_path = args.dm_path
    dm = load_model_from_path(
        model_name=dm_name,
        ckpt_path=dm_path,
        registry=models_class_cond.__dict__,
        device=device
    )

    if args.approximator == 'rbf':
        interpolator = RBFKernelInterpolator(dm, ae, device=device)
        interpolator.load(args.h_path)
    elif args.approximator == 'land':
        interpolator = LandInterpolator(dm, ae, device=device)
        interpolator.load(args.h_path)
    elif args.approximator == 'score':
        interpolator = ScoreBasedInterpolator(dm, ae, device=device)
    elif args.approximator == 'el':
        interpolator = ELInterpolator(dm, ae, device=device)
    elif args.approximator == 'stein':
        interpolator = SteinScoreInterpolator(dm, ae, device=device)
    else:
        raise ValueError    
    
    # 计算批次数
    num_batches = (num_pairs + batch_size - 1) // batch_size
    print(f"\n[Info] Total pairs: {num_pairs}, batch_size: {batch_size}, num_batches: {num_batches}", flush=True)
    
    all_paths = []
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        # 当前批次的起止索引
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_pairs)
        current_batch_size = end_idx - start_idx
        
        print(f"\n[Batch {batch_idx+1}/{num_batches}] Processing pairs {start_idx} to {end_idx-1}", flush=True)
        
        ## Step 1: 采样当前批次的端点
        print(f"  [Step 1] Sampling {current_batch_size} pairs of clean latents...", flush=True)
        with torch.no_grad():
            # 每对需要2个点
            class_labels = torch.tensor([category] * (2 * current_batch_size), device=device).long()
            # seed 偏移：每个batch使用不同的seed范围
            batch_seed_start = seed + 2 * start_idx
            batch_seeds = torch.arange(batch_seed_start, batch_seed_start + 2 * current_batch_size, device=device)
            clean_latents = dm.sample(cond=class_labels, batch_seeds=batch_seeds).float()
        
        # 分成 start 和 end
        start_latents = clean_latents[0::2]  # [current_batch_size, ...]
        end_latents = clean_latents[1::2]    # [current_batch_size, ...]
        
        print(f"  Clean latents: start={start_latents.shape}, end={end_latents.shape}", flush=True)
        
        ## Step 2: 优化路径
        print(f"  [Step 2] Optimizing paths...", flush=True)
        
        if args.approximator not in ['rbf', 'land']:
            # 批量加噪
            noisy_start = interpolator.add_noise(start_latents, sigma=SIGMA)
            noisy_end = interpolator.add_noise(end_latents, sigma=SIGMA)
            
            batch_class_labels = torch.tensor([category] * current_batch_size, device=device).long()
            paths, info = interpolator.optimize_path(
                start_latent=noisy_start,
                end_latent=noisy_end,
                num_steps=10, 
                lr=args.lr, 
                sigma=SIGMA,
                class_label=batch_class_labels,
                max_iters=args.max_iters
            )
            
            # 批量去噪
            paths = interpolator.denoise_path(paths, sigma_start=SIGMA, 
                                              class_label=batch_class_labels, 
                                              num_denoise_steps=5)
        else:
            paths, info = interpolator.optimize_path(
                start_latent=start_latents,
                end_latent=end_latents,
                num_steps=10, 
                lr=args.lr
            )
        
        # 保存当前批次的路径（移到CPU以节省GPU内存）
        all_paths.append(paths.cpu())
        
        # 清理GPU内存
        del clean_latents, start_latents, end_latents, paths
        if args.approximator not in ['rbf', 'land']:
            del noisy_start, noisy_end
        torch.cuda.empty_cache()
    
    # 合并所有路径
    all_paths = torch.cat(all_paths, dim=0)  # [num_pairs, num_steps, ...]
    print(f"\n[Info] All paths shape: {all_paths.shape}", flush=True)
    
    # Step 3: 顺序解码保存
    print(f"\n[Step 3] Saving {num_pairs} paths (sequential decode)...", flush=True)
    for pair_idx in tqdm(range(num_pairs), desc="Decoding"):
        for step_idx in range(all_paths.shape[1]):
            latent = all_paths[pair_idx, step_idx].unsqueeze(0).to(device)
            decode_and_save(
                ae=ae,
                latent=latent,
                save_path=os.path.join(output_dir, f"pair{pair_idx}_{args.approximator}_step{step_idx}.obj"),
                device=device
            )
        
        

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--dataset', type=str, default='shapenet')
    parser.add_argument('--approximator', type=str, default='rbf',
                        help='approximator type')
    parser.add_argument('--h_path', type=str, default='output/kernel/shapenet/rbf/h.pth',
                        help='path to pre-trained kernel (RBF or Land)')
    parser.add_argument('--ae_name', type=str, default='kl_d512_m512_l8')
    parser.add_argument('--ae_path', type=str, default='output/ae/kl_d512_m512_l8/checkpoint-199.pth')
    parser.add_argument('--dm_name', type=str, default='kl_d512_m512_l8_d24_edm')
    parser.add_argument('--dm_path', type=str, default='output/class_cond_dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth')
    parser.add_argument('--noise_level', type=float, default=0.10)
    parser.add_argument('--category', type=int, default=5)
    parser.add_argument('--seed', type=int, default=99995)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_pairs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for sampling and optimization')
    parser.add_argument('--max_iters', type=int, default=400, help='max iterations for optimization')
    
    args = parser.parse_args()
    main(args)
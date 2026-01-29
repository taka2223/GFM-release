"""
Minimal test for noise-denoise consistency using GeodesicInterpolator.
"""
from ast import arg
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
import mcubes
import torch
import trimesh
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from gfm.models import models_class_cond, models_ae
from gfm.path.geodesic_interpolation import GeodesicInterpolator,RBFKernelInterpolator, LandInterpolator, ScoreBasedInterpolator, ELInterpolator, SteinScoreInterpolator
from gfm.helper.help import decode_and_save, load_model_from_path


def benchmark_optimize_path(interpolator, start_latent, end_latent, args, 
                            class_label, sigma, warmup=1, runs=3):
    """Benchmark optimization with proper CUDA timing."""
    
    # Warmup runs 
    for _ in range(warmup):
        _ = interpolator.optimize_path(
            start_latent=start_latent.clone(),
            end_latent=end_latent.clone(),
            num_steps=10, 
            lr=args.lr, 
            sigma=sigma,
            class_label=class_label,
            max_iters=5
        )
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    times = []
    all_losses = []
    final_path = None
    
    for run_idx in range(runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start_event.record()
        
        paths, info = interpolator.optimize_path(
            start_latent=start_latent.clone(),
            end_latent=end_latent.clone(),
            num_steps=10, 
            lr=args.lr, 
            sigma=sigma,
            class_label=class_label,
            max_iters=args.max_iters
        )
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms / 1000)  # convert to seconds
        all_losses.append(info['losses'])
        final_path = paths
    
    peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
    
    return final_path, {
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'memory_gb': peak_memory_gb,
        'losses': all_losses[-1],
        'all_losses': all_losses,
        'iterations': len(all_losses[-1]),
        'final_loss': all_losses[-1][-1] if all_losses[-1] else None,
    }


def save_convergence_plot(all_stats, output_dir, approximator):
    """Save convergence curve plot."""
    plt.figure(figsize=(8, 5))
    
    losses = all_stats['losses']
    iterations = range(1, len(losses) + 1)
    
    plt.plot(iterations, losses, label=approximator, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log')
    plt.title(f'Convergence Behavior - {approximator}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'convergence_{approximator}.pdf'), dpi=150)
    plt.savefig(os.path.join(output_dir, f'convergence_{approximator}.png'), dpi=150)
    plt.close()


def main(args):
    SIGMA = args.noise_level
    category = args.category
    seed = args.seed
    num_pairs = args.num_pairs
    batch_size = args.batch_size
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

    num_batches = (num_pairs + batch_size - 1) // batch_size
    print(f"\n[Info] Total pairs: {num_pairs}, batch_size: {batch_size}, num_batches: {num_batches}", flush=True)
    
    all_paths = []
    all_stats = {
        'times': [],
        'memories': [],
        'final_losses': [],
        'losses_per_batch': [],
    }
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_pairs)
        current_batch_size = end_idx - start_idx
        
        print(f"\n[Batch {batch_idx+1}/{num_batches}] Processing pairs {start_idx} to {end_idx-1}", flush=True)

        print(f"  [Step 1] Sampling {current_batch_size} pairs of clean latents...", flush=True)
        with torch.no_grad():
            class_labels = torch.tensor([category] * (2 * current_batch_size), device=device).long()
            batch_seed_start = seed + 2 * start_idx
            batch_seeds = torch.arange(batch_seed_start, batch_seed_start + 2 * current_batch_size, device=device)
            clean_latents = dm.sample(cond=class_labels, batch_seeds=batch_seeds).float()
        
        start_latents = clean_latents[0::2]
        end_latents = clean_latents[1::2]
        
        print(f"  Clean latents: start={start_latents.shape}, end={end_latents.shape}", flush=True)

        print(f"  [Step 2] Optimizing paths...", flush=True)
        
        if args.approximator not in ['rbf', 'land']:
            noisy_start = interpolator.add_noise(start_latents, sigma=SIGMA)
            noisy_end = interpolator.add_noise(end_latents, sigma=SIGMA)
            batch_class_labels = torch.tensor([category] * current_batch_size, device=device).long()
            
            if args.benchmark:
                paths, stats = benchmark_optimize_path(
                    interpolator=interpolator,
                    start_latent=noisy_start[:1],
                    end_latent=noisy_end[:1],
                    args=args,
                    class_label=batch_class_labels[:1],
                    sigma=SIGMA,
                    warmup=args.warmup_runs,
                    runs=args.benchmark_runs
                )

                all_stats['times'].append(stats['time_mean'])
                all_stats['memories'].append(stats['memory_gb'])
                all_stats['final_losses'].append(stats['final_loss'])
                all_stats['losses_per_batch'].append(stats['losses'])
                
                print(f"  [Benchmark] Time: {stats['time_mean']:.3f} ± {stats['time_std']:.3f} s, "
                      f"Memory: {stats['memory_gb']:.2f} GB, "
                      f"Final Loss: {stats['final_loss']:.2e}", flush=True)

                if args.benchmark and all_stats['times']:
                    summary = {
                        'method': args.approximator,
                        'sigma': SIGMA,
                        'time_mean': float(np.mean(all_stats['times'])),
                        'time_std': float(np.std(all_stats['times'])),
                        'memory_gb': float(np.max(all_stats['memories'])),
                        'final_loss_mean': float(np.mean([l for l in all_stats['final_losses'] if l is not None])),
                        'iterations': args.max_iters,
                    }
                    
                    print(f"\n{'='*50}")
                    print(f"[Summary] {args.approximator}")
                    print(f"  Time: {summary['time_mean']:.3f} ± {summary['time_std']:.3f} s")
                    print(f"  Peak Memory: {summary['memory_gb']:.2f} GB")
                    print(f"  Final Loss: {summary['final_loss_mean']:.2e}")
                    print(f"  Iterations: {summary['iterations']}")
                    print(f"{'='*50}\n")

                    with open(os.path.join(output_dir, 'benchmark_stats.json'), 'w') as f:
                        json.dump(summary, f, indent=2)

                    if all_stats['losses_per_batch']:
                        avg_losses = all_stats['losses_per_batch'][0]
                        save_convergence_plot({'losses': avg_losses}, output_dir, args.approximator)

                        np.save(os.path.join(output_dir, 'losses.npy'), np.array(avg_losses))

                paths, info = interpolator.optimize_path(
                    start_latent=noisy_start,
                    end_latent=noisy_end,
                    num_steps=10, 
                    lr=args.lr, 
                    sigma=SIGMA,
                    class_label=batch_class_labels,
                    max_iters=args.max_iters
                )
            else:
                paths, info = interpolator.optimize_path(
                    start_latent=noisy_start,
                    end_latent=noisy_end,
                    num_steps=10, 
                    lr=args.lr, 
                    sigma=SIGMA,
                    class_label=batch_class_labels,
                    max_iters=args.max_iters
                )
                all_stats['losses_per_batch'].append(info.get('losses', []))
            
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
        
        all_paths.append(paths.cpu())
        
        del clean_latents, start_latents, end_latents, paths
        if args.approximator not in ['rbf', 'land']:
            del noisy_start, noisy_end
        torch.cuda.empty_cache()

    all_paths = torch.cat(all_paths, dim=0)
    print(f"\n[Info] All paths shape: {all_paths.shape}", flush=True)

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

    parser.add_argument('--benchmark', action='store_true', help='enable benchmarking')
    parser.add_argument('--warmup_runs', type=int, default=1, help='warmup runs for benchmark')
    parser.add_argument('--benchmark_runs', type=int, default=3, help='number of runs for timing')
    
    args = parser.parse_args()
    main(args)
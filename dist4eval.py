"""
Point Cloud Evaluation Script (KeOps + GPU FPS).

Metrics:
    - Chamfer Distance (CD): Baseline metric
    - Hausdorff Distance: Worst-case, sensitive to outliers/artifacts
    - F-Score@τ: Intuitive "pass rate" at distance threshold

Feature:
    - 100% GPU Accelerated.
    - Exact "Hard" Chamfer Distance using PyKeOps.
    - GPU Batched FPS: Load high-res -> FPS -> target resolution.
    - Proper normalization and separate Precision/Recall reporting.

Usage:
    python eval_full.py \
        --gen_dir /path/to/generated \
        --ref_dir /path/to/reference \
        --output_dir results/
"""

import os
import re
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from pykeops.torch import LazyTensor
from torch_cluster import fps

# ============ 0. Naming & Path Parsing ============

def parse_experiment_info(gen_dir_path: Path) -> dict:
    parts = gen_dir_path.parts
    name = gen_dir_path.name
    
    info = {'method': 'unknown', 'sigma': 'unknown', 'category': 'unknown'}
    
    for i, part in enumerate(parts):
        if part.startswith('test_denoise') or part == name:
            if i > 0:
                info['method'] = parts[i - 1]
            break
            
    sigma_match = re.search(r'sigma([0-9.]+)', name)
    cat_match = re.search(r'category([0-9]+)', name)
    
    if sigma_match: info['sigma'] = sigma_match.group(1)
    if cat_match: info['category'] = cat_match.group(1)
        
    return info

# ============ 1. Loading & FPS (CPU -> GPU) ============

def load_and_sample(path: str, num_sample: int) -> np.ndarray:
    """CPU: Random sample high-res points (e.g. 16384)"""
    try:
        mesh = trimesh.load(path, force='mesh')
        points, _ = trimesh.sample.sample_surface(mesh, num_sample)
        return points.astype(np.float32)
    except Exception as e:
        return np.zeros((num_sample, 3), dtype=np.float32)

def fps_batch(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    GPU: Farthest Point Sampling batch-wise.
    points: [B, N, 3]
    num_samples: Target number of points (e.g. 2048)
    """
    B, N, D = points.shape
    if N <= num_samples:
        return points
    
    # Flatten for torch_cluster.fps
    src = points.reshape(-1, D)
    batch_ptr = torch.arange(0, (B + 1) * N, N, device=points.device)
    
    # Ratio for FPS
    ratio = num_samples / N
    
    # Get indices [B * num_samples]
    idx = fps(src, ptr=batch_ptr, ratio=ratio, random_start=True)
    
    idx = idx.reshape(B, -1)
    
    # Handle potential size mismatch if any (rare with standard shapes)
    if idx.shape[1] != num_samples:
        idx = idx[:, :num_samples]

    relative_idx = idx % N
    sampled = torch.gather(points, 1, relative_idx.unsqueeze(-1).expand(-1, -1, D))
    return sampled

def normalize_pointcloud(points: torch.Tensor) -> torch.Tensor:
    """
    Normalize point clouds to unit sphere centered at origin.
    points: [B, N, 3]
    """
    centroid = points.mean(dim=1, keepdim=True)  # [B, 1, 3]
    points = points - centroid
    # Max distance from origin per point cloud
    scale = points.norm(dim=-1).max(dim=1, keepdim=True)[0].unsqueeze(-1)  # [B, 1, 1]
    scale = scale.clamp(min=1e-8)  # Avoid division by zero
    return points / scale

def load_data_with_fps(paths: list[str], in_points: int, out_points: int, num_workers: int, normalize: bool = True) -> torch.Tensor:
    """
    Load high-res (CPU) -> FPS downsample (GPU) -> Normalize.
    """
    print(f"Loading {len(paths)} meshes (Input: {in_points} -> FPS -> Output: {out_points})...")
    
    # 1. CPU Loading
    data_list = np.zeros((len(paths), in_points, 3), dtype=np.float32)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(load_and_sample, p, in_points): i for i, p in enumerate(paths)}
        for future in tqdm(as_completed(future_to_idx), total=len(paths), desc="CPU Loading"):
            idx = future_to_idx[future]
            data_list[idx] = future.result()
            
    # 2. GPU FPS (Batched)
    print("Running GPU FPS...")
    batch_size = 128
    final_data = []
    
    tensor_data = torch.from_numpy(data_list)  # [Total, 16384, 3] (CPU)
    
    for i in tqdm(range(0, len(paths), batch_size), desc="GPU FPS"):
        batch = tensor_data[i : i+batch_size].cuda()
        sampled = fps_batch(batch, out_points)  # [B, 2048, 3]
        if normalize:
            sampled = normalize_pointcloud(sampled)
        final_data.append(sampled.cpu())  # Move back to CPU to store temporarily
        
    return torch.cat(final_data, dim=0).cuda()  # Final move to GPU for Metric Compute

# ============ 2. Fast Distance Kernels ============

def chamfer_distance_kernel(x, y):
    """
    PyKeOps Hard Chamfer Distance (Corrected Dimensions)
    x: [B, N, 3]
    y: [B, N, 3]
    Returns squared distances (sum of both directions)
    """
    x_i = LazyTensor(x[:, :, None, :])  # [B, N, 1, 3]
    y_j = LazyTensor(y[:, None, :, :])  # [B, 1, N, 3]

    D_ij = ((x_i - y_j) ** 2).sum(-1)
    
    # min over y for each x
    min_dist_xy = D_ij.min(dim=2)  # [B, N, 1]
    # min over x for each y
    min_dist_yx = D_ij.min(dim=1)  # [B, 1, N]
    
    v_xy = min_dist_xy.view(x.shape[0], -1)
    v_yx = min_dist_yx.view(x.shape[0], -1)
    
    return v_xy.mean(dim=1) + v_yx.mean(dim=1)

# ============ 3. Matrix Computation ============

def compute_matrix(points_gen: torch.Tensor, points_ref: torch.Tensor, batch_size: int = 64) -> np.ndarray:
    """
    Compute pairwise Chamfer Distance matrix.
    Uses repeat_interleave for high GPU parallelism.
    """
    n_gen = points_gen.shape[0]
    n_ref = points_ref.shape[0]
    matrix = np.zeros((n_gen, n_ref), dtype=np.float32)
    
    print(f"Computing CD matrix ({n_gen}x{n_ref})...")
    
    for i in tqdm(range(0, n_gen, batch_size), desc="CD rows"):
        i_end = min(i + batch_size, n_gen)
        gen_batch = points_gen[i:i_end]
        
        row_dists = []
        for j in range(0, n_ref, batch_size):
            j_end = min(j + batch_size, n_ref)
            ref_batch = points_ref[j:j_end]
            
            b_gen = gen_batch.shape[0]
            b_ref = ref_batch.shape[0]
            
            # Expand for pairwise comparison
            g_rep = gen_batch.repeat_interleave(b_ref, dim=0)  # [b_gen * b_ref, N, 3]
            r_rep = ref_batch.repeat(b_gen, 1, 1)              # [b_gen * b_ref, M, 3]
            
            with torch.no_grad():
                d = chamfer_distance_kernel(g_rep, r_rep)
            d = d.view(b_gen, b_ref).cpu().numpy()
            row_dists.append(d)
        
        matrix[i:i_end, :] = np.concatenate(row_dists, axis=1)
    
    return matrix


def compute_min_distances(points_gen: torch.Tensor, points_ref: torch.Tensor, batch_size: int = 64):
    """
    Memory-efficient: Only compute min distances, not full matrix.
    Still uses repeat_interleave for speed.
    
    Returns:
        min_gen2ref: [n_gen] - min CD from each gen to any ref
        argmin_gen2ref: [n_gen] - index of closest ref for each gen (for COV)
        min_ref2gen: [n_ref] - min CD from each ref to any gen
    """
    n_gen = points_gen.shape[0]
    n_ref = points_ref.shape[0]
    
    min_gen2ref = torch.full((n_gen,), float('inf'), device=points_gen.device)
    argmin_gen2ref = torch.zeros(n_gen, dtype=torch.long, device=points_gen.device)
    min_ref2gen = torch.full((n_ref,), float('inf'), device=points_ref.device)
    
    print(f"Computing min distances ({n_gen} gen, {n_ref} ref)...")
    
    for i in tqdm(range(0, n_gen, batch_size), desc="Processing"):
        i_end = min(i + batch_size, n_gen)
        gen_batch = points_gen[i:i_end]
        b_gen = gen_batch.shape[0]
        
        for j in range(0, n_ref, batch_size):
            j_end = min(j + batch_size, n_ref)
            ref_batch = points_ref[j:j_end]
            b_ref = ref_batch.shape[0]
            
            # Expand for pairwise comparison
            g_rep = gen_batch.repeat_interleave(b_ref, dim=0)
            r_rep = ref_batch.repeat(b_gen, 1, 1)
            
            with torch.no_grad():
                d = chamfer_distance_kernel(g_rep, r_rep)
            cd_block = d.view(b_gen, b_ref)  # [b_gen, b_ref]
            
            # Update min_gen2ref and argmin
            block_min, block_argmin = cd_block.min(dim=1)  # [b_gen]
            update_mask = block_min < min_gen2ref[i:i_end]
            min_gen2ref[i:i_end] = torch.where(update_mask, block_min, min_gen2ref[i:i_end])
            argmin_gen2ref[i:i_end] = torch.where(update_mask, j + block_argmin, argmin_gen2ref[i:i_end])
            
            # Update min_ref2gen
            block_min_ref = cd_block.min(dim=0)[0]  # [b_ref]
            min_ref2gen[j:j_end] = torch.minimum(min_ref2gen[j:j_end], block_min_ref)
    
    return (
        min_gen2ref.cpu().numpy(),
        argmin_gen2ref.cpu().numpy(),
        min_ref2gen.cpu().numpy()
    )

# ============ 4. Metrics & Main ============

def calculate_metrics_from_matrix(matrix: np.ndarray, thresholds: list = [0.05, 0.1, 0.15]) -> dict:
    """
    Calculate metrics from full CD distance matrix.
    """
    results = {}
    
    # Convert to actual distances (matrix is squared)
    dist_matrix = np.sqrt(matrix)
    
    min_gen2ref = dist_matrix.min(axis=1)
    min_ref2gen = dist_matrix.min(axis=0)
    
    # CD
    results["CD-P"] = float(min_gen2ref.mean())
    results["CD-R"] = float(min_ref2gen.mean())
    results["CD"] = (results["CD-P"] + results["CD-R"]) / 2
    
    # Hausdorff (99th percentile, more robust than max)
    results["HD-P"] = float(np.percentile(min_gen2ref, 99))
    results["HD-R"] = float(np.percentile(min_ref2gen, 99))
    results["HD"] = max(results["HD-P"], results["HD-R"])
    
    # F-Score
    for tau in thresholds:
        precision = float((min_gen2ref < tau).mean())
        recall = float((min_ref2gen < tau).mean())
        fscore = 2 * precision * recall / (precision + recall + 1e-8)
        results[f"F@{tau}-P"] = precision
        results[f"F@{tau}-R"] = recall
        results[f"F@{tau}"] = float(fscore)
    
    # Percentiles
    results["CD-P50"] = float(np.percentile(min_gen2ref, 50))
    results["CD-P90"] = float(np.percentile(min_gen2ref, 90))
    
    # Coverage
    closest_ref_indices = matrix.argmin(axis=1)
    unique_refs = np.unique(closest_ref_indices)
    results["COV"] = float(len(unique_refs) / matrix.shape[1])
    
    return results


def calculate_metrics_from_min(
    min_gen2ref: np.ndarray, 
    argmin_gen2ref: np.ndarray,
    min_ref2gen: np.ndarray,
    n_ref: int,
    thresholds: list = [0.05, 0.1, 0.15]
) -> dict:
    """
    Calculate metrics from pre-computed min distances (memory efficient).
    """
    results = {}
    
    # Convert to actual distances (input is squared)
    min_gen2ref = np.sqrt(min_gen2ref)
    min_ref2gen = np.sqrt(min_ref2gen)
    
    # CD
    results["CD-P"] = float(min_gen2ref.mean())
    results["CD-R"] = float(min_ref2gen.mean())
    results["CD"] = (results["CD-P"] + results["CD-R"]) / 2
    
    # Hausdorff (99th percentile, more robust than max)
    results["HD-P"] = float(np.percentile(min_gen2ref, 99))
    results["HD-R"] = float(np.percentile(min_ref2gen, 99))
    results["HD"] = max(results["HD-P"], results["HD-R"])
    
    # F-Score
    for tau in thresholds:
        precision = float((min_gen2ref < tau).mean())
        recall = float((min_ref2gen < tau).mean())
        fscore = 2 * precision * recall / (precision + recall + 1e-8)
        results[f"F@{tau}-P"] = precision
        results[f"F@{tau}-R"] = recall
        results[f"F@{tau}"] = float(fscore)
    
    # Percentiles
    results["CD-P50"] = float(np.percentile(min_gen2ref, 50))
    results["CD-P90"] = float(np.percentile(min_gen2ref, 90))
    
    # Coverage
    unique_refs = np.unique(argmin_gen2ref)
    results["COV"] = float(len(unique_refs) / n_ref)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True)
    parser.add_argument("--ref_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/")
    
    # Points Config
    parser.add_argument("--in_points", type=int, default=16384, help="Points loaded from mesh (CPU)")
    parser.add_argument("--out_points", type=int, default=4096, help="Points after FPS (GPU) for metric")
    
    # F-Score thresholds (adjusted for normalized point clouds)
    parser.add_argument("--thresholds", type=float, nargs='+', default=[0.05, 0.1, 0.15],
                        help="Thresholds for F-Score computation")
    
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_normalize", action="store_true", help="Disable point cloud normalization")
    parser.add_argument("--step_filter", type=str, default=None, 
                        help="Regex pattern to filter files, e.g. 'step[3-6]'")
    parser.add_argument("--fast", action="store_true",
                        help="Memory-efficient mode: don't store full matrix")
    args = parser.parse_args()
    
    # 0. Path Parsing
    gen_dir_path = Path(args.gen_dir)
    exp_info = parse_experiment_info(gen_dir_path)
    
    out_name = f"eval_{exp_info['method']}_sigma{exp_info['sigma']}_cat{exp_info['category']}.npz"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / out_name
    
    print("="*60)
    print(f"Experiment: {exp_info}")
    print(f"FPS: {args.in_points} -> {args.out_points}")
    print(f"Normalize: {not args.no_normalize}")
    print(f"F-Score thresholds: {args.thresholds}")
    print(f"Fast mode: {args.fast}")
    print(f"Output: {output_path}")
    print("="*60)
    
    # 1. Load Data (With FPS)
    gen_paths = sorted(gen_dir_path.glob("*.obj"))
    if args.step_filter:
        pattern = re.compile(args.step_filter)
        gen_paths = [p for p in gen_paths if pattern.search(p.name)]
    gen_paths = gen_paths[:args.limit]
    
    # Ref pattern handling (keep backward compatibility)
    ref_paths = sorted(Path(args.ref_dir).glob("47-*.obj"))[:args.limit]
    if len(ref_paths) == 0: 
        ref_paths = sorted(Path(args.ref_dir).glob("*.obj"))[:args.limit]
    
    if not gen_paths or not ref_paths:
        print("Error: No files found.")
        return

    normalize = not args.no_normalize
    
    # Use load_data_with_fps
    gen_tensor = load_data_with_fps(
        [str(p) for p in gen_paths], 
        args.in_points, 
        args.out_points, 
        32,
        normalize=normalize
    )
    
    ref_tensor = load_data_with_fps(
        [str(p) for p in ref_paths], 
        args.in_points, 
        args.out_points, 
        32,
        normalize=normalize
    )
    
    # 2. Compute distances
    if args.fast:
        # Memory-efficient mode
        min_gen2ref, argmin_gen2ref, min_ref2gen = compute_min_distances(
            gen_tensor, ref_tensor, args.batch_size
        )
        metrics = calculate_metrics_from_min(
            min_gen2ref, argmin_gen2ref, min_ref2gen,
            n_ref=len(ref_paths),
            thresholds=args.thresholds
        )
        results = {
            "exp_info": exp_info,
            "args": vars(args),
            "n_gen": len(gen_paths),
            "n_ref": len(ref_paths),
            "min_gen2ref": min_gen2ref,
            "argmin_gen2ref": argmin_gen2ref,
            "min_ref2gen": min_ref2gen,
            **metrics
        }
    else:
        # Full matrix mode
        matrix = compute_matrix(gen_tensor, ref_tensor, args.batch_size)
        metrics = calculate_metrics_from_matrix(matrix, thresholds=args.thresholds)
        results = {
            "exp_info": exp_info,
            "args": vars(args),
            "n_gen": len(gen_paths),
            "n_ref": len(ref_paths),
            "cd_matrix": matrix,
            **metrics
        }
    
    # Print Results
    print("\n" + "="*60)
    print(f"Results: {exp_info['method']} | sigma={exp_info['sigma']} | cat={exp_info['category']}")
    print(f"Generated: {results['n_gen']} | Reference: {results['n_ref']}")
    print("="*60)
    
    # Chamfer Distance
    print("\n--- Chamfer Distance (↓ lower is better) ---")
    print(f"  CD-P (Fidelity):    {metrics['CD-P']:.4f}")
    print(f"  CD-R (Diversity):   {metrics['CD-R']:.4f}")
    print(f"  CD   (Symmetric):   {metrics['CD']:.4f}")
    print(f"  CD-P50/P90:         {metrics['CD-P50']:.4f} / {metrics['CD-P90']:.4f}")
    
    # Hausdorff Distance
    print("\n--- Hausdorff Distance 99% (↓ lower is better, sensitive to outliers) ---")
    print(f"  HD-P (Fidelity):    {metrics['HD-P']:.4f}")
    print(f"  HD-R (Diversity):   {metrics['HD-R']:.4f}")
    print(f"  HD   (Max):         {metrics['HD']:.4f}")
    
    # F-Score
    print("\n--- F-Score (↑ higher is better) ---")
    for tau in args.thresholds:
        print(f"  F@{tau}: {metrics[f'F@{tau}']:.4f}  (P: {metrics[f'F@{tau}-P']:.4f}, R: {metrics[f'F@{tau}-R']:.4f})")
    
    # Coverage
    print(f"\n--- Coverage ---")
    print(f"  COV: {metrics['COV']:.4f} ({metrics['COV']*100:.1f}%)")
    
    print("\n" + "="*60)
    
    np.savez(output_path, **results)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
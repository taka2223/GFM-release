# GFM: Geodesic Flow Matching on Learned Latent Spaces

**Official PyTorch Implementation**

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA compatible GPU

### Install from source
```bash
# Clone the repository
git clone https://github.com/anonymous/GFM-release.git
cd GFM-release

# Install dependencies
pip install -e .

# Additional dependencies
pip install pykeops torch-cluster accelerate
```
+ Note: pykeops is optional and only for metric calculation. It requires Cuda toolkit >=10.0 installed. 

### Dependencies
```
torch>=1.10.0
torchvision
numpy
einops
timm
torch-cluster
PyMCubes
trimesh
matplotlib
tqdm
PyYAML
scipy
pykeops
accelerate
```

## Scripts Overview

| Script | Description |
|--------|-------------|
| `data2fit.py` | Generate samples for fitting RBF/LAND kernels and distribution-level reference sets |
| `pair4vis.py` | Generate single interpolation trajectory for visualization |
| `optim_path.py` | Generate distribution-level geodesic samples |
| `dist4eval.py` | Evaluate metrics (CD, F-Score, etc.) between point cloud distributions |
| `decode.py` | Decode latent representations to 3D meshes |
| `test_benchmark.py` | Benchmark computation time and memory usage |
| `sample_class_cond.py` | Sample from the class-conditioned diffusion model |

## Usage

### 1. Generate Reference Samples (for kernel fitting)
```bash
python data2fit.py \
    --dm kl_d512_m512_l8_d24_edm \
    --dm-pth output/dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth
```

```bash
python gfm/train_kernel.py \
    --latents_pth /path/to/latents_gen_by_data2fit.pth \
```

### 2. Distribution-level Geodesic Interpolation
Generate geodesic paths between sampled latent pairs:
```bash
python optim_path.py \
    --dataset shapenet \
    --approximator rbf \
    --h_path output/kernel/shapenet/rbf/h.pth \
    --ae_name kl_d512_m512_l8 \
    --ae_path output/ae/kl_d512_m512_l8/checkpoint-199.pth \
    --dm_name kl_d512_m512_l8_d24_edm \
    --dm_path output/dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth \
    --category 5 \
    --noise_level 0.3 \
    --num_pairs 200 \
    --batch_size 16 \
    --max_iters 500 \
    --output_dir ./output/shapenet
```

Available approximators: `rbf`, `land`, `score`, `el`, `stein`

### 3. Single Trajectory Visualization
Generate a single geodesic path for visualization:
```bash
python pair4vis.py \
    --dataset shapenet \
    --approximator rbf \
    --h_path output/kernel/shapenet/rbf/h.pth \
    --ae_name kl_d512_m512_l8 \
    --ae_path output/ae/kl_d512_m512_l8/checkpoint-199.pth \
    --dm_name kl_d512_m512_l8_d24_edm \
    --dm_path output/dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth \
    --category 5 \
    --noise_level 0.3 \
    --output_dir ./output/vis
```

### 4. Decode Latents to Meshes
```bash
python decode.py \
    --ae kl_d512_m512_l8 \
    --ae-pth output/ae/kl_d512_m512_l8/checkpoint-199.pth \
    --latents-path ./output/latents.pth \
    --output-dir ./output/meshes \
    --batch-size 4 \
    --density 128
```

For multi-GPU decoding:
```bash
accelerate launch decode.py \
    --ae kl_d512_m512_l8 \
    --ae-pth output/ae/kl_d512_m512_l8/checkpoint-199.pth \
    --latents-path ./output/latents.pth \
    --output-dir ./output/meshes
```

### 5. Evaluate Distribution Metrics
Compute Chamfer Distance, F-Score etc. between generated and reference point clouds:
```bash
python dist4eval.py \
    --gen_dir ./output/shapenet/rbf/test_denoise_sigma0.3_category5_seed99995_pairs200 \
    --ref_dir ./data/shapenet \
    --batch_size 256 \
    --thresholds 0.05 0.1 0.15 0.2
```

### 6. Benchmark Computation
Measure time and memory consumption for different methods:
```bash
python test_benchmark.py \
    --dataset shapenet \
    --approximator rbf \
    --h_path output/kernel/shapenet/rbf/h.pth \
    --ae_name kl_d512_m512_l8 \
    --ae_path output/ae/kl_d512_m512_l8/checkpoint-199.pth \
    --dm_name kl_d512_m512_l8_d24_edm \
    --dm_path output/dm/kl_d512_m512_l8_d24_edm/checkpoint-499.pth \
    --category 5 \
    --output_dir ./output/benchmark
```

## Batch Experiments

Shell scripts are provided in `scripts/` for running batch experiments:

```bash
# Run all approximators on multiple categories
bash scripts/run_optpath.sh

# Run evaluation on generated results
bash scripts/run_eval.sh

# Run ablation experiments
bash scripts/run_abl_comp.sh
bash scripts/run_sigma_abl.sh
```

## Project Structure

```
GFM-release/
├── gfm/                      # Main package
│   ├── engine/               # Training engines
│   ├── helper/               # Helper functions
│   ├── models/               # Model definitions (AE, Diffusion)
│   ├── path/                 # Geodesic interpolation methods
│   └── util/                 # Utilities (datasets, lr schedulers)
├── scripts/                  # Batch experiment scripts
├── notebooks/                # Jupyter notebooks
│   ├── torus_ood.ipynb           # Toy torus manifold experiments
│   └── torus_metrics.ipynb              # Additional experiments
├── data2fit.py               # Generate samples for kernel fitting
├── pair4vis.py               # Single trajectory visualization
├── optim_path.py             # Distribution-level geodesic sampling
├── dist4eval.py              # Distribution metric evaluation
├── decode.py                 # Latent decoding
├── test_benchmark.py         # Time/memory benchmarking
├── sample_class_cond.py      # Class-conditioned sampling
└── eval.py                   # Autoencoder evaluation
```

## License

This repository is for academic research use only.

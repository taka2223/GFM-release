import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
import mcubes
import torch
import trimesh
from pathlib import Path
from accelerate import Accelerator


from gfm.models import models_ae

def main():
    parser = argparse.ArgumentParser(description='Distributed AE Decoding (No DDP Wrapper)')
    
    # AE 参数
    parser.add_argument('--ae', type=str, default='kl_d512_m512_l8')
    parser.add_argument('--ae-pth', type=str, default='output/ae/kl_d512_m512_l8/checkpoint-199.pth')
    
    # 输入输出
    parser.add_argument('--latents-path', type=str, required=True, help='Path to latents .pth')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    
    # 运行配置
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--density', type=int, default=128, help='Grid density')
    parser.add_argument('--category-id', type=int, default=5, help='Category ID')
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'])

    args = parser.parse_args()

    # 1. 初始化 Accelerator
    # mixed_precision='bf16' 可以在 autocast 时自动生效
    accelerator = Accelerator(mixed_precision=args.precision)
    device = accelerator.device


    latents_path = Path(args.latents_path)
    if args.output_dir is None:
        stem = latents_path.stem
        parent_name = latents_path.parent.name
        save_dir = Path(f"class_cond_obj/{parent_name}/decoded_{stem}")
    else:
        save_dir = Path(args.output_dir)

    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"==========================================")
        print(f"Loading latents from: {latents_path}")
        print(f"Saving meshes to:     {save_dir}")
        print(f"GPUs:                 {accelerator.num_processes}")
        print(f"Precision:            {args.precision}")
        print(f"==========================================")

    accelerator.wait_for_everyone()

    # 3. 加载 Latents (CPU)
    loaded_data = torch.load(latents_path, map_location='cpu', weights_only=False)
    if isinstance(loaded_data, dict):
        if 'latents' in loaded_data: all_latents = loaded_data['latents']
        elif 'data' in loaded_data: all_latents = loaded_data['data']
        else: all_latents = next(iter(loaded_data.values())) 
    else:
        all_latents = loaded_data

    if not isinstance(all_latents, torch.Tensor):
        raise ValueError(f"Could not find Tensor in {args.latents_path}")

    total_samples = all_latents.shape[0]

    # 4. 加载 AE 模型
    # 注意：这里不再使用 accelerator.prepare(ae)
    ae = models_ae.__dict__[args.ae]()
    ae.load_state_dict(torch.load(args.ae_pth, map_location='cpu', weights_only=False)['model'])
    
    # 手动处理设备
    ae.to(device)
    
    # 手动处理权重精度 (为了节省显存)
    if args.precision == 'bf16':
        ae.to(torch.bfloat16)
    elif args.precision == 'fp16':
        ae.to(torch.float16)
    
    ae.eval()

    # 5. 准备 Grid
    density = args.density
    gap = 2. / density
    x = np.linspace(-1, 1, density + 1)
    y = np.linspace(-1, 1, density + 1)
    z = np.linspace(-1, 1, density + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(
        np.stack([xv, yv, zv]).astype(np.float32)
    ).view(3, -1).transpose(0, 1)[None].to(device)
    
    # 同样转换 Grid 的精度以匹配模型
    if args.precision == 'bf16':
        grid = grid.to(torch.bfloat16)
    elif args.precision == 'fp16':
        grid = grid.to(torch.float16)

    # 6. 任务切分
    # 使用 Python 切片手动分配任务
    all_indices = list(range(total_samples))
    my_indices = all_indices[accelerator.process_index :: accelerator.num_processes]

    if accelerator.is_main_process:
        print(f"Workload distribution: ~{len(my_indices)} samples per GPU.")

    # 7. 循环解码
    batch_size = args.batch_size
    num_batches = (len(my_indices) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(my_indices))
            batch_idxs = my_indices[start:end]
            
            if len(batch_idxs) == 0: continue
            
            # A. 搬运数据
            latents_batch = all_latents[batch_idxs].to(device)
            # 确保输入数据精度匹配模型
            if args.precision == 'bf16':
                latents_batch = latents_batch.to(torch.bfloat16)
            elif args.precision == 'fp16':
                latents_batch = latents_batch.to(torch.float16)

            # B. 扩展 Grid
            current_bs = len(batch_idxs)
            grid_batch = grid.expand(current_bs, -1, -1)

            # C. 推理 (直接调用 .decode，因为没有被 DDP 包裹)
            # accelerator.autocast 依然可以用来保证算子兼容性
            with accelerator.autocast():
                logits = ae.decode(latents_batch, grid_batch)
            
            # D. 转回 Float32 + CPU
            logits_cpu = logits.detach().float().cpu()

            # E. Marching Cubes
            for j, global_idx in enumerate(batch_idxs):
                vol_data = logits_cpu[j]
                volume = vol_data.view(density+1, density+1, density+1).permute(1, 0, 2).numpy()
                
                try:
                    verts, faces = mcubes.marching_cubes(volume, 0)
                    if len(verts) == 0: continue
                    
                    verts *= gap
                    verts -= 1
                    
                    m = trimesh.Trimesh(verts, faces)
                    filename = save_dir / f"{args.category_id:02d}-{global_idx:05d}.obj"
                    m.export(str(filename))
                except Exception as e:
                    print(f"Error {global_idx}: {e}")

            if i % 10 == 0 and accelerator.is_main_process:
                print(f"Progress: {i}/{num_batches} batches...")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Done.")

if __name__ == "__main__":
    main()
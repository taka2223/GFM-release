import mcubes
import torch
import trimesh
import numpy as np

def decode_and_save(ae,
                    latent: torch.Tensor,
                    save_path: str,
                    density: int = 128,
                    device: torch.device = torch.device('cuda:0')):
    """Decode latent to mesh and save as OBJ."""
    gap = 2.0 / density
    x = np.linspace(-1, 1, density + 1)
    y = np.linspace(-1, 1, density + 1)
    z = np.linspace(-1, 1, density + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = (torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(
        3, -1).transpose(0, 1)[None].to(device))
    with torch.no_grad():
        logits = ae.decode(latent, grid)
        volume = (logits.view(density + 1, density + 1,
                              density + 1).permute(1, 0, 2).cpu().numpy())
        verts, faces = mcubes.marching_cubes(volume, 0)
        verts = verts * gap - 1

        mesh = trimesh.Trimesh(verts, faces)
        mesh.export(save_path)
        print(f"Saved: {save_path}")
        
def load_model_from_path(
    model_name: str,
    ckpt_path: str,
    registry: dict,
    device: torch.device
) -> torch.nn.Module:
    """Load a single model from checkpoint."""
    model = registry[model_name]()
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model"])
    return model.eval().to(device)

def emd_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: torch.device = torch.device('cuda:0')
) -> torch.Tensor:
    """Compute Earth Mover's Distance (EMD) between two point clouds."""
    return torch.norm(pred - target, p=2, dim=1).mean()

def chamfer_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: torch.device = torch.device('cuda:0')
) -> torch.Tensor:
    """Compute Chamfer Distance between two point clouds."""
    pred = pred.to(device)
    target = target.to(device)
    dist1 = torch.norm(pred.unsqueeze(2) - target.unsqueeze(1), p=2, dim=3)
    dist2 = torch.norm(target.unsqueeze(2) - pred.unsqueeze(1), p=2, dim=3)
    return torch.mean(torch.min(dist1, dim=2)[0]) + torch.mean(torch.min(dist2, dim=2)[0])

def MMD_wrapper(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: torch.device = torch.device('cuda:0')
) -> torch.Tensor:
    """Compute Maximum Mean Discrepancy (MMD) between two point clouds."""
    pass

def COV_wrapper(
    pred: torch.Tensor,
    target: torch.Tensor,
    device: torch.device = torch.device('cuda:0')
) -> torch.Tensor:
    """Compute Covariance Matrix between two point clouds."""
    pass


def normalize_mesh(mesh, scale=0.9999):
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale_ = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale_ * 2 * scale)

    return mesh


def sample_pointcloud(mesh, num=4096):
    points, face_idx = mesh.sample(num, return_index=True)
    points = torch.from_numpy(points.astype(np.float32))
    return points

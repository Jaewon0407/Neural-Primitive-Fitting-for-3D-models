# infer.py

import os
import open3d as o3d
import torch
from torch.utils.data import DataLoader

from config import (
    DATA_ROOT,
    NUM_POINTS,
    NUM_PRIMITIVES,
    DEVICE,
    CHECKPOINT_PATH,
    OUTPUT_DIR,
    NUM_SDF_POINTS,
    SDF_FILENAME,
)
from dataset import ShapePrimitiveDataset
from model import PointNetPrimitiveModel
from utils import sample_points_from_cuboids, cuboids_to_mesh


def infer_and_export_primitives():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use dataset to get surface points and shape names.
    # SDF is loaded but not used here.
    dataset = ShapePrimitiveDataset(
        root=DATA_ROOT,
        num_points=NUM_POINTS,
        num_sdf_points=NUM_SDF_POINTS,
        sdf_filename=SDF_FILENAME,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load trained model
    model = PointNetPrimitiveModel(num_primitives=NUM_PRIMITIVES).to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print(f"Loaded model from {CHECKPOINT_PATH}")
    print(f"Running inference on {len(dataset)} shapes")

    with torch.no_grad():
        for batch in dataloader:
            surf = batch["surf_points"].to(DEVICE).float()  # (1,Ns,3)
            name = batch["name"][0]

            centers, half_sizes = model(surf)               # (1,K,3)
            centers_np = centers.squeeze(0).cpu().numpy()
            half_sizes_np = half_sizes.squeeze(0).cpu().numpy()

            # A) export primitive point cloud
            pred_surf = sample_points_from_cuboids(
                centers,
                half_sizes,
                num_samples_per_shape=NUM_POINTS,
            )
            pred_surf_np = pred_surf.squeeze(0).cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_surf_np)
            out_path = os.path.join(OUTPUT_DIR, f"{name}_primitives.ply")
            o3d.io.write_point_cloud(out_path, pcd)
            print(f"Saved predicted primitive point cloud to {out_path}")

            # B) export cuboid mesh
            mesh = cuboids_to_mesh(centers_np, half_sizes_np)
            if mesh is not None:
                out_mesh_path = os.path.join(OUTPUT_DIR, f"{name}_cuboids_mesh.ply")
                o3d.io.write_triangle_mesh(out_mesh_path, mesh)
                print(f"Saved predicted cuboid mesh to {out_mesh_path}")


if __name__ == "__main__":
    infer_and_export_primitives()

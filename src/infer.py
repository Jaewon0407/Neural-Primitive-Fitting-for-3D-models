# infer.py

import os
import numpy as np
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
)
from dataset import ShapePrimitiveDataset
from model import PointNetPrimitiveModel
from utils import sample_points_from_cuboids, cuboids_to_mesh

def infer_and_export_primitives():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = ShapePrimitiveDataset(DATA_ROOT, num_points=NUM_POINTS)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False)

    model = PointNetPrimitiveModel(num_primitives=NUM_PRIMITIVES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # Extract shape names from dataset file paths
    shape_names = []
    for path in dataset.file_paths:
        # data/dog/surface_points.ply -> dog
        shape_dir = os.path.dirname(path)
        shape_name = os.path.basename(shape_dir)
        shape_names.append(shape_name)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(DEVICE)
            centers, half_sizes = model(batch)

            centers_np = centers[0].cpu().numpy()
            half_sizes_np = half_sizes[0].cpu().numpy()

            pred_points = sample_points_from_cuboids(
                centers, half_sizes, num_samples_per_shape=NUM_POINTS
            )  # (1, N, 3)

            pred_points_np = pred_points[0].cpu().numpy()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_points_np)

            shape_name = shape_names[i] if i < len(shape_names) else f"shape_{i}"
            out_path = os.path.join(OUTPUT_DIR, f"{shape_name}_primitives.ply")
            o3d.io.write_point_cloud(out_path, pcd)
            print(f"Saved predicted primitive point cloud to {out_path}")

            # (B) Export cuboid meshes
            mesh = cuboids_to_mesh(centers_np, half_sizes_np)
            if mesh is not None:
                out_mesh_path = os.path.join(OUTPUT_DIR, f"{shape_name}_cuboids_mesh.ply")
                o3d.io.write_triangle_mesh(out_mesh_path, mesh)
                print(f"Saved predicted cuboid mesh to {out_mesh_path}")

if __name__ == "__main__":
    infer_and_export_primitives()

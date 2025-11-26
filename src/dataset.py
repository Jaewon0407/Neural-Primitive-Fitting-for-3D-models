# dataset.py

import os
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

from config import DATA_ROOT, NUM_POINTS

class ShapePrimitiveDataset(Dataset):
    """
    Loads surface_points.ply from each shape folder:
    data/
        dog/surface_points.ply
        hand/surface_points.ply
        ...
    Returns normalized point cloud of shape (NUM_POINTS, 3)
    """
    def __init__(self, data_root=DATA_ROOT, num_points=NUM_POINTS):
        self.data_root = data_root
        self.num_points = num_points

        # Collect surface_points.ply paths
        self.file_paths = []
        for shape_dir in sorted(os.listdir(data_root)):
            full_dir = os.path.join(data_root, shape_dir)
            if not os.path.isdir(full_dir):
                continue
            ply_path = os.path.join(full_dir, "surface_points.ply")
            if os.path.exists(ply_path):
                self.file_paths.append(ply_path)

        if len(self.file_paths) == 0:
            raise RuntimeError(f"No surface_points.ply files found under {data_root}")

        # Preload normalized point clouds
        self.point_clouds = []
        for path in self.file_paths:
            pts = self.load_ply_points(path)  # (N, 3)
            pts = self.normalize_points(pts)
            self.point_clouds.append(pts)

    def load_ply_points(self, path):
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)  # (N, 3)
        return pts

    def normalize_points(self, pts):
        # Center at origin, scale to fit into unit sphere
        centroid = np.mean(pts, axis=0, keepdims=True)
        pts_centered = pts - centroid
        max_norm = np.max(np.linalg.norm(pts_centered, axis=1))
        if max_norm < 1e-6:
            max_norm = 1.0
        pts_normalized = pts_centered / max_norm
        return pts_normalized

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        pts = self.point_clouds[idx]
        # Randomly sample NUM_POINTS (with replacement if needed)
        if pts.shape[0] >= self.num_points:
            choice = np.random.choice(pts.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(pts.shape[0], self.num_points, replace=True)
        sampled = pts[choice, :]  # (NUM_POINTS, 3)

        sampled = torch.from_numpy(sampled).float()
        return sampled

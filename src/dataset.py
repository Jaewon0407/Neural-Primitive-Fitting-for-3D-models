# dataset.py

import os
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

from config import DATA_ROOT, NUM_POINTS, NUM_SDF_POINTS, SDF_FILENAME


class ShapePrimitiveDataset(Dataset):
    """
    Dataset that returns both surface points and SDF samples for each shape.

    Expected directory structure:
        DATA_ROOT/
            shape1/
                surface_points.ply
                SDF_FILENAME  (e.g. sdf.npz with "points" and "values")
            shape2/
                surface_points.ply
                SDF_FILENAME
            ...

    Each item is a dict with keys:
        - "surf_points": (NUM_POINTS, 3) float32 tensor
        - "sdf_points":  (NUM_SDF_POINTS, 3) float32 tensor
        - "sdf_values":  (NUM_SDF_POINTS,) float32 tensor
        - "name":        string shape name
    """

    def __init__(
        self,
        root: str = DATA_ROOT,
        num_points: int = NUM_POINTS,
        num_sdf_points: int = NUM_SDF_POINTS,
        sdf_filename: str = SDF_FILENAME,
    ):
        super().__init__()
        self.root = root
        self.num_points = num_points
        self.num_sdf_points = num_sdf_points
        self.sdf_filename = sdf_filename

        self.shapes = []  # list of (name, ply_path, sdf_path)
        if not os.path.isdir(self.root):
            raise RuntimeError(f"DATA_ROOT '{self.root}' is not a directory")

        for name in sorted(os.listdir(self.root)):
            shape_dir = os.path.join(self.root, name)
            if not os.path.isdir(shape_dir):
                continue
            ply_path = os.path.join(shape_dir, "surface_points.ply")
            sdf_path = os.path.join(shape_dir, self.sdf_filename)
            if os.path.isfile(ply_path) and os.path.isfile(sdf_path):
                self.shapes.append((name, ply_path, sdf_path))

        if not self.shapes:
            raise RuntimeError(
                f"No shapes with surface_points.ply and {self.sdf_filename} found in {self.root}"
            )

        print(f"Found {len(self.shapes)} shapes in {self.root}")

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, idx: int):
        name, ply_path, sdf_path = self.shapes[idx]

        # 1) surface points from .ply
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points, dtype=np.float32)  # (N_all,3)

        if pts.shape[0] == 0:
            raise RuntimeError(f"Point cloud in {ply_path} is empty")

        # sample NUM_POINTS points
        if pts.shape[0] >= self.num_points:
            choice = np.random.choice(pts.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(pts.shape[0], self.num_points, replace=True)
        surf = pts[choice]  # (NUM_POINTS,3)

        # 2) SDF samples from .npz
        npz = np.load(sdf_path)
        q_points = npz["sdf_points"].astype(np.float32)  # (Nq,3)
        q_values = npz["sdf_values"].astype(np.float32)  # (Nq,)

        if q_points.shape[0] == 0:
            raise RuntimeError(f"SDF points in {sdf_path} are empty")

        if q_points.shape[0] >= self.num_sdf_points:
            choice_q = np.random.choice(
                q_points.shape[0], self.num_sdf_points, replace=False
            )
        else:
            choice_q = np.random.choice(
                q_points.shape[0], self.num_sdf_points, replace=True
            )
        q_points = q_points[choice_q]
        q_values = q_values[choice_q]

        return {
            "surf_points": torch.from_numpy(surf),        # (NUM_POINTS,3)
            "sdf_points": torch.from_numpy(q_points),     # (NUM_SDF_POINTS,3)
            "sdf_values": torch.from_numpy(q_values),     # (NUM_SDF_POINTS,)
            "name": name,
        }

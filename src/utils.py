# utils.py

import torch
import open3d as o3d
import numpy as np

from config import NUM_POINTS


def chamfer_distance(pcd1, pcd2):
    """
    pcd1: (B, N, 3)
    pcd2: (B, M, 3)
    Returns scalar Chamfer distance over the batch.
    """
    B, N, _ = pcd1.shape
    _, M, _ = pcd2.shape

    # (B, N, 1, 3) and (B, 1, M, 3) -> (B, N, M, 3)
    diff = pcd1.unsqueeze(2) - pcd2.unsqueeze(1)
    dist_sq = torch.sum(diff ** 2, dim=3)  # (B, N, M)

    # For each point in pcd1, find nearest in pcd2
    min_dist1, _ = torch.min(dist_sq, dim=2)  # (B, N)
    # For each point in pcd2, find nearest in pcd1
    min_dist2, _ = torch.min(dist_sq, dim=1)  # (B, M)

    loss = torch.mean(min_dist1) + torch.mean(min_dist2)
    return loss


def sample_points_from_cuboids(centers, half_sizes, num_samples_per_shape=NUM_POINTS):
    """
    Differentiable sampling of points inside cuboids.

    centers:    (B, K, 3)
    half_sizes: (B, K, 3)
    num_samples_per_shape: int

    Returns:
        pcd_pred: (B, num_samples_per_shape, 3)
    """
    device = centers.device
    B, K, _ = centers.shape

    samples_per_primitive = max(1, num_samples_per_shape // K)

    # (B, K, samples_per_primitive, 3) random in [-1, 1]
    rand = 2.0 * torch.rand(B, K, samples_per_primitive, 3, device=device) - 1.0

    # scale by half sizes and shift by centers
    centers_exp = centers.unsqueeze(2)         # (B, K, 1, 3)
    sizes_exp = half_sizes.unsqueeze(2)        # (B, K, 1, 3)

    pts = rand * sizes_exp + centers_exp      # (B, K, S, 3)

    pts = pts.view(B, K * samples_per_primitive, 3)  # (B, K*S, 3)

    if K * samples_per_primitive >= num_samples_per_shape:
        # take first num_samples_per_shape points
        pts = pts[:, :num_samples_per_shape, :]
    else:
        # pad by repeating
        repeat_factor = (num_samples_per_shape + K * samples_per_primitive - 1) // (K * samples_per_primitive)
        pts_expanded = pts.repeat(1, repeat_factor, 1)
        pts = pts_expanded[:, :num_samples_per_shape, :]

    return pts

def sample_points_from_cuboids_surface(centers, half_sizes, num_samples_per_shape):
    device = centers.device
    B, K, _ = centers.shape
    samples_per_prim = max(1, num_samples_per_shape // K)

    all_pts = []

    for b in range(B):
        c = centers[b]      # (K, 3)
        s = half_sizes[b]   # (K, 3)

        pts_list = []
        for k in range(K):
            ck = c[k]   # (3,)
            sk = s[k]   # (3,)

            # random faces: 0..5
            face_ids = torch.randint(0, 6, (samples_per_prim,), device=device)

            u = (2 * torch.rand(samples_per_prim, device=device) - 1) * sk[0]
            v = (2 * torch.rand(samples_per_prim, device=device) - 1) * sk[1]
            w = (2 * torch.rand(samples_per_prim, device=device) - 1) * sk[2]

            x = torch.zeros(samples_per_prim, device=device)
            y = torch.zeros(samples_per_prim, device=device)
            z = torch.zeros(samples_per_prim, device=device)

            # assign coordinates based on face
            # +x / -x
            mask = face_ids == 0
            x[mask] = sk[0]
            y[mask] = v[mask]
            z[mask] = w[mask]

            mask = face_ids == 1
            x[mask] = -sk[0]
            y[mask] = v[mask]
            z[mask] = w[mask]

            # +y / -y
            mask = face_ids == 2
            x[mask] = u[mask]
            y[mask] = sk[1]
            z[mask] = w[mask]

            mask = face_ids == 3
            x[mask] = u[mask]
            y[mask] = -sk[1]
            z[mask] = w[mask]

            # +z / -z
            mask = face_ids == 4
            x[mask] = u[mask]
            y[mask] = v[mask]
            z[mask] = sk[2]

            mask = face_ids == 5
            x[mask] = u[mask]
            y[mask] = v[mask]
            z[mask] = -sk[2]

            pts = torch.stack([x, y, z], dim=-1) + ck  # (S, 3)
            pts_list.append(pts)

        pts_cat = torch.cat(pts_list, dim=0)  # (K*S, 3)
        if pts_cat.shape[0] >= num_samples_per_shape:
            pts_cat = pts_cat[:num_samples_per_shape]
        else:
            reps = (num_samples_per_shape + pts_cat.shape[0] - 1) // pts_cat.shape[0]
            pts_cat = pts_cat.repeat(reps, 1)[:num_samples_per_shape]
        all_pts.append(pts_cat)

    all_pts = torch.stack(all_pts, dim=0)  # (B, N, 3)
    return all_pts

def cuboids_to_mesh(centers, half_sizes):
    """
    centers: (K, 3) numpy
    half_sizes: (K, 3) numpy
    Returns a single Open3D TriangleMesh with all cuboids combined.
    """
    all_meshes = []

    for k in range(centers.shape[0]):
        c = centers[k]       # (3,)
        s = half_sizes[k]    # (3,)

        # Box dimensions are full sizes = 2 * half_sizes
        w, h, d = 2 * s[0], 2 * s[1], 2 * s[2]

        # Create box with one corner at (0, 0, 0)
        box = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)

        # Move box so its center is at (cx, cy, cz)
        # Default box center is at (w/2, h/2, d/2)
        box.translate(c - np.array([w / 2, h / 2, d / 2]))

        # Optional: set a uniform color
        box.paint_uniform_color(np.random.rand(3))  # random color per cuboid

        all_meshes.append(box)

    # Combine all cuboids into one mesh
    if len(all_meshes) == 0:
        return None

    combined = all_meshes[0]
    for m in all_meshes[1:]:
        combined += m

    return combined

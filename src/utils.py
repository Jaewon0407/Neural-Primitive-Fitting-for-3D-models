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
    assert pcd1.dim() == 3 and pcd2.dim() == 3
    B, N, _ = pcd1.shape
    B2, M, _ = pcd2.shape
    assert B == B2, "pcd1 and pcd2 must have same batch size"

    # (B, N, 1, 3) - (B, 1, M, 3) -> (B, N, M, 3)
    diff = pcd1.unsqueeze(2) - pcd2.unsqueeze(1)
    dist_sq = torch.sum(diff ** 2, dim=-1)  # (B, N, M)

    min_dist1, _ = torch.min(dist_sq, dim=2)  # (B, N)
    min_dist2, _ = torch.min(dist_sq, dim=1)  # (B, M)

    loss = torch.mean(min_dist1) + torch.mean(min_dist2)
    return loss


def sample_points_from_cuboids(
    centers, half_sizes, num_samples_per_shape=NUM_POINTS, surface_only=True
):
    """
    Sample points on or in predicted cuboids.

    centers:    (B, K, 3)
    half_sizes: (B, K, 3)
    Returns pcd_pred: (B, num_samples_per_shape, 3)
    """
    device = centers.device
    B, K, _ = centers.shape
    samples_per_prim = max(1, num_samples_per_shape // max(1, K))

    all_pts = []
    for b in range(B):
        cb = centers[b]      # (K, 3)
        sb = half_sizes[b]   # (K, 3)
        pts_b = []
        for k in range(K):
            c = cb[k]        # (3,)
            s = sb[k].clamp(min=1e-4)  # avoid degenerate

            if surface_only:
                # sample faces uniformly: pick faces, then (u,v) on face
                n = samples_per_prim
                # choose faces: 0:+x,1:-x,2:+y,3:-y,4:+z,5:-z
                face_ids = torch.randint(0, 6, (n,), device=device)
                u = torch.rand(n, device=device)
                v = torch.rand(n, device=device)
                u = (u - 0.5) * 2.0
                v = (v - 0.5) * 2.0

                x = torch.zeros(n, device=device)
                y = torch.zeros(n, device=device)
                z = torch.zeros(n, device=device)

                # +x / -x
                mask = face_ids == 0
                x[mask] = s[0]
                y[mask] = u[mask] * s[1]
                z[mask] = v[mask] * s[2]

                mask = face_ids == 1
                x[mask] = -s[0]
                y[mask] = u[mask] * s[1]
                z[mask] = v[mask] * s[2]

                # +y / -y
                mask = face_ids == 2
                x[mask] = u[mask] * s[0]
                y[mask] = s[1]
                z[mask] = v[mask] * s[2]

                mask = face_ids == 3
                x[mask] = u[mask] * s[0]
                y[mask] = -s[1]
                z[mask] = v[mask] * s[2]

                # +z / -z
                mask = face_ids == 4
                x[mask] = u[mask] * s[0]
                y[mask] = v[mask] * s[1]
                z[mask] = s[2]

                mask = face_ids == 5
                x[mask] = u[mask] * s[0]
                y[mask] = v[mask] * s[1]
                z[mask] = -s[2]

                pts = torch.stack([x, y, z], dim=-1) + c  # (n,3)
            else:
                # uniform inside box
                n = samples_per_prim
                r = torch.rand(n, 3, device=device) * 2.0 - 1.0  # [-1,1]
                pts = c + r * s

            pts_b.append(pts)
        pts_b = torch.cat(pts_b, dim=0)  # (K * n, 3)

        # adjust count to num_samples_per_shape
        total = pts_b.shape[0]
        if total >= num_samples_per_shape:
            idx = torch.randperm(total, device=device)[:num_samples_per_shape]
            pts_b = pts_b[idx]
        else:
            # repeat to fill
            repeat = (num_samples_per_shape + total - 1) // total
            pts_b = pts_b.repeat(repeat, 1)[:num_samples_per_shape]
        all_pts.append(pts_b)
    pcd_pred = torch.stack(all_pts, dim=0)  # (B, num_samples_per_shape, 3)
    return pcd_pred


def box_sdf(points, centers, half_sizes):
    """
    Compute signed distance from points to axis-aligned boxes.

    points:     (B, N, 3)
    centers:    (B, K, 3)
    half_sizes: (B, K, 3)
    Returns:    (B, N, K) distances
    """
    # expand dims for broadcasting
    p = points.unsqueeze(2)      # (B,N,1,3)
    c = centers.unsqueeze(1)     # (B,1,K,3)
    s = half_sizes.unsqueeze(1)  # (B,1,K,3)

    # |x - c| - s
    q = torch.abs(p - c) - s     # (B,N,K,3)
    d = q.max(dim=-1).values     # (B,N,K)
    return d


def union_sdf(points, centers, half_sizes):
    """
    SDF of union of cuboids: min over individual box SDFs.

    points:     (B, N, 3)
    centers:    (B, K, 3)
    half_sizes: (B, K, 3)
    Returns:    (B, N)
    """
    d_all = box_sdf(points, centers, half_sizes)  # (B,N,K)
    d_union, _ = d_all.min(dim=-1)               # (B,N)
    return d_union


def cuboids_to_mesh(centers, half_sizes):
    """
    Convert cuboid parameters to a single Open3D TriangleMesh.

    centers:    (K, 3) numpy array or torch tensor
    half_sizes: (K, 3) numpy array or torch tensor
    """
    if isinstance(centers, torch.Tensor):
        centers = centers.detach().cpu().numpy()
    if isinstance(half_sizes, torch.Tensor):
        half_sizes = half_sizes.detach().cpu().numpy()

    K = centers.shape[0]
    meshes = []

    for k in range(K):
        c = centers[k]
        s = np.abs(half_sizes[k])
        w, h, d = 2 * s[0], 2 * s[1], 2 * s[2]

        if w <= 0 or h <= 0 or d <= 0:
            continue

        box = o3d.geometry.TriangleMesh.create_box(width=w, height=h, depth=d)
        box.translate(c - np.array([w / 2.0, h / 2.0, d / 2.0]))
        box.paint_uniform_color(np.random.rand(3))
        meshes.append(box)

    if not meshes:
        return None

    mesh = meshes[0]
    for m in meshes[1:]:
        mesh += m
    return mesh

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

def compute_overlap_penalty(centers, half_sizes):
    """
    Differentiable overlap penalty for axis-aligned cuboids.

    centers:    (B, K, 3)
    half_sizes: (B, K, 3)

    Returns:
        overlappenalty: scalar tensor
    """
    B, K,_  = centers.shape
    if K <= 1:
        return torch.tensor(0.0, device=centers.device)

    # Expand to pairwise (B, K, K, 3)
    c_i = centers.unsqueeze(2)
    c_j = centers.unsqueeze(1)
    s_i = half_sizes.unsqueeze(2)
    s_j = half_sizes.unsqueeze(1)

    # Bounding box min and max
    min_i = c_i - s_i    # (B, K, K, 3)
    max_i = c_i + s_i
    min_j = c_j - s_j
    max_j = c_j + s_j

    # Intersection edges
    inter_min = torch.max(min_i, min_j)
    inter_max = torch.min(max_i, max_j)

    # Intersection side lengths (clamped at zero)
    inter_sizes = torch.clamp(inter_max - inter_min, min=0.0)  # (B, K, K, 3)

    # Intersection volume
    inter_vol = torch.prod(inter_sizes, dim=-1)  # (B, K, K)

    # Compute volumes
    vol_i = torch.prod(2 * s_i, dim=-1)  # (B, K, K)
    vol_j = torch.prod(2 * s_j, dim=-1)

    # Union volume
    union = vol_i + vol_j - inter_vol + 1e-8

    # IoU-style overlap
    overlap = inter_vol / union  # (B, K, K)

    # Only keep upper triangular (no self, no double counting)
    mask = torch.triu(torch.ones(K, K, device=centers.device), diagonal=1).bool()
    mask = mask.unsqueeze(0)   # (1, K, K)

    overlap_pairs = overlap[mask]  # all valid pairs across batch

    if overlap_pairs.numel() == 0:
        return torch.tensor(0.0, device=centers.device)

    return overlap_pairs.mean()

##version2
def sample_points_from_cuboids_surface(centers, half_sizes, num_samples_per_shape):
    """
    centers:    (B, K, 3)
    half_sizes: (B, K, 3)
    returns:    (B, num_samples_per_shape, 3)
    """
    device = centers.device
    dtype = centers.dtype
    B, K, _ = centers.shape

    # Number of samples per primitive (like in your original code)
    samples_per_prim = max(1, num_samples_per_shape // K)
    S = samples_per_prim  # just a shorter name

    # (B, K, S) – random face index for each (batch, primitive, sample)
    face_ids = torch.randint(0, 6, (B, K, S), device=device)

    # Half-sizes: (B, K, 1) so we can broadcast to S
    sx = half_sizes[..., 0:1]  # (B, K, 1)
    sy = half_sizes[..., 1:2]  # (B, K, 1)
    sz = half_sizes[..., 2:3]  # (B, K, 1)

    # Random coordinates in [-1, 1], then scaled by the half-sizes
    u = (2 * torch.rand(B, K, S, device=device, dtype=dtype) - 1)
    v = (2 * torch.rand(B, K, S, device=device, dtype=dtype) - 1)
    w = (2 * torch.rand(B, K, S, device=device, dtype=dtype) - 1)

    # Expand half-sizes to match (B, K, S)
    sx_exp = sx.expand(-1, -1, S)
    sy_exp = sy.expand(-1, -1, S)
    sz_exp = sz.expand(-1, -1, S)

    # Scale u, v, w the same way as in your original CPU code
    u = u * sx_exp
    v = v * sy_exp
    w = w * sz_exp

    # Local coordinates on each face
    x = torch.zeros(B, K, S, device=device, dtype=dtype)
    y = torch.zeros(B, K, S, device=device, dtype=dtype)
    z = torch.zeros(B, K, S, device=device, dtype=dtype)

    # Masks for each face
    m0 = face_ids == 0  # +x
    m1 = face_ids == 1  # -x
    m2 = face_ids == 2  # +y
    m3 = face_ids == 3  # -y
    m4 = face_ids == 4  # +z
    m5 = face_ids == 5  # -z

    # +x face
    x[m0] = sx_exp[m0]
    y[m0] = v[m0]
    z[m0] = w[m0]

    # -x face
    x[m1] = -sx_exp[m1]
    y[m1] = v[m1]
    z[m1] = w[m1]

    # +y face
    x[m2] = u[m2]
    y[m2] = sy_exp[m2]
    z[m2] = w[m2]

    # -y face
    x[m3] = u[m3]
    y[m3] = -sy_exp[m3]
    z[m3] = w[m3]

    # +z face
    x[m4] = u[m4]
    y[m4] = v[m4]
    z[m4] = sz_exp[m4]

    # -z face
    x[m5] = u[m5]
    y[m5] = v[m5]
    z[m5] = -sz_exp[m5]

    # Stack into (B, K, S, 3)
    pts = torch.stack([x, y, z], dim=-1)

    # Move to world coordinates: centers is (B, K, 3) → (B, K, 1, 3)
    pts = pts + centers.unsqueeze(2)

    # Flatten per batch: (B, K*S, 3)
    pts = pts.view(B, K * S, 3)

    # Match num_samples_per_shape exactly, like your original code
    total_per_batch = K * S
    if total_per_batch >= num_samples_per_shape:
        pts = pts[:, :num_samples_per_shape, :]
    else:
        reps = (num_samples_per_shape + total_per_batch - 1) // total_per_batch
        pts = pts.repeat(1, reps, 1)[:, :num_samples_per_shape, :]

    return pts


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



def point_in_cuboid(points, centers, half_sizes):
    """
    Vectorized version - Check if points are inside cuboids.

    Args:
        points: (B, N, 3) - input points
        centers: (B, K, 3) - cuboid centers
        half_sizes: (B, K, 3) - cuboid half sizes

    Returns:
        inside: (B, N, K) - boolean tensor indicating if point n is inside cuboid k
    """

    # Expand dimensions for broadcasting
    points_expanded = points.unsqueeze(2)          # (B, N, 1, 3)
    centers_expanded = centers.unsqueeze(1)        # (B, 1, K, 3)
    half_sizes_expanded = half_sizes.unsqueeze(1)  # (B, 1, K, 3)

    # Check if points are within cuboid bounds
    diff = torch.abs(points_expanded - centers_expanded)  # (B, N, K, 3)
    inside = torch.all(diff <= half_sizes_expanded, dim=-1)  # (B, N, K)

    return inside


def compute_coverage_loss(points, centers, half_sizes):
    """
    Vectorized coverage loss - encourages primitives to cover all input points.

    Args:
        points: (B, N, 3) - input points
        centers: (B, K, 3) - predicted cuboid centers
        half_sizes: (B, K, 3) - predicted cuboid half sizes

    Returns:
        coverage_loss: scalar tensor
    """

    inside = point_in_cuboid(points, centers, half_sizes)  # (B, N, K)

    # For each point, check if it's covered by at least one cuboid
    covered = torch.any(inside, dim=-1)  # (B, N)

    # Coverage loss = fraction of points not covered
    coverage_loss = 1.0 - torch.mean(covered.float())

    return coverage_loss



def cuboid_union_sdf(points, centers, half_sizes):
    """
    Approximate signed distance from points to the union of axis-aligned cuboids.

    Args:
        points:     (B, N, 3) query points
        centers:    (B, K, 3) cuboid centers
        half_sizes: (B, K, 3) cuboid half-sizes (positive)

    Returns:
        sdf:        (B, N) signed distance to union of cuboids
                    negative inside union, positive outside
    """
    # Expand for pairwise distances: (B, N, 1, 3) - (B, 1, K, 3) -> (B, N, K, 3)
    p_rel = points.unsqueeze(2) - centers.unsqueeze(1)          # (B, N, K, 3)
    q = torch.abs(p_rel) - half_sizes.unsqueeze(1)              # (B, N, K, 3)

    # Outside distance: distance to nearest point on box when outside
    outside = torch.clamp(q, min=0.0)                           # (B, N, K, 3)
    outside_dist = outside.norm(dim=-1)                         # (B, N, K)

    # Inside term: if all components of q are <= 0, we're inside the box
    # max over axes is <= 0, and we take the maximum as "how deep" we are.
    inside = torch.clamp(q.max(dim=-1).values, max=0.0)         # (B, N, K)

    # Standard box SDF formula
    sdf_per_box = outside_dist + inside                         # (B, N, K)

    # Union of boxes => minimum SDF over K
    sdf_union, _ = sdf_per_box.min(dim=-1)                      # (B, N)

    return sdf_union


def sdf_volume_loss(centers, half_sizes,
                    sdf_points, sdf_values,
                    num_samples=20000):
    """
    Enforce that the union of cuboids matches the ground-truth SDF sign.

    Args:
        centers:     (B, K, 3)
        half_sizes:  (B, K, 3)
        sdf_points:  (P, 3) all precomputed SDF sample points (on DEVICE)
        sdf_values:  (P,)   their signed distances (negative inside shape)
        num_samples: number of SDF points to subsample per iteration

    Returns:
        loss: scalar tensor
    """
    device = centers.device
    B, K, _ = centers.shape

    P = sdf_points.shape[0]
    num_samples = min(num_samples, P)

    # Random subset of SDF samples
    idx = torch.randint(0, P, (num_samples,), device=device)
    pts = sdf_points[idx]           # (M, 3)
    gt = sdf_values[idx]           # (M,)

    # Duplicate points for each batch element
    pts_batch = pts.unsqueeze(0).expand(B, -1, -1)   # (B, M, 3)
    gt_batch = gt.unsqueeze(0).expand(B, -1)         # (B, M)

    # Predicted SDF from cuboids
    pred_sdf = cuboid_union_sdf(pts_batch, centers, half_sizes)  # (B, M)

    # We care mostly about the sign consistency:
    #   - For ground truth inside (gt < 0), we want pred_sdf <= 0
    #   - For ground truth outside (gt > 0), we want pred_sdf >= 0
    inside_mask = gt_batch < 0
    outside_mask = gt_batch > 0

    # Hinge-style penalties
    inside_loss = torch.relu(pred_sdf[inside_mask])**2           # push <= 0
    outside_loss = torch.relu(-pred_sdf[outside_mask])**2        # push >= 0

    # (Optional) extra weight near the true surface if you like:
    # near_surface = (gt_batch.abs() < 0.02)
    # surface_loss = (pred_sdf[near_surface].abs())**2

    loss = inside_loss.mean() + outside_loss.mean()
    return loss

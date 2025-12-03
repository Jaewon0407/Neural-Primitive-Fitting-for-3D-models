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

# def sample_points_from_cuboids_surface(centers, half_sizes, num_samples_per_shape):
#     device = centers.device
#     B, K, _ = centers.shape
#     samples_per_prim = max(1, num_samples_per_shape // K)

#     all_pts = []

#     for b in range(B):
#         c = centers[b]      # (K, 3)
#         s = half_sizes[b]   # (K, 3)

#         pts_list = []
#         for k in range(K):
#             ck = c[k]   # (3,)
#             sk = s[k]   # (3,)

#             # random faces: 0..5
#             face_ids = torch.randint(0, 6, (samples_per_prim,), device=device)

#             u = (2 * torch.rand(samples_per_prim, device=device) - 1) * sk[0]
#             v = (2 * torch.rand(samples_per_prim, device=device) - 1) * sk[1]
#             w = (2 * torch.rand(samples_per_prim, device=device) - 1) * sk[2]

#             x = torch.zeros(samples_per_prim, device=device)
#             y = torch.zeros(samples_per_prim, device=device)
#             z = torch.zeros(samples_per_prim, device=device)

#             # assign coordinates based on face
#             # +x / -x
#             mask = face_ids == 0
#             x[mask] = sk[0]
#             y[mask] = v[mask]
#             z[mask] = w[mask]

#             mask = face_ids == 1
#             x[mask] = -sk[0]
#             y[mask] = v[mask]
#             z[mask] = w[mask]

#             # +y / -y
#             mask = face_ids == 2
#             x[mask] = u[mask]
#             y[mask] = sk[1]
#             z[mask] = w[mask]

#             mask = face_ids == 3
#             x[mask] = u[mask]
#             y[mask] = -sk[1]
#             z[mask] = w[mask]

#             # +z / -z
#             mask = face_ids == 4
#             x[mask] = u[mask]
#             y[mask] = v[mask]
#             z[mask] = sk[2]

#             mask = face_ids == 5
#             x[mask] = u[mask]
#             y[mask] = v[mask]
#             z[mask] = -sk[2]

#             pts = torch.stack([x, y, z], dim=-1) + ck  # (S, 3)
#             pts_list.append(pts)

#         pts_cat = torch.cat(pts_list, dim=0)  # (K*S, 3)
#         if pts_cat.shape[0] >= num_samples_per_shape:
#             pts_cat = pts_cat[:num_samples_per_shape]
#         else:
#             reps = (num_samples_per_shape + pts_cat.shape[0] - 1) // pts_cat.shape[0]
#             pts_cat = pts_cat.repeat(reps, 1)[:num_samples_per_shape]
#         all_pts.append(pts_cat)

#     all_pts = torch.stack(all_pts, dim=0)  # (B, N, 3)
#     return all_pts

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


def cuboid_overlap_loss(centers, half_sizes):
    """
    Penalize overlap between axis-aligned cuboids.

    centers: (B, K, 3)
    half_sizes: (B, K, 3)

    Returns:
        scalar overlap loss over the batch.
    """
    B, K, _ = centers.shape

    # For each pair (i, j) of cuboids in the batch, compute intersection volume.
    c_i = centers.unsqueeze(2)     # (B, K, 1, 3)
    c_j = centers.unsqueeze(1)     # (B, 1, K, 3)
    s_i = half_sizes.unsqueeze(2)  # (B, K, 1, 3)
    s_j = half_sizes.unsqueeze(1)  # (B, 1, K, 3)

    # Compute min and max corners
    min_i = c_i - s_i
    max_i = c_i + s_i
    min_j = c_j - s_j
    max_j = c_j + s_j

    # Overlap along each axis
    overlap_min = torch.maximum(min_i, min_j)   # (B, K, K, 3)
    overlap_max = torch.minimum(max_i, max_j)   # (B, K, K, 3)
    overlap_sizes = torch.clamp(overlap_max - overlap_min, min=0.0)  # (B, K, K, 3)

    overlap_vol = (
        overlap_sizes[..., 0] *
        overlap_sizes[..., 1] *
        overlap_sizes[..., 2]
    )  # (B, K, K)

    # Remove self-overlap along diagonal
    eye = torch.eye(K, device=centers.device).view(1, K, K)
    overlap_vol = overlap_vol * (1.0 - eye)

    # Average overlap volume per pair, then average over batch
    num_pairs = max(1, K * (K - 1))
    loss_per_sample = overlap_vol.sum(dim=(1, 2)) / num_pairs  # (B,)
    return loss_per_sample.mean()
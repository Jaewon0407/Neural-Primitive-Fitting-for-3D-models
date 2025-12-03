# train.py

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    DATA_ROOT,
    NUM_POINTS,
    NUM_PRIMITIVES,
    NUM_SDF_POINTS,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    DEVICE,
    CHECKPOINT_PATH,
    W_CHAMFER,
    W_SDF,
    W_SIZE,
)
from dataset import ShapePrimitiveDataset
from model import PointNetPrimitiveModel
from utils import box_sdf, chamfer_distance, cuboid_overlap_loss, sample_points_from_cuboids, union_sdf


def train_model():
    # Dataset and loader
    dataset = ShapePrimitiveDataset(
        root=DATA_ROOT,
        num_points=NUM_POINTS,
        num_sdf_points=NUM_SDF_POINTS,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # Model and optimiser
    model = PointNetPrimitiveModel(num_primitives=NUM_PRIMITIVES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training on {len(dataset)} shapes")
    model.train()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            surf = batch["surf_points"].to(DEVICE).float()     # (B, Ns, 3)
            sdf_pts = batch["sdf_points"].to(DEVICE).float()   # (B, Nq, 3)
            sdf_val = batch["sdf_values"].to(DEVICE).float()   # (B, Nq)

            optimizer.zero_grad()

            centers, half_sizes = model(surf)  # (B,K,3) each

            # 1) surface Chamfer loss
            pred_surf = sample_points_from_cuboids(
                centers,
                half_sizes,
                num_samples_per_shape=surf.shape[1],
            )
            loss_ch = chamfer_distance(surf, pred_surf)

            # 2) SDF loss at query points
            pred_sdf = union_sdf(sdf_pts, centers, half_sizes)  # (B,Nq)
            gt_sdf   = sdf_val                                  # (B,Nq)
            loss_sdf = F.mse_loss(pred_sdf, gt_sdf)


            margin = 0.01  # small safety margin around zero, same order as eps

            inside_mask  = gt_sdf < -margin
            outside_mask = gt_sdf >  margin

            # Overflow: GT outside (>0) but pred says inside (<0)
            # want pred_sdf >= +margin there
            overflow = torch.relu(margin - pred_sdf[outside_mask])  # (..,)
            loss_overflow = overflow.mean() if overflow.numel() > 0 else 0.0

            # Underfill: GT inside (<0) but pred says outside (>0)
            # want pred_sdf <= -margin there
            underfill = torch.relu(pred_sdf[inside_mask] + margin)
            loss_underfill = underfill.mean() if underfill.numel() > 0 else 0.0

            # d_all: (B, Nq, K) from box_sdf
            d_all = box_sdf(sdf_pts, centers, half_sizes)
            pred_sdf, owners = d_all.min(dim=-1)  # (B,Nq), (B,Nq) indices of responsible cube

            # per-primitive responsibility: fraction of points where this cube is the closest
            B, Nq = owners.shape
            K = centers.shape[1]
            resp = []
            for k in range(K):
                resp_k = (owners == k).float().mean()  # fraction of samples "owned" by cube k
                resp.append(resp_k)
            resp = torch.stack(resp)  # (K,)
            
            # encourage more even responsibility; large resp_k means cube k covers too much
            dominance_reg = (resp ** 2).sum()  # big if one cube dominates

            loss_overlap = cuboid_overlap_loss(centers, half_sizes)

            # 3) size regulariser (penalise very large boxes)
            # sum of squared half-sizes per primitive, averaged over batch
            size_reg = (half_sizes ** 2).sum(dim=-1).mean()

            loss = W_CHAMFER * loss_ch + 0.01 * loss_overlap +W_SDF * loss_sdf + 10 * loss_overflow + 0.001 * loss_underfill + W_SIZE * size_reg + 0.1 * dominance_reg

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}  "
            f"loss={avg_loss:.6f}  "
            f"(ch={loss_ch.item():.6f}, sdf={loss_sdf.item():.6f}, reg={size_reg.item():.6f})"
        )

    # Save model
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Training finished, model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train_model()

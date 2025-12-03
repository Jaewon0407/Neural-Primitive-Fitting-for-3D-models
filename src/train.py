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
from utils import chamfer_distance, sample_points_from_cuboids, union_sdf


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
            loss_sdf = F.mse_loss(pred_sdf, sdf_val)

            # 3) size regulariser (penalise very large boxes)
            # sum of squared half-sizes per primitive, averaged over batch
            size_reg = (half_sizes ** 2).sum(dim=-1).mean()

            loss = W_CHAMFER * loss_ch + W_SDF * loss_sdf + W_SIZE * size_reg

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

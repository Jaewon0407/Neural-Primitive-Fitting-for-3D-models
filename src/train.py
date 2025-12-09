# train.py

import os
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from config import (
    DATA_ROOT,
    NUM_POINTS,
    NUM_PRIMITIVES,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    DEVICE,
    CHECKPOINT_PATH,
)
from dataset import ShapePrimitiveDataset
from model import PointNetPrimitiveModel
from utils import chamfer_distance, compute_coverage_loss, compute_overlap_penalty, sample_points_from_cuboids_surface, sdf_volume_loss
lambda_cov = 1      # example, keep small if coverage is already good
lambda_sdf = 1      # start small; you can tune this


def train_model():
    # Dataset and dataloader
    dataset = ShapePrimitiveDataset(DATA_ROOT, num_points=NUM_POINTS)
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True)
    
     # Load SDF data (single shape case)
    # Automatically find the first subfolder inside data/
    subdirs = [d for d in os.listdir(DATA_ROOT)
            if os.path.isdir(os.path.join(DATA_ROOT, d))]

    if not subdirs:
        raise FileNotFoundError(f"No subfolders found inside {DATA_ROOT}")

    # Assume there is only one folder (e.g. "hand", "chair", etc.)
    shape_folder = subdirs[0]

    # Construct the path to the npz file
    sdf_path = os.path.join(DATA_ROOT, shape_folder, "voxel_and_sdf.npz")

    if not os.path.exists(sdf_path):
        raise FileNotFoundError(f"Could not find voxel_and_sdf.npz in {sdf_path}")
    sdf_npz = np.load(sdf_path)
    sdf_points_np = sdf_npz["sdf_points"].astype(np.float32)
    sdf_values_np = sdf_npz["sdf_values"].astype(np.float32)

    sdf_points = torch.from_numpy(sdf_points_np).to(DEVICE)      # (P, 3)
    sdf_values = torch.from_numpy(sdf_values_np).to(DEVICE)      # (P,)

    # Model
    model = PointNetPrimitiveModel(num_primitives=NUM_PRIMITIVES).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # batch: (B, N, 3)
            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            centers, half_sizes = model(batch)  # (B, K, 3), (B, K, 3)

            # # Sample predicted points from cuboids
            pred_points = sample_points_from_cuboids_surface(
                centers, half_sizes, num_samples_per_shape=NUM_POINTS
            )  # (B, N, 3)


            # Chamfer distance between predicted and ground truth
            loss_recon = chamfer_distance(pred_points, batch)

            overlap = compute_overlap_penalty(centers, half_sizes)

            coverage_loss = compute_coverage_loss(pred_points, centers, half_sizes)

            # New volume / SDF loss
            sdf_loss_val = sdf_volume_loss(
                centers, half_sizes, sdf_points, sdf_values, num_samples=20000
            )

            # Simple regularization to avoid very large cuboids
            size_reg = torch.mean(half_sizes**2)

            loss = loss_recon + 0.0001 * size_reg + 0.5 * overlap + 0.2 * coverage_loss + lambda_sdf * sdf_loss_val

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}  Loss: {avg_loss:.6f}")

    # Save model
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Training finished, model saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train_model()

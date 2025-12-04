# train.py

import os
import torch
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
from utils import chamfer_distance, compute_coverage_loss, compute_overlap_penalty, sample_points_from_cuboids_surface as sample_points_from_cuboids


def train_model():
    # Dataset and dataloader
    dataset = ShapePrimitiveDataset(DATA_ROOT, num_points=NUM_POINTS)
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True)

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
            pred_points = sample_points_from_cuboids(
                centers, half_sizes, num_samples_per_shape=NUM_POINTS
            )  # (B, N, 3)


            # Chamfer distance between predicted and ground truth
            loss_recon = chamfer_distance(pred_points, batch)

            overlap = compute_overlap_penalty(centers, half_sizes)

            coverage_loss = compute_coverage_loss(pred_points, centers, half_sizes)

            # Simple regularization to avoid very large cuboids
            size_reg = torch.mean(half_sizes**2)

            loss = loss_recon + 0.00001 * size_reg + 0.0001 * overlap + 0.2 * coverage_loss

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

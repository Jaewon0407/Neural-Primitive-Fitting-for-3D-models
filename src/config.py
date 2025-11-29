# config.py

import torch
import random
import numpy as np

# Paths
DATA_ROOT = "data"           # folder with dog, hand, pot, rod, sofa
CHECKPOINT_PATH = "checkpoints/pointnet_primitives.pth"
OUTPUT_DIR = "outputs"

# Data
NUM_POINTS = 4096            # number of points sampled per shape

# Model
NUM_PRIMITIVES = 64           # number of cuboids per shape
FEATURE_DIM = 256            # PointNet global feature size

# Training
BATCH_SIZE = 1
NUM_EPOCHS = 2500
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# config.py

import torch
import random
import numpy as np

# Paths
DATA_ROOT = "/Users/ivanpostolov/Documents/GitHub/data"           # folder with per-shape subdirs
CHECKPOINT_PATH = "checkpoints/pointnet_primitives.pth"
OUTPUT_DIR = "outputs"

# Data
NUM_POINTS = 4096            # surface points per shape
NUM_SDF_POINTS = 8192        # SDF query samples per shape
SDF_FILENAME = "sdf.npz"     # file inside each shape dir with "points" and "values"

# Model
FEATURE_DIM = 512
NUM_PRIMITIVES = 32          # number of cuboids per shape

# Training
BATCH_SIZE = 1
NUM_EPOCHS = 250
LEARNING_RATE = 5e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss weights
W_CHAMFER = 0.1
W_SDF = 1
W_SIZE = 0.01 # size regularizer on half_sizes

# Random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

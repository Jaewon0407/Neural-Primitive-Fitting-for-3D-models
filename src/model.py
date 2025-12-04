import torch
import torch.nn as nn

from config import FEATURE_DIM, NUM_PRIMITIVES


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, feature_dim=FEATURE_DIM):
        super().__init__()
        self.mlp1 = nn.Linear(input_dim, 128)
        self.mlp2 = nn.Linear(128, 256)
        self.mlp3 = nn.Linear(256, 512)
        self.mlp4 = nn.Linear(512, 1024)
        self.mlp5 = nn.Linear(1024, feature_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (B, N, 3)
        returns: (B, feature_dim)
        """
        x = self.relu(self.mlp1(x))   # (B, N, 64)
        x = self.relu(self.mlp2(x))   # (B, N, 128)
        x = self.relu(self.mlp3(x))   # (B, N, 256)
        x = self.relu(self.mlp4(x))   # (B, N, 512)
        x = self.relu(self.mlp5(x))

        # Symmetric max pooling over points
        x, _ = torch.max(x, dim=1)    # (B, feature_dim)
        return x


class PrimitiveDecoder(nn.Module):
    """
    Predicts NUM_PRIMITIVES axis aligned cuboids.

    Each cuboid has:
        center: (cx, cy, cz)
        half sizes: (sx, sy, sz), enforced positive with softplus

    Total parameters per primitive: 6
    """
    def __init__(self, feature_dim=FEATURE_DIM, num_primitives=NUM_PRIMITIVES):
        super().__init__()
        self.num_primitives = num_primitives
        self.param_dim_per_prim = 6
        out_dim = num_primitives * self.param_dim_per_prim

        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, out_dim)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, global_feat):
        """
        global_feat: (B, feature_dim)
        returns:
            centers: (B, K, 3)
            half_sizes: (B, K, 3)
        """
        x = self.relu(self.fc1(global_feat))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # (B, K*6)

        B = x.shape[0]
        x = x.view(B, self.num_primitives, self.param_dim_per_prim)

        centers = x[:, :, 0:3]                  # (B, K, 3)
        raw_sizes = x[:, :, 3:6]                # (B, K, 3)
        # half_sizes = self.softplus(raw_sizes) + 0.02  # avoid tiny sizes

        base_size = 0.01
        half_sizes = base_size * torch.exp(torch.clamp(raw_sizes, -3, 3))

        # min_size = 0.005
        # max_size = 0.5

        # half_sizes = min_size + max_size * torch.sigmoid(raw_sizes)

        return centers, half_sizes


class PointNetPrimitiveModel(nn.Module):
    def __init__(self, num_primitives=NUM_PRIMITIVES):
        super().__init__()
        self.encoder = PointNetEncoder(input_dim=3, feature_dim=FEATURE_DIM)
        self.decoder = PrimitiveDecoder(feature_dim=FEATURE_DIM,
                                        num_primitives=num_primitives)

    def forward(self, points):
        """
        points: (B, N, 3)
        returns:
            centers: (B, K, 3)
            half_sizes: (B, K, 3)
        """
        global_feat = self.encoder(points)
        centers, half_sizes = self.decoder(global_feat)
        return centers, half_sizes
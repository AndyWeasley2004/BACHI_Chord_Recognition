from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SameConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel_size
        pad_left = (k - 1) // 2
        pad_right = k // 2
        x = F.pad(x, (pad_left, pad_right))
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, input_features: int, blocks: int = 6):
        super().__init__()
        layers = []
        in_channels = input_features
        for i in range(blocks):
            filters = 2 ** (blocks - 1 - i)
            kernel = 2 ** i
            conv = SameConv1d(in_channels, filters, kernel_size=kernel)
            bn = nn.BatchNorm1d(filters)
            layers.append(nn.Sequential(conv, bn, nn.ReLU(inplace=True)))
            in_channels = in_channels + filters
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> use channels-last then transpose to (B, F, T)
        x = x.transpose(1, 2)
        out = x
        for layer in self.blocks:
            h = layer(out)
            # Time dimension remains constant due to SameConv1d
            out = torch.cat([out, h], dim=1)
        out = out.transpose(1, 2)
        return out


class AugmentedNetPT(nn.Module):
    def __init__(self, input_feature_dims: List[int], output_class_dims: List[int], blocks: int = 6):
        super().__init__()
        self.input_branches = nn.ModuleList(
            [ConvBlock(in_dim, blocks=blocks) for in_dim in input_feature_dims]
        )
        total_features = sum(in_dim + sum(2 ** (blocks - 1 - i) for i in range(blocks)) for in_dim in input_feature_dims)
        self.fc1 = nn.Linear(total_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.gru1 = nn.GRU(32, 30, batch_first=True, bidirectional=True)
        self.bn3 = nn.BatchNorm1d(60)
        self.gru2 = nn.GRU(60, 30, batch_first=True, bidirectional=True)
        self.bn4 = nn.BatchNorm1d(60)
        self.heads = nn.ModuleList([nn.Linear(60, c) for c in output_class_dims])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # inputs: list of (B, T, F)
        branches = [branch(x) for branch, x in zip(self.input_branches, inputs)]
        h = torch.cat(branches, dim=2)
        B, T, F = h.shape
        h = self.fc1(h)
        h = self.bn1(h.reshape(B * T, -1)).reshape(B, T, -1)
        h = torch.relu(h)
        h = self.fc2(h)
        h = self.bn2(h.reshape(B * T, -1)).reshape(B, T, -1)
        h = torch.relu(h)
        h, _ = self.gru1(h)
        h = self.bn3(h.reshape(B * T, -1)).reshape(B, T, -1)
        h, _ = self.gru2(h)
        h = self.bn4(h.reshape(B * T, -1)).reshape(B, T, -1)
        outputs = [head(h) for head in self.heads]
        return outputs
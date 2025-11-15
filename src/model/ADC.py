import torch
import torch.nn as nn
from src.config.config import Config


class ADC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t_minus_1, b_values):
        B, N, W, H = y_prime_t_minus_1.shape

        # Take log of signal
        logS = torch.log(y_prime_t_minus_1)  # (B, N, W, H)

        # Design matrices: X = [1, -b]  shape (B, N, 2)
        ones = torch.ones_like(b_values)
        X = torch.stack([ones, -b_values], dim=-1)  # (B, N, 2)

        # Compute pseudoinverse for each subject in batch
        # torch.linalg.pinv works batched: (B, 2, N)
        X_pinv = torch.linalg.pinv(X)  # (B, 2, N)

        # Flatten spatial dims for linear solve
        logS_flat = logS.permute(0, 2, 3, 1).reshape(B, W * H, N)  # (B, HW, N)

        # Batched least squares: (B, HW, 2)
        betas = torch.bmm(logS_flat, X_pinv.transpose(1, 2))  # (B, HW, 2)

        # Reshape back to maps
        lnS0_map = betas[..., 0].view(B, W, H)
        ADC_map = betas[..., 1].view(B, W, H)
        S0_map = torch.exp(lnS0_map)

        return S0_map, ADC_map

import torch
import torch.nn as nn
from src.config.config import Config


class ADC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t_minus_1, b_values):
        batch_size, n_bvals, height, width = y_prime_t_minus_1.shape

        # Use the first sample's b-values for the design matrix
        X = torch.stack(
            [
                torch.ones(n_bvals, device=y_prime_t_minus_1.device),
                b_values[0],
            ],
            dim=1,
        )  # shape (n_bvals, 2)

        # SVD for pseudoinverse
        U, S_diag, Vh = torch.linalg.svd(X, full_matrices=False)
        X_pinv = Vh.T @ torch.diag(1.0 / S_diag) @ U.T  # shape (2, n_bvals)

        eps = 1e-8
        logS = torch.log(
            torch.clamp(y_prime_t_minus_1, min=eps)
        )  # shape: (B, n_bvals, H, W)
        logS_flat = logS.permute(0, 2, 3, 1).reshape(-1, n_bvals)  # (B*H*W, n_bvals)

        # Solve linear system using pseudoinverse
        betas = (X_pinv @ logS_flat.T).T  # shape: (B*H*W, 2)
        lnS0_map = betas[:, 0].view(batch_size, height, width)
        ADC_map = -betas[:, 1].view(batch_size, height, width)  # negative slope = ADC

        S0_map = torch.exp(lnS0_map)

        return S0_map, ADC_map

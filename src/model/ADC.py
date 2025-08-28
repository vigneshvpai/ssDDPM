import torch
import torch.nn as nn
from src.config.config import Config


class ADC(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_b_values = Config.ADC_CONFIG["n_bvals"]
        self.adc_type = Config.ADC_CONFIG["adc_type"]
        self.num_dirs = Config.ADC_CONFIG["num_dirs"]

    def forward(self, y_prime_t_minus_1, b_values):
        if self.adc_type == "avg":
            batch_size, total_slices, height, width = y_prime_t_minus_1.shape
            n_slices = total_slices // self.num_b_values

            y = y_prime_t_minus_1.view(
                batch_size, n_slices, self.num_b_values, height, width
            )
            b = b_values.view(batch_size, n_slices, self.num_b_values)

            X = torch.stack(
                [torch.ones(self.num_b_values, device=y.device), b[0, 0]], dim=1
            )  # shape (n_bvals, 2)

            # SVD for pseudoinverse
            U, S_diag, Vh = torch.linalg.svd(X, full_matrices=False)
            X_pinv = Vh.T @ torch.diag(1.0 / S_diag) @ U.T  # shape (2, n_bvals)

            eps = 1e-6
            logS = torch.log(torch.clamp(y, min=eps))  # shape: (B, slices, bvals, H, W)

            logS_flat = logS.permute(0, 1, 3, 4, 2).reshape(
                -1, self.num_b_values
            )  # (B*slices*H*W, n_bvals)

            # Solve linear system using pseudoinverse
            betas = (X_pinv @ logS_flat.T).T  # shape: (B*slices*H*W, 2)

            lnS0_map = betas[:, 0].view(batch_size, n_slices, height, width)
            S0_map = torch.exp(lnS0_map)
            ADC_map = -betas[:, 1].view(
                batch_size, n_slices, height, width
            )  # negative slope = ADC

            return S0_map, ADC_map
        elif self.adc_type == "dir":
            raise NotImplementedError("Directional ADC not implemented")

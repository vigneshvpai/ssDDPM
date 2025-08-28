import torch
import torch.nn as nn


class ADC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t_minus_1, b_values):
        print(y_prime_t_minus_1.shape)
        print(b_values.shape)

        ones = torch.ones_like(b_values)  # Shape: [2, 625]
        A = torch.stack([b_values, ones], dim=-1)  # Shape: [2, 625, 2]
        print(A)

        S0_hat = torch.ones((2, 625, 144, 128))
        D_hat = torch.ones((2, 625))

        return S0_hat, D_hat

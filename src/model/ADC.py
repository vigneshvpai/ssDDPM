import torch.nn as nn


class ADC(nn.Module):
    def __init__(self, b_values):
        super().__init__()

    def forward(self, y_prime_t_minus_1, b_values):
        pass

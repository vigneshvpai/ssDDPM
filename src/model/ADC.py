import torch.nn as nn


class ADC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_prime_t_minus_1, b_values):
        print(y_prime_t_minus_1.shape)
        print(b_values.shape)
        return 1, 1

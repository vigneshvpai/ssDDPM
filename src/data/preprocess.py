import os
import torch
from src.config.config import Config


class Preprocess:
    def __init__(self, pt_data_root=Config.PT_DATA_ROOT):
        self.pt_data_root = pt_data_root

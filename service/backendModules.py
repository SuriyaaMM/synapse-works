from config import logging
from typing import Optional, Tuple

import torch
from torch import nn 

class Cat(nn.Module):
    def __init__(self, dimension: Optional[int]):
        super().__init__()
        self.dimension = dimension

    def forward(self, *tensors):
        return torch.cat(tensors, self.dimension)
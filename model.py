
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self) -> None:
        super(Model, self).__init__()
        
        self.layers = nn.Sequential(
			Linear(param1=value1, param2=value1)
			ReLU(param1=value1, param2=value1)
        )
    
    
    def forward(self, data):
        return self.layers(data)

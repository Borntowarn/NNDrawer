
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self) -> None:
        super(Model, self).__init__()
        
        


    def forward(self, data):
        data = F.hardshrink(data, )
        return data

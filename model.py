
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    
    def __init__(self) -> None:
        super(Model, self).__init__()
        
        self.layers = nn.Sequential(
			Linear(Doc=None, Args={'self': {'Type': None, 'Default': None}, 'in_features': {'Type': "<class 'int'>", 'Default': None}, 'out_features': {'Type': "<class 'int'>", 'Default': None}, 'bias': {'Type': "<class 'bool'>", 'Default': 'True'}, 'device': {'Type': None, 'Default': 'None'}, 'dtype': {'Type': None, 'Default': 'None'}})
			ReLU(Doc=None, Args={'self': {'Type': None, 'Default': None}, 'inplace': {'Type': "<class 'bool'>", 'Default': 'False'}})
        )
    
    
    def forward(self, data):
        data = self.layers(data)
        return data

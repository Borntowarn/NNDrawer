
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
	
	def __init__(self) -> None:
		super(Model, self).__init__()
		
		self.layer_1 = nn.Sequential(
			nn.Linear(in_features=64, out_features=32),
			nn.Linear(in_features=32, out_features=1)
		)
    		
		self.layer_2 = nn.Sequential(
			nn.ReLU()
		)
    		
		self.layer_3 = nn.Sequential(
			nn.Linear(in_features=64, out_features=32),
			nn.Linear(in_features=32, out_features=1)
		)
    		
		self.layer_4 = nn.Sequential(
			nn.Dropout()
		)
    

	def forward(self, data):
		# Custom node "Base_block"
		data = self.layer_1(data)
		
		# Custom node "Func_block"
		data = F.adaptive_avg_pool2d(data)
		data = self.layer_2(data)
		data = self.layer_3(data)
		data = self.layer_4(data)
		data = F.celu(data)
		
		return data

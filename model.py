
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
	
	def __init__(self) -> None:
		super(Model, self).__init__()
		
		self.layer_1 = nn.Sequential(
			nn.Linear(in_features=64, out_features=64),
			nn.Linear(in_features=64, out_features=64)
		)
    		
		self.layer_2 = nn.Sequential(
			nn.Linear(in_features=64, out_features=64),
			nn.Linear(in_features=64, out_features=64)
		)
    		
		self.layer_3 = nn.Sequential(
			nn.ReLU(inplace=True),
			nn.Dropout()
		)
    		
		self.layer_4 = nn.Sequential(
			nn.Tanh()
		)
    

	def forward(self, data):
		# Custom node "Linear"
		data = self.layer_1(data)
		
		# Custom node "ReInRec"
		data = F.avg_pool2d(data, )
		data = self.layer_2(data)
		data = self.layer_3(data)
		data = F.ctc_loss(data, )
		data = self.layer_4(data)
		
		return data


import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
	
	def __init__(self) -> None:
		super(Model, self).__init__()
		
		
		self.layer_1 = nn.Sequential(
			nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3),
			nn.Sigmoid()
		)
    		
		self.layer_2 = nn.Sequential(
			nn.RNN()
		)
    


	def forward(self, data):
		data = F.batch_norm(data, )
		data = self.layer_1(data)
		data = F.adaptive_avg_pool1d(data, )
		data = self.layer_2(data)
		return data

import torch
import torch.nn as nn
import torch.nn.functional as F

class MDN(nn.Module) :
	def __init__(self, input_size, num_mixtures) :
		super(MDN, self).__init__()
		self.input_size = input_size
		self.num_mixtures = num_mixtures
		self.pi_layer = nn.Linear(input_size, num_mixtures)
		self.sigma_layer = nn.Linear(input_size, num_mixtures)
		self.mu_layer = nn.Linear(input_size, num_mixtures)

	def forward(self, input) :
		pi = F.softmax(self.pi_layer(input))
		sigma = torch.exp(self.sigma_layer(input))
		mu = self.mu_layer(input)
		return [pi, sigma, mu]

import torch
from torch.nn.parallel.scatter_gather import scatter, gather
from torch.nn.parallel.replicate import replicate
import os
import numpy as np


class GraphDataParallel(torch.nn.Module):
    def __init__(self):
        super(GraphDataParallel, self).__init__()
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            pass
            # raise RuntimeError('No GPUs found, please use MACECalculator instead.')
        devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
        print('Using devices:', devices)
        
    def forward(self, x):
        raise NotImplementedError
    
    def scatter(self, x, devices):
        return scatter(x, devices)
    
    def gather(self, x, device):
        return gather(x, device)
    
    def replicate(self, x, devices):
        return replicate(x, devices)
    
    def partition(self, nodes_features, edge_features, edge_index, devices):
        raise NotImplementedError
    
    def parallel_apply(self, replicas, devices):
        raise NotImplementedError

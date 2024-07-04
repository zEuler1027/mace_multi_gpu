import torch
from torch.nn.parallel.scatter_gather import scatter, gather
from torch.nn.parallel.replicate import replicate
import numpy as np
from metispy import metis
import networkx as nx


class GraphDataParallel(torch.nn.Module):
    def __init__(self):
        super(GraphDataParallel, self).__init__()
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise RuntimeError('No distributed GPUs found, please use MACECalculator instead.')
        devices = [torch.device(f'cuda:{i}') for i in range(self.num_gpus)]
        print('Using devices:', devices)
        # self.num_gpus = 3
        
    def forward(self, x):
        raise NotImplementedError
    
    def check_memory(self):
        for i in range(self.num_gpus):
            print(f'GPU {i} memory:', torch.cuda.memory_allocated(i) / 1024 ** 3, 'GB')
    
    def get_scatter_nodes_mask(self, num_nodes):
        # Get the number of nodes per GPU
        nodes_per_gpu = num_nodes // self.num_gpus
        node_idx = torch.repeat_interleave(torch.arange(self.num_gpus), nodes_per_gpu)
        if node_idx.size(0) < num_nodes:
            nodes_mask = torch.cat(
                [node_idx, torch.ones(num_nodes - node_idx.size(0), dtype=torch.long) * self.num_gpus - 1]
            )
        return nodes_mask
    
    def get_scatter_edges_index(self, edge_index):
        edges_mask = self.get_scatter_edges_mask(edge_index)
        return [edge_index[:, edges_mask[i]] for i in range(self.num_gpus)]
        
    def get_scatter_edges_mask(self, edge_index):
        parts = self.get_metis_partition(edge_index)
        node_mask = torch.tensor(parts, dtype=torch.long)
        edges_mask = [node_mask[edge_index[1]] == i for i in range(self.num_gpus)]
        return edges_mask
    
    def get_metis_partition(self, edge_index):
        edges = edge_index.t().tolist()
        G = nx.Graph()
        G.add_edges_from(edges)
        _, parts = metis.part_graph(G, self.num_gpus)
        return parts
        
    def scatter(self, x, devices):
        return scatter(x, devices)
    
    def gather(self, x, device):
        return gather(x, device)
    
    def replicate(self, x, devices):
        return replicate(x, devices)
    
    def parallel_apply(self, replicas, devices):
        raise NotImplementedError

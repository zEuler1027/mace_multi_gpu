import torch
from typing import Dict, Optional
from pprint import pprint
from mace.calculators.utils.mace_utils import get_symmetric_displacement, get_edge_vectors_and_lengths, get_outputs
from mace.tools.scatter import scatter_sum
from torch.nn.parallel.replicate import replicate

class DPMACE(torch.nn.Module):
    def __init__(self, dp: torch.nn.Module):
        super(DPMACE, self).__init__()
        self.dp = dp
        
    def forward(
        self,
        model,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        
        # setup
        data["positions"].requires_grad = True
        data["node_attrs"].reauires_grad = True
        num_graphs = data["ptr"].numel() - 1
        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=data["positions"].dtype,
            device=data["positions"].device,
        )
        if compute_virials or compute_stress or compute_displacement:
            (
                data['positions'],
                data['shifts'],
                displacement,
            ) = get_symmetric_displacement(
                positions=data['positions'],
                unit_shifts=data['shifts'],
                cell=data["cell"],
                edge_index=data['edge_index'],
                num_graphs=num_graphs,
                batch=data['batch'],
            )
            
        # Atomic energies
        node_e0 = model.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum(
            src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs
        ) # [n_graphs, ]
        
        # Embedding
        node_feats = model.node_embedding(data["node_attrs"])
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        edge_attrs = model.spherical_harmonics(vectors)
        edge_feats = model.radial_embedding(
            lengths, data['node_attrs'], data['edge_index'], model.atomic_numbers
        )
        pair_node_energy = torch.zeros_like(node_e0)

        # Interaction 
        node_es_list = [pair_node_energy]
        node_feats_list = []

        # graph data parallel
        num_gpus = self.dp.num_gpus
        interactions = replicate(model.interactions, devices=self.dp.devices)
        pprint(interactions)
        for interaction, product, readout in zip(
            model.interactions, model.products, model.readouts
        ):
            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )

            node_feats = product(
                node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            node_feats_list.append(node_feats)
            node_es_list.append(readout(node_feats).squeeze(-1))  # {[n_nodes, ], }
        # concat
        node_feats_out = torch.cat(node_feats_list, dim=-1)
        
        # sum over interactions
        node_inter_es = torch.sum(
            torch.stack(node_es_list, dim=0), dim=0
        )
        node_inter_es = model.scale_shift(node_inter_es)
        
        #sum over nodes in graph
        inter_e = scatter_sum(
            src=node_inter_es, index=data["batch"], dim=-1, dim_size=num_graphs
        )
        
        # add E_0 and (scaled) interaction energy
        totol_energy = e0 + inter_e
        node_energy = node_e0 + node_inter_es
        forces, virials, stress, hessian = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
        )
        output = {
            "energy": totol_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }
        
        return output
    
    

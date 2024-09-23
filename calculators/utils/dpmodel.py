import torch
from typing import Dict, Optional
from pprint import pprint
from mace.calculators.utils.mace_utils import (
    get_symmetric_displacement, 
    get_edge_vectors_and_lengths, 
    get_outputs, 
    MACEModule
)
from mace.tools.scatter import scatter_sum


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
        # clear cache
        torch.cuda.empty_cache()
        # setup
        data["positions"].requires_grad_(True)
        data["node_attrs"].requires_grad_(True)
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
                unit_shifts=data['unit_shifts'],
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

        # -----------------------------------–––---------------------------------------------
        # graph data parallel by songze, repicate the module
        interactions = [
            self.dp.replicate(MACEModule(model.interactions[i]), devices=self.dp.devices) 
            for i in range(len(model.interactions))
        ]

        products = [
            self.dp.replicate(MACEModule(model.products[i]), devices=self.dp.devices) 
            for i in  range(len(model.products))
        ]
        
        readouts = [
            self.dp.replicate(model.readouts[i], devices=self.dp.devices)
            for i in range(len(model.readouts))
        ]
        
        # broadcast node_attrs
        node_attrs = self.dp.broadcast(data["node_attrs"])
        
        # scatter edge_attrs and edge_feats (define by metis graph partition)
        edges_mask = self.dp.get_scatter_edges_mask(data['edge_index'])
        edge_attrs = self.dp.scatter_edges_feats(edge_attrs, edges_mask)
        edge_feats = self.dp.scatter_edges_feats(edge_feats, edges_mask)
        data['edge_index'] = self.dp.get_scatter_edges_index(data['edge_index'], edges_mask)
        
        
        for interaction, product, readout in zip(
            interactions, products, readouts
        ):
            node_feats = self.dp.broadcast(node_feats)
            inputs = [
                {
                'node_attrs': node_attrs[i],
                'node_feats': node_feats[i],
                'edge_attrs': edge_attrs[i],
                'edge_feats': edge_feats[i],
                'edge_index': data['edge_index'][i]
                }
                for i in range(self.dp.num_gpus)
            ]
            '''
            sender = inputs[1]['edge_index'][0]
            receiver = inputs[1]['edge_index'][1]
            num_nodes = inputs[1]['node_feats'].shape[0]
            node_feats = interaction[1].model.linear_up(inputs[1]['node_feats'])
            tp_weights = interaction[1].model.conv_tp_weights(inputs[1]['edge_feats'])
            mji = interaction[1].model.conv_tp(
                inputs[1]['node_feats'][sender], inputs[1]['edge_attrs'], tp_weights
            )
            print('mji:', mji)
            message = scatter_sum(
                src=mji, index=receiver, dim=0, dim_size=num_nodes
            )
            print('message', message)
            message = interaction[1].model.linear(message) / interaction[1].model.avg_num_neighbors
            print('message1', message)
            message = interaction[1].model.skip_tp(message, inputs[1]['node_attrs'])
            print('message2', message)
            print(interaction[1].model.reshape(message))
            print(interaction[1].model(**inputs[1]))
            print(interaction[0](inputs[0]))
            '''
            outputs = self.dp.parallel_apply(interaction, inputs)
            node_feats = self.dp.reduce(
                [outputs[i][0] for i in range(self.dp.num_gpus)], 
                self.dp.devices[0]
            )
            if outputs[0][1] is None:
                sc = [outputs[0][1], ] * self.dp.num_gpus
            else:
                sc = self.dp.scatter_nodes_feats(outputs[0][1], self.dp.devices)
            
            node_feats = self.dp.scatter_nodes_feats(node_feats, self.dp.devices)
            node_attrs_scatter = self.dp.scatter_nodes_feats(node_attrs[0], self.dp.devices)
            
            inputs = [
                {
                    'node_feats': node_feats[i],
                    'sc': sc[i],
                    'node_attrs': node_attrs_scatter[i]
                }
                for i in range(self.dp.num_gpus)
            ]

            outputs = self.dp.parallel_apply(product, inputs)
            node_feats = self.dp.gather_nodes_feats(outputs, self.dp.devices[0])
            node_out = self.dp.gather_nodes_feats(
                self.dp.parallel_apply(readout, outputs), self.dp.devices[0]
            )
            node_feats_list.append(node_feats)
            node_es_list.append(node_out.squeeze(-1))
        # -----------------------------------–––---------------------------------------------
        '''
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
        '''
        
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

        forces, virials, stress, _ = get_outputs(
            energy=inter_e,
            positions=data["positions"],
            displacement=displacement,
            cell=data["cell"],
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
        )
        output = {
            "energy": totol_energy,
            "node_energy": node_energy,
            "interaction_energy": inter_e,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "displacement": displacement,
            "node_feats": node_feats_out,
        }

        return output
    
    

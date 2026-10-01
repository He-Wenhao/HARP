import abc
import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from typing import Optional


class BackboneBase(nn.Module, abc.ABC):
    """Abstract base class for GNN backbones used in direct property prediction.

    Subclasses must set self.node_out_irreps (e3nn Irreps) and
    self.edge_out_irreps (e3nn Irreps or None) in __init__.
    """

    def __init__(self) -> None:
        super().__init__()
        self.node_out_irreps: Optional[Irreps] = None
        self.edge_out_irreps: Optional[Irreps] = None  # None = no edge features

    @abc.abstractmethod
    def forward(self, minibatch: dict) -> dict:
        """Process a minibatch and return node/edge features.

        Args:
            minibatch: dict with keys:
                'pos'          : [N_atoms, 3]
                'sh'           : [N_edges, sh_dim]  spherical harmonics
                'rnorm'        : [N_edges]           edge distances
                'f_in'         : [N_atoms, num_species]  one-hot element types
                'edge_src'     : [N_edges] long
                'edge_dst'     : [N_edges] long
                'batch'        : [N_atoms] long  molecule indices
                'num_nodes'    : int
                'num_neighbors': [N_atoms]
                (optional) 'nodeRDM': [N_atoms, rdm_dim]

        Returns dict with:
            'node_features' : [N_atoms, node_out_irreps.dim]
            'edge_features' : [N_edges, edge_out_irreps.dim]  or None
            'batch'         : [N_atoms]
            'edge_src'      : [N_edges]
            'edge_dst'      : [N_edges]
        """
        ...

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import Irreps
from model.from_schnetpack.acsf import GaussianBasis
from model.graph2irreps import NodeUpdateBlock, EdgeUpdateBlock
from model.backbone.base import BackboneBase


class InternalBackbone(BackboneBase):
    """Current V_theta equivariant GNN (3-layer, lmax=4) without prediction heads.

    Architecture is identical to V_theta's message-passing stack so that
    DirectPropertyModel(InternalBackbone, PropertyHead) is directly comparable
    to legacy V_theta runs (though weights are NOT shared / transferred).

    Args:
        irreps       : Irreps_build instance from util_basis.Irreps
        device       : torch device
        nodeRDM_flag : whether to concatenate nodeRDM to initial node features
        ele_emb      : element embedding width (scalar channels)
        dim_embR     : Gaussian radial basis dimension
    """

    def __init__(self, irreps, device, nodeRDM_flag: bool = False,
                 ele_emb: int = 10, dim_embR: int = 128,
                 num_layers: int = 3) -> None:
        super().__init__()
        self.device = device
        self.nodeRDM_flag = nodeRDM_flag

        self.embR_func = GaussianBasis(0.0, 8.0, dim_embR)

        num_species = irreps.get_input_irreps()
        self.num_species = num_species
        irreps_sh  = irreps.get_sh_irreps(lmax=4)
        max_irreps = irreps.get_max_irreps_square()

        input_irreps      = Irreps(f'{ele_emb}x0e')
        input_irreps_edge = Irreps(f'{dim_embR}x0e') + irreps_sh
        if nodeRDM_flag:
            input_irreps = input_irreps + max_irreps

        widths = [64, 32, 16, 8, 8]
        w = widths
        irreps_hidden = [input_irreps] + [
            f'{w[0]}x0e+{w[1]}x1o+{w[2]}x2e+{w[3]}x3o+{w[4]}x4e',
            f'{w[0]}x0e+{w[1]}x1e+{w[1]}x1o+{w[2]}x2e+{w[2]}x2o+{w[3]}x3e+{w[3]}x3o+{w[4]}x4e+{w[4]}x4o',
            f'{w[0]}x0e+{w[0]}x0o+{w[1]}x1e+{w[1]}x1o+{w[2]}x2e+{w[2]}x2o+{w[3]}x3e+{w[3]}x3o+{w[4]}x4e+{w[4]}x4o',
        ]
        irreps_edge = [input_irreps_edge] + irreps.get_hidden_irreps(lmax=4, widths=widths)

        self.ele_embed = torch.nn.Sequential(torch.nn.Linear(num_species, ele_emb))

        act       = {1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        # num_layers defaults to 3 (original behavior); the v1-EGNN ablation
        # sets num_layers=2 via [backbone] config to mirror the published model.
        assert 1 <= num_layers <= 3, f"InternalBackbone supports 1-3 layers, got {num_layers}"
        node_blocks, edge_blocks = [], []
        for i in range(num_layers):
            node_blocks.append(NodeUpdateBlock(
                num_species=num_species,   fc_len_in=dim_embR,
                irreps_sh=irreps_sh,       irreps_in_node=irreps_hidden[i],
                irreps_out_node=irreps_hidden[i + 1],
                irreps_in_edge=irreps_edge[i],
                act=act, act_gates=act_gates,
                use_selftp=False, use_sc=True, concat=True,
                only_ij=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False,
            ))
            edge_blocks.append(EdgeUpdateBlock(
                num_species=num_species,   fc_len_in=dim_embR,
                irreps_sh=irreps_sh,       irreps_in_node=node_blocks[-1].irreps_out,
                irreps_in_edge=irreps_edge[i],
                irreps_out_edge=irreps_edge[i + 1],
                act=act, act_gates=act_gates,
                use_selftp=False, use_sc=True, init_edge=False,
                nonlin=False, norm='e3LayerNorm', if_sort_irreps=False,
            ))

        self.node_update_blocks = torch.nn.ModuleList(node_blocks)
        self.edge_update_blocks = torch.nn.ModuleList(edge_blocks)

        # Expose Irreps for PropertyHead construction
        self.node_out_irreps = node_blocks[-1].irreps_out
        self.edge_out_irreps = edge_blocks[-1].irreps_out

    def forward(self, minibatch: dict) -> dict:
        sh           = minibatch['sh']
        rnorm        = minibatch['rnorm']
        embR         = self.embR_func(rnorm)
        node_one_hot = minibatch['f_in']
        edge_src     = minibatch['edge_src']
        edge_dst     = minibatch['edge_dst']
        edge_indices = torch.stack((edge_src, edge_dst), dim=0)
        batch        = minibatch['batch']

        embZ             = self.ele_embed(node_one_hot)
        node_feature     = embZ
        node_one_hot_num = torch.argmax(node_one_hot, dim=1)
        edge_one_hot     = F.one_hot(
            self.num_species * node_one_hot_num[edge_src] + node_one_hot_num[edge_dst],
            num_classes=self.num_species ** 2,
        ).float()
        edge_feature = torch.cat((embR, sh), dim=1)

        if self.nodeRDM_flag:
            node_feature = torch.cat((node_feature, minibatch['nodeRDM']), dim=1)

        for node_blk, edge_blk in zip(self.node_update_blocks, self.edge_update_blocks):
            node_feature = node_blk(
                node_feature, node_one_hot, sh, edge_feature,
                embR, edge_indices, batch, selfloop_edge=None, edge_length=rnorm,
            )
            edge_feature = edge_blk(
                node_feature, edge_one_hot, sh, edge_feature,
                embR, edge_indices, batch,
            )

        return {
            'node_features': node_feature,
            'edge_features': edge_feature,
            'batch':         batch,
            'edge_src':      edge_src,
            'edge_dst':      edge_dst,
        }

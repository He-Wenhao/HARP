# in tp_conv, concatenate both source node and destination node as input feature

import torch
from torch_cluster import radius_graph
from torch_scatter import scatter
import torch.nn.functional as F
from e3nn import o3
from e3nn import nn
from e3nn.nn import Gate
from model.from_schnetpack.acsf import GaussianBasis
from e3nn.o3 import Irreps, Linear, SphericalHarmonics, FullyConnectedTensorProduct
from model.from_nequip.tp_utils import tp_path_exists
from model.from_deephe3.e3modules import SphericalBasis, sort_irreps, e3LayerNorm, e3ElementWise, SkipConnection, SeparateWeightTensorProduct, SelfTp



def cosine_radius_embedding(rnorm,min_radius = 0.5, max_radius = 2.0):

    crit1, crit2 = rnorm<max_radius, rnorm>min_radius;
    emb = (torch.cos(rnorm/max_radius*torch.pi)+1)/2; 
    emb = (emb*crit1*crit2 + (~crit2)).reshape(len(emb),1);

    return emb

class node_update(torch.nn.Module):
    def __init__(self,irreps_hidden,irreps_hidden_next,irreps_edge,irreps_sh,irreps_embZ,device,emb_neurons=16):
        super().__init__()
        self.device = device;
        

        
        self.tp_convs = o3.FullyConnectedTensorProduct(
                irreps_in1=irreps_hidden+irreps_hidden+irreps_edge,
                irreps_in2=irreps_sh,
                irreps_out=irreps_hidden_next,
                shared_weights=False
            )
        self.fcs = nn.FullyConnectedNet([1, emb_neurons,emb_neurons, 
                            self.tp_convs.weight_numel], torch.tanh)
        self.linears = o3.FullyConnectedTensorProduct(
                irreps_in1=irreps_embZ,
                irreps_in2=irreps_hidden_next,
                irreps_out=irreps_hidden_next,
            )
        irrep = str(irreps_hidden_next);
        Nact, Ntot = irrep.count('0'), irrep.count('+')+1;
        self.activations = nn.Activation(irreps_hidden_next, 
                            [torch.tanh]*Nact+[None]*(Ntot-Nact))

    def forward(self,node_feature,edge_feature,edge_src,edge_dst,sh,embR, num_nodes, num_neighbors, embZ):
        edge_src_dist = torch.cat((node_feature[edge_src],node_feature[edge_dst],edge_feature),dim=1)
        edge_feature = self.tp_convs(edge_src_dist, sh, self.fcs(embR));
        node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors[:,None]**0.5);
        node_feature = self.linears(embZ, node_feature);
        node_feature = self.activations(node_feature);
        return node_feature, edge_feature


def get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, 
                    act={1: torch.nn.functional.silu, -1: torch.tanh}, 
                    act_gates={1: torch.sigmoid, -1: torch.tanh}
                    ):
    # get gate nonlinearity after tensor product
    # irreps_in1 and irreps_in2 are irreps to be multiplied in tensor product
    # irreps_out is desired irreps after gate nonlin
    # notice that nonlin.irreps_out might not be exactly equal to irreps_out
            
    irreps_scalars = Irreps([
        (mul, ir)
        for mul, ir in irreps_out
        if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()
    irreps_gated = Irreps([
        (mul, ir)
        for mul, ir in irreps_out
        if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()
    if irreps_gated.dim > 0:
        if tp_path_exists(irreps_in1, irreps_in2, "0e"):
            ir = "0e"
        elif tp_path_exists(irreps_in1, irreps_in2, "0o"):
            ir = "0o"
            warnings.warn('Using odd representations as gates')
        else:
            raise ValueError(
                f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} is unable to produce gates needed for irreps_gated={irreps_gated}")
    else:
        ir = None
    irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

    gate_nonlin = Gate(
        irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
        irreps_gated  # gated tensors
    )
    
    assert gate_nonlin.irreps_out == irreps_out
    
    return gate_nonlin
        

class EquiConv(torch.nn.Module):
    def __init__(self, fc_len_in, irreps_in1, irreps_in2, irreps_out, norm='', nonlin=True, 
                 act = {1: torch.nn.functional.silu, -1: torch.tanh},
                 act_gates = {1: torch.sigmoid, -1: torch.tanh}
                 ):
        super(EquiConv, self).__init__()
        
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, act, act_gates)
            irreps_tp_out = self.nonlin.irreps_in
        else:
            irreps_tp_out = Irreps([(mul, ir) for mul, ir in irreps_out if tp_path_exists(irreps_in1, irreps_in2, ir)])
        
        self.tp = SeparateWeightTensorProduct(irreps_in1, irreps_in2, irreps_tp_out)
        
        if nonlin:
            self.cfconv = e3ElementWise(self.nonlin.irreps_out)
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.cfconv = e3ElementWise(irreps_tp_out)
            self.irreps_out = irreps_tp_out
        
        # fully connected net to create tensor product weights
        linear_act = torch.nn.SiLU()
        self.fc = torch.nn.Sequential(torch.nn.Linear(fc_len_in, 64),
                                linear_act,
                                torch.nn.Linear(64, 64),
                                linear_act,
                                torch.nn.Linear(64, self.cfconv.len_weight)
                                )

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.cfconv.irreps_in)
            else:
                raise ValueError(f'unknown norm: {norm}')

    def forward(self, fea_in1, fea_in2, fea_weight, batch_edge):
        z = self.tp(fea_in1, fea_in2)

        if self.nonlin is not None:
            z = self.nonlin(z)

        weight = self.fc(fea_weight)
        z = self.cfconv(z, weight)

        if self.norm is not None:
            z = self.norm(z, batch_edge)

        # TODO self-connection here
        return z


class NodeUpdateBlock(torch.nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_out_node, irreps_in_edge,
                 act, act_gates, use_selftp=False, use_sc=True, concat=True, only_ij=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(NodeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_sh = Irreps(irreps_sh)
        irreps_out_node = Irreps(irreps_out_node)
        irreps_in_edge = Irreps(irreps_in_edge)

        if concat:
            irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
            if if_sort_irreps:
                self.sort = sort_irreps(irreps_in1)
                irreps_in1 = self.sort.irreps_out
        else:
            irreps_in1 = irreps_in_node
        irreps_in2 = irreps_sh

        self.lin_pre = Linear(irreps_in=irreps_in_node, irreps_out=irreps_in_node, biases=True)
        
        self.nonlin = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_node, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_node
            conv_nonlin = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, act=act, act_gates=act_gates)
        self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True)
        
        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out
        
        self.sc = None
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_node, f'{num_species}x0e', self.conv.irreps_out)
            
        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')
        
        self.skip_connect = SkipConnection(irreps_in_node, self.irreps_out)
        
        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)

        self.irreps_in_node = irreps_in_node
        self.use_sc = use_sc
        self.concat = concat
        self.only_ij = only_ij
        self.if_sort_irreps = if_sort_irreps

    def forward(self, node_fea, node_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch, selfloop_edge, edge_length):
            
        node_fea_old = node_fea
        
        if self.use_sc:
            node_self_connection = self.sc(node_fea, node_one_hot)

        node_fea = self.lin_pre(node_fea)

        index_i = edge_index[0]
        index_j = edge_index[1]
        if self.concat:
            fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
            if self.if_sort_irreps:
                fea_in = self.sort(fea_in)
            edge_update = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        else:
            edge_update = self.conv(node_fea[index_j], edge_sh, edge_length_embedded, batch[edge_index[0]])
        
        # sigma = 3
        # n = 2
        # edge_update = edge_update * torch.exp(- edge_length ** n / sigma ** n / 2).view(-1, 1)
        
        node_fea = scatter(edge_update, index_i, dim=0, dim_size=node_fea.shape[0], reduce='add')
        if self.only_ij:
            node_fea = node_fea + scatter(edge_update[~selfloop_edge], index_j[~selfloop_edge], dim=0, dim_size=node_fea.shape[0], reduce='add')
            
        node_fea = self.lin_post(node_fea)

        if self.use_sc:
            node_fea = node_fea + node_self_connection
            
        if self.nonlin is not None:
            node_fea = self.nonlin(node_fea)
            
        if self.norm is not None:
            node_fea = self.norm(node_fea, batch)
            
        node_fea = self.skip_connect(node_fea_old, node_fea)
        
        if self.self_tp is not None:
            node_fea = self.self_tp(node_fea)
        
        return node_fea


class EdgeUpdateBlock(torch.nn.Module):
    def __init__(self, num_species, fc_len_in, irreps_sh, irreps_in_node, irreps_in_edge, irreps_out_edge,
                 act, act_gates, use_selftp=False, use_sc=True, init_edge=False, nonlin=False, norm='e3LayerNorm', if_sort_irreps=False):
        super(EdgeUpdateBlock, self).__init__()
        irreps_in_node = Irreps(irreps_in_node)
        irreps_in_edge = Irreps(irreps_in_edge)
        irreps_out_edge = Irreps(irreps_out_edge)

        irreps_in1 = irreps_in_node + irreps_in_node + irreps_in_edge
        if if_sort_irreps:
            self.sort = sort_irreps(irreps_in1)
            irreps_in1 = self.sort.irreps_out
        irreps_in2 = irreps_sh

        self.lin_pre = Linear(irreps_in=irreps_in_edge, irreps_out=irreps_in_edge, biases=True)
        
        self.nonlin = None
        self.lin_post = None
        if nonlin:
            self.nonlin = get_gate_nonlin(irreps_in1, irreps_in2, irreps_out_edge, act, act_gates)
            irreps_conv_out = self.nonlin.irreps_in
            conv_nonlin = False
        else:
            irreps_conv_out = irreps_out_edge
            conv_nonlin = True
            
        self.conv = EquiConv(fc_len_in, irreps_in1, irreps_in2, irreps_conv_out, nonlin=conv_nonlin, act=act, act_gates=act_gates)
        self.lin_post = Linear(irreps_in=self.conv.irreps_out, irreps_out=self.conv.irreps_out, biases=True)
        
        if use_sc:
            self.sc = FullyConnectedTensorProduct(irreps_in_edge, f'{num_species**2}x0e', self.conv.irreps_out)

        if nonlin:
            self.irreps_out = self.nonlin.irreps_out
        else:
            self.irreps_out = self.conv.irreps_out

        self.norm = None
        if norm:
            if norm == 'e3LayerNorm':
                self.norm = e3LayerNorm(self.irreps_out)
            else:
                raise ValueError(f'unknown norm: {norm}')
        
        self.skip_connect = SkipConnection(irreps_in_edge, self.irreps_out) # ! consider init_edge
        
        self.self_tp = None
        if use_selftp:
            self.self_tp = SelfTp(self.irreps_out, self.irreps_out)
            
        self.use_sc = use_sc
        self.init_edge = init_edge
        self.if_sort_irreps = if_sort_irreps
        self.irreps_in_edge = irreps_in_edge

    def forward(self, node_fea, edge_one_hot, edge_sh, edge_fea, edge_length_embedded, edge_index, batch):
        
        if not self.init_edge:
            edge_fea_old = edge_fea
            if self.use_sc:
                edge_self_connection = self.sc(edge_fea, edge_one_hot)
            edge_fea = self.lin_pre(edge_fea)
            
        index_i = edge_index[0]
        index_j = edge_index[1]
        fea_in = torch.cat([node_fea[index_i], node_fea[index_j], edge_fea], dim=-1)
        if self.if_sort_irreps:
            fea_in = self.sort(fea_in)
        edge_fea = self.conv(fea_in, edge_sh, edge_length_embedded, batch[edge_index[0]])
        
        edge_fea = self.lin_post(edge_fea)

        if self.use_sc:
            edge_fea = edge_fea + edge_self_connection
            
        if self.nonlin is not None:
            edge_fea = self.nonlin(edge_fea)

        if self.norm is not None:
            edge_fea = self.norm(edge_fea, batch[edge_index[0]])
        
        if not self.init_edge:
            edge_fea = self.skip_connect(edge_fea_old, edge_fea)
        
        if self.self_tp is not None:
            edge_fea = self.self_tp(edge_fea)

        return edge_fea




class V_theta(torch.nn.Module):
    
    def __init__(self, device, irreps, nodeRDM_flag, ele_emb: int=10, emb_neurons: int = 16, dim_embR = 128, scaling=0.2) -> None:
        super().__init__()
        
        # Initialize a Equivariance graph convolutional neural network
        # nodes with distance smaller than max_radius are connected by bonds
        # num_basis is the number of basis for edge feature embedding
        
        self.device = device;
        
        self.embR_func = GaussianBasis(0.0, 8.0, dim_embR);
        
        self.scaling = scaling;
        self.nodeRDM_flag = nodeRDM_flag
        
        ######## Define the input and output irreps ########
        irreps_pair = irreps.get_pair_irreps();
        irreps_node = irreps.get_onsite_irreps();
        irreps_sh = irreps.get_sh_irreps(lmax=4);
        max_irreps = irreps.get_max_irreps_square();
        num_species = irreps.get_input_irreps();
        self.num_species = num_species
        irreps_linear = o3.Irreps(str(ele_emb)+'x0e')
        input_irreps = o3.Irreps(str(ele_emb)+'x0e')
        input_irreps_edge = o3.Irreps(str(dim_embR)+'x0e') + irreps_sh
        if nodeRDM_flag:
            input_irreps += max_irreps
        # DeepH-style multiplicities: taper with l  [l=0, l=1, l=2, l=3, l=4]
        widths = [64, 32, 16, 8, 8]
        w = widths
        irreps_hidden = [input_irreps] + [
            f'{w[0]}x0e+{w[1]}x1o+{w[2]}x2e+{w[3]}x3o+{w[4]}x4e',
            f'{w[0]}x0e+{w[1]}x1e+{w[1]}x1o+{w[2]}x2e+{w[2]}x2o+{w[3]}x3e+{w[3]}x3o+{w[4]}x4e+{w[4]}x4o',
            f'{w[0]}x0e+{w[0]}x0o+{w[1]}x1e+{w[1]}x1o+{w[2]}x2e+{w[2]}x2o+{w[3]}x3e+{w[3]}x3o+{w[4]}x4e+{w[4]}x4o',
        ]
        irreps_edge = [input_irreps_edge] + irreps.get_hidden_irreps(lmax=4, widths=widths);

        ######### Element embedding network #########
        self.ele_embed = torch.nn.Sequential(
            torch.nn.Linear(num_species, ele_emb)
        )
        ######### Define the graph convolutional neural network #########
        node_update_blocks = [];
        edge_update_blocks = [];
        
        for i in range(3):
            '''
            node_update_blocks.append(node_update(irreps_hidden=irreps_hidden[i],
                                        irreps_hidden_next=irreps_hidden[i+1],
                                        irreps_edge = irreps_edge[i],
                                        irreps_sh=irreps_sh,
                                        irreps_embZ=irreps_linear,
                                        device=self.device,
                                        emb_neurons=emb_neurons))
            '''
            node_update_blocks.append(NodeUpdateBlock(
                                        num_species = num_species, 
                                        fc_len_in = dim_embR, 
                                        irreps_sh = irreps_sh, 
                                        irreps_in_node = irreps_hidden[i], 
                                        irreps_out_node = irreps_hidden[i+1], 
                                        irreps_in_edge = irreps_edge[i],
                                        act={1: torch.nn.functional.silu, -1: torch.tanh}, 
                                        act_gates={1: torch.sigmoid, -1: torch.tanh}, 
                                        use_selftp=False, 
                                        use_sc=True, 
                                        concat=True, 
                                        only_ij=False, 
                                        nonlin=False, 
                                        norm='e3LayerNorm', 
                                        if_sort_irreps=False))
            edge_update_blocks.append(EdgeUpdateBlock(
                                        num_species = num_species, 
                                        fc_len_in= dim_embR, 
                                        irreps_sh= irreps_sh, 
                                        irreps_in_node = node_update_blocks[-1].irreps_out, 
                                        irreps_in_edge = irreps_edge[i], 
                                        irreps_out_edge = irreps_edge[i+1],
                                        act={1: torch.nn.functional.silu, -1: torch.tanh}, 
                                        act_gates={1: torch.sigmoid, -1: torch.tanh}, 
                                        use_selftp=False, 
                                        use_sc=True, 
                                        init_edge=False, 
                                        nonlin=False, 
                                        norm='e3LayerNorm', 
                                        if_sort_irreps=False
                                    ))

        self.node_update_blocks = torch.nn.ModuleList(node_update_blocks);
        self.edge_update_blocks = torch.nn.ModuleList(edge_update_blocks);

        # Retrieve actual output irreps from the last update blocks
        final_node_irreps = node_update_blocks[-1].irreps_out
        final_edge_irreps = edge_update_blocks[-1].irreps_out

        ########### Direct observable prediction heads ###########
        # Helper: 2-layer scalar head (in_irreps → 32x0e → tanh → 1x0e)
        def _scalar_head(in_irreps):
            return torch.nn.ModuleList([
                o3.Linear(in_irreps, o3.Irreps("32x0e")),
                nn.Activation(o3.Irreps("32x0e"), [torch.tanh]),
                o3.Linear(o3.Irreps("32x0e"), o3.Irreps("1x0e")),
            ])

        # Per-atom scalar contributions (summed per molecule for E and E_gap)
        self.node_E      = _scalar_head(final_node_irreps)
        self.node_Egap   = _scalar_head(final_node_irreps)
        self.node_charge = _scalar_head(final_node_irreps)

        # Per-atom dipole contribution (l=1o, 3 components): sum → molecular dipole
        self.node_dipole1   = o3.Linear(final_node_irreps, o3.Irreps(f"32x0e+{w[1]}x1o"))
        self.node_dipole_act = nn.Activation(o3.Irreps(f"32x0e+{w[1]}x1o"), [torch.tanh, None])
        self.node_dipole2   = o3.Linear(o3.Irreps(f"32x0e+{w[1]}x1o"), o3.Irreps("1x1o"))

        # Per-atom quadrupole contribution (l=0e + l=2e, 6 components): sum → molecular tensor
        self.node_quad1   = o3.Linear(final_node_irreps, o3.Irreps(f"32x0e+{w[2]}x2e"))
        self.node_quad_act = nn.Activation(o3.Irreps(f"32x0e+{w[2]}x2e"), [torch.tanh, None])
        self.node_quad2   = o3.Linear(o3.Irreps(f"32x0e+{w[2]}x2e"), o3.Irreps("1x0e+1x2e"))

        # Per-atom polarizability contribution (same structure as quadrupole)
        self.node_alpha1   = o3.Linear(final_node_irreps, o3.Irreps(f"32x0e+{w[2]}x2e"))
        self.node_alpha_act = nn.Activation(o3.Irreps(f"32x0e+{w[2]}x2e"), [torch.tanh, None])
        self.node_alpha2   = o3.Linear(o3.Irreps(f"32x0e+{w[2]}x2e"), o3.Irreps("1x0e+1x2e"))

        # Per-edge bond order (l=0e scalar)
        self.edge_bond1   = o3.Linear(final_edge_irreps, o3.Irreps("32x0e"))
        self.edge_bond_act = nn.Activation(o3.Irreps("32x0e"), [torch.tanh])
        self.edge_bond2   = o3.Linear(o3.Irreps("32x0e"), o3.Irreps("1x0e"))

        # Zero-initialize all output layers for stable training start
        with torch.no_grad():
            for head in [self.node_E, self.node_Egap, self.node_charge]:
                head[2].weight.zero_()
            self.node_dipole2.weight.zero_()
            self.node_quad2.weight.zero_()
            self.node_alpha2.weight.zero_()
            self.edge_bond2.weight.zero_()
    def forward(self, minibatch) -> torch.Tensor:
        
        # Forward function of the neural network model
        # positions is a Nx3 tensor, including the N times 3D atomic cartesian coordinate
        # elements is a N-dim list, whose ith component is either 'H' or 'C' denoting 
        # the ith atomic species.
        # The output is a (N*14)x(N*14) V_theta matrix.
        # The i*14+j row/column means the ith atom's jth basis orbital
        # The basis on an atom is ranked as (s1,s2,s3,p1_(-1,0,1),p2_(-1,0,1),d1_(-2,-1,0,1,2))
        sh = minibatch['sh'];
        rnorm = minibatch['rnorm'];
        #embR = cosine_radius_embedding(rnorm);
        embR = self.embR_func(rnorm);
        if self.nodeRDM_flag:
            nodeRDM = minibatch['nodeRDM'];
        node_one_hot = minibatch['f_in'];
        edge_src = minibatch['edge_src'];
        edge_dst = minibatch['edge_dst'];
        edge_indices = torch.stack((edge_src,edge_dst),dim=0);
        num_nodes = minibatch['num_nodes'];
        num_neighbors = minibatch['num_neighbors'];
        mask = minibatch['pair_ind'];
        batch = minibatch['batch'];
        
        embZ = self.ele_embed(node_one_hot);
        node_feature = embZ;
        node_one_hot_num = torch.argmax(node_one_hot,dim=1);
        edge_one_hot = F.one_hot(self.num_species * node_one_hot_num[edge_src] + node_one_hot_num[edge_dst],
                                 num_classes=self.num_species**2).type(torch.float32);
        edge_feature = torch.cat((embR,sh),dim=1)
        
        if self.nodeRDM_flag:
            node_feature = torch.cat((node_feature,nodeRDM),dim=1)

        for i, (node_update_block, edge_update_block) in enumerate(zip(self.node_update_blocks, self.edge_update_blocks)):
            node_feature = node_update_block(node_feature, node_one_hot, sh, edge_feature, embR, edge_indices, batch, selfloop_edge=None, edge_length = rnorm)
            
            edge_feature = edge_update_block(node_feature, edge_one_hot, sh, edge_feature, embR, edge_indices, batch)

            #edge_src_dist = torch.cat((node_feature[edge_src],node_feature[edge_dst]),dim=1)
            #edge_feature = self.tp_convs[i](edge_src_dist, sh, self.fcs[i](emb));
            #node_feature = scatter(edge_feature, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors[:,None]**0.5);
            #node_feature = self.linears[i](inputs, node_feature);
            #node_feature = self.activations[i](node_feature);
            '''
        for i in range(3):
            node_feature, edge_feature = self.node_update_blocks[i](
                                                node_feature = node_feature,
                                                edge_feature = edge_feature,
                                                edge_src = edge_src,
                                                edge_dst = edge_dst,
                                                sh = sh,
                                                embR = embR,
                                                num_nodes = num_nodes,
                                                num_neighbors = num_neighbors,
                                                embZ = embZ
                                            );
            '''
            

        def _apply_scalar_head(head, x):
            return head[2](head[1](head[0](x)))

        # Per-atom predictions
        atom_E      = _apply_scalar_head(self.node_E,      node_feature)  # [N_atoms, 1]
        atom_Egap   = _apply_scalar_head(self.node_Egap,   node_feature)  # [N_atoms, 1]
        atom_charge = _apply_scalar_head(self.node_charge, node_feature)  # [N_atoms, 1]

        atom_dipole = self.node_dipole2(self.node_dipole_act(self.node_dipole1(node_feature)))  # [N_atoms, 3]
        atom_quad   = self.node_quad2(self.node_quad_act(self.node_quad1(node_feature)))        # [N_atoms, 6]
        atom_alpha  = self.node_alpha2(self.node_alpha_act(self.node_alpha1(node_feature)))     # [N_atoms, 6]

        # Per-edge prediction
        edge_bond = self.edge_bond2(self.edge_bond_act(self.edge_bond1(edge_feature)))  # [N_edges, 1]

        return {
            'atom_E':      atom_E,
            'atom_Egap':   atom_Egap,
            'atom_charge': atom_charge,
            'atom_dipole': atom_dipole,
            'atom_quad':   atom_quad,
            'atom_alpha':  atom_alpha,
            'edge_bond':   edge_bond,
            'batch':       batch,
            'edge_src':    edge_src,
            'edge_dst':    edge_dst,
        }



"""Hamiltonian (DeepH-style) head on top of an arbitrary node-feature backbone.

Motivation
----------
The direct-property model (DirectPropertyModel + PropertyHead) predicts each
observable with its own equivariant head and aggregates per-atom contributions
by a plain sum/mean. That makes molecular properties size-LINEAR by
construction, so it cannot extrapolate to larger systems (the size-scaling
failure noted in the v8 README).

This head instead predicts a correction ``V`` to the (orthonormalised) BP86
Fock matrix ``h``. Observables are then computed from the eigenstates of
``H = h + V`` through the analytic Feynman-Hellmann formulas in
``model/Fock2obs.py`` (energy, dipole, charge, bond order, gap, polarizability).
Because every observable is a functional of the same self-consistent
Hamiltonian, size-extensivity / size-scaling is guaranteed by the physics
rather than by an aggregation choice.

Architecture
------------
- A backbone (e.g. the v8 eSEN / EquiformerV2 backbone) supplies per-atom
  equivariant node features ``node_out_irreps``.
- Onsite (diagonal) Fock blocks come from per-element ``o3.Linear`` maps of the
  node features -> the element's onsite irreps (same interface the legacy
  ``V_theta`` used, so ``to_mat.raw_to_mat`` is reused verbatim).
- The v8 backbone exposes NO per-edge features and builds its own internal
  graph, so off-diagonal (pair/bond) blocks are produced here by an equivariant
  edge module (``EquiConv``) that consumes ``node_i, node_j`` and the edge
  spherical harmonics on the *sampler's* radius graph (the same graph
  ``to_mat.raw_to_mat`` indexes through ``pair_ind`` / ``map1``).
- screen / gap / delta_E scalar+tensor heads reproduce the legacy corrections
  consumed by ``state_solver`` (T-matrix, gap rescale, energy delta).

The forward() return dict has exactly the keys the legacy ``V_theta`` produced
(``node``, ``edge``, ``screen``, ``gap``, ``delta_E``) so the trainer's
``inference()`` -> ``to_mat.raw_to_mat`` -> ``state_solver`` path is unchanged.
"""

import torch
import torch.nn as nn
from e3nn import o3
from e3nn import nn as e3nn_nn

from model.backbone.base import BackboneBase
from model.graph2irreps import EquiConv
from model.from_schnetpack.acsf import GaussianBasis


# Edge-feature hidden irreps. Onsite/pair Fock blocks only ever carry
# natural-parity irreps (l even -> e, l odd -> o : 0e,1o,2e,3o,4e), which is the
# same convention the eSEN node features and the spherical harmonics use, so no
# parity gymnastics are needed. l up to 4 is required because d (x) d orbital
# pairs produce l=4 blocks.
_EDGE_HIDDEN = '64x0e+32x1o+16x2e+8x3o+8x4e'


class HamiltonianHead(nn.Module):
    """Map backbone node features -> Fock correction blocks (V_raw dict)."""

    def __init__(self, node_in_irreps, irreps, device,
                 dim_embR: int = 128, scaling: float = 0.2,
                 sh_lmax: int = 4, edge_hidden: str = _EDGE_HIDDEN) -> None:
        super().__init__()
        self.device = device
        self.scaling = float(scaling)

        node_in = o3.Irreps(node_in_irreps)

        irreps_node = irreps.get_onsite_irreps()     # one Irreps per element
        irreps_pair = irreps.get_pair_irreps()       # one Irreps per element-pair (j<=i)
        irreps_sh   = irreps.get_sh_irreps(lmax=sh_lmax)

        # Validate the backbone reaches l=4 so d-d onsite blocks are representable.
        max_l_node = max(ir.l for _, ir in node_in)
        if max_l_node < 4:
            raise ValueError(
                f"HamiltonianHead: backbone node_out_irreps max l = {max_l_node} < 4. "
                f"Fock blocks need l up to 4 (d x d). Set the backbone lmax to 4 "
                f"(e.g. [backbone] lmax_list = 4)."
            )

        # ── Radial embedding feeding the edge-conv tensor-product weights ──
        self.embR_func = GaussianBasis(0.0, 8.0, dim_embR)

        # ── Equivariant edge module: cat[node_i, node_j] (x) sh -> edge feats ──
        # nonlin=False -> EquiConv silently keeps only paths that exist (no Gate
        # assertion to satisfy) and applies per-edge channel weights from embR.
        self.edge_conv = EquiConv(
            fc_len_in=dim_embR,
            irreps_in1=(node_in + node_in),
            irreps_in2=irreps_sh,
            irreps_out=o3.Irreps(edge_hidden),
            nonlin=False,
        )
        edge_out = self.edge_conv.irreps_out

        # ── Onsite (diagonal) block heads: one o3.Linear per element ──
        self.linear_node = nn.ModuleList([o3.Linear(node_in, ir) for ir in irreps_node])
        # ── Pair (off-diagonal) block heads: one o3.Linear per element-pair ──
        self.linear_edge = nn.ModuleList([o3.Linear(edge_out, ir) for ir in irreps_pair])

        # ── screen / gap / delta_E heads (consumed by state_solver) ──
        self.screen1   = o3.Linear(node_in, o3.Irreps('32x0e+1x2e'))
        self.screen_act = e3nn_nn.Activation(o3.Irreps('32x0e+1x2e'), [torch.tanh, None])
        self.screen2   = o3.Linear(o3.Irreps('32x0e+1x2e'), o3.Irreps('1x0e+1x2e'))

        self.gap1   = o3.Linear(node_in, o3.Irreps('32x0e'))
        self.gap_act = e3nn_nn.Activation(o3.Irreps('32x0e'), [torch.tanh])
        self.gap2   = o3.Linear(o3.Irreps('32x0e'), o3.Irreps('3x0e'))

        self.dE1   = o3.Linear(node_in, o3.Irreps('32x0e+1x2e'))
        self.dE_act = e3nn_nn.Activation(o3.Irreps('32x0e+1x2e'), [torch.tanh, None])
        self.dE2   = o3.Linear(o3.Irreps('32x0e+1x2e'), o3.Irreps('1x0e'))

        # ── Zero-init every block-producing layer so the initial correction
        #    V = 0  =>  H = h (pure BP86) and observables start at the BP86
        #    baseline. Training then learns the delta. ──
        with torch.no_grad():
            for lin in self.linear_node:
                lin.weight.zero_()
            for lin in self.linear_edge:
                lin.weight.zero_()
            self.screen2.weight.zero_()
            self.gap2.weight.zero_()
            self.dE2.weight.zero_()

    def forward(self, node_features: torch.Tensor, minibatch: dict) -> dict:
        edge_src = minibatch['edge_src']
        edge_dst = minibatch['edge_dst']
        sh       = minibatch['sh']
        rnorm    = minibatch['rnorm']
        batch    = minibatch['batch']
        pair_ind = minibatch['pair_ind']

        # Off-diagonal blocks: build edge features on the sampler's radius graph.
        embR    = self.embR_func(rnorm)
        fea_in1 = torch.cat([node_features[edge_src], node_features[edge_dst]], dim=-1)
        edge_features = self.edge_conv(fea_in1, sh, embR, batch[edge_src])

        # Per-element onsite blocks (computed for all atoms; raw_to_mat selects
        # the matching element per atom) and per-pair off-diagonal blocks.
        node_onsite = [lin(node_features) * self.scaling for lin in self.linear_node]
        edge_pair   = [self.linear_edge[i](edge_features[pair_ind[i]]) * self.scaling
                       for i in range(len(self.linear_edge))]

        screen  = self.screen2(self.screen_act(self.screen1(node_features)))
        gap     = self.gap2(self.gap_act(self.gap1(node_features)))
        delta_E = self.dE2(self.dE_act(self.dE1(node_features)))

        return {
            'node':    node_onsite,
            'edge':    edge_pair,
            'screen':  screen,
            'gap':     gap,
            'delta_E': delta_E,
        }


class HamiltonianModel(nn.Module):
    """Compose a node-feature backbone with a HamiltonianHead.

    forward(minibatch) returns the V_raw dict expected by
    ``to_mat.raw_to_mat`` (keys: node, edge, screen, gap, delta_E).
    """

    def __init__(self, backbone: BackboneBase, head: HamiltonianHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, minibatch: dict) -> dict:
        backbone_out = self.backbone(minibatch)
        return self.head(backbone_out['node_features'], minibatch)

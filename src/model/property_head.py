import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3, nn as e3nn_nn
from e3nn.o3 import Irreps, Linear

# Literature-ish isotropic atomic polarizabilities (a.u. = Bohr^3) used only to
# initialise the per-element Thole baseline so training starts from physical
# atomic polarizabilities (matrix_exp(log_base)*I). Values are coarse; the head
# learns environment-dependent corrections on top. Unknown elements -> ~8 a.u.
_ATOMIC_ALPHA_AU = {
    'H': 2.5, 'C': 9.0, 'N': 7.0, 'O': 5.0, 'F': 3.5,
    'Si': 20.0, 'P': 25.0, 'S': 16.0, 'Cl': 15.0,
}


class PropertyHead(nn.Module):
    """Equivariant output heads for direct molecular property prediction.

    Takes per-atom (and optionally per-edge) equivariant features from any
    BackboneBase subclass and predicts seven molecular properties:
        E, E_gap, atomic_charge  — scalar (l=0e) aggregated per molecule
        dipole x,y,z             — l=1o aggregated per molecule
        quadrupole (6 components) — l=0e + l=2e aggregated per molecule
        polarizability (3×3)      — l=0e + l=2e aggregated per molecule
        bond_order               — scalar per edge

    The return dict has the same keys as V_theta.forward() so that
    train.py's direct_predict_properties() works without modification.

    Args:
        node_out_irreps : e3nn Irreps string or object for backbone node output.
                          Must contain at least l=1o and l=2e channels.
        edge_out_irreps : e3nn Irreps string/object for backbone edge output, or None.
                          If None, bond order uses a node-pair scalar fallback.
    """

    def __init__(self, node_out_irreps, edge_out_irreps=None,
                 alpha_head='sum', element_list=None) -> None:
        super().__init__()

        # alpha_head: 'sum'   -> alpha_mol = baseline + sum_i alpha_i (size-linear, legacy)
        #             'thole' -> alpha_mol = relay matrix over per-atom PD alpha_i
        #                        (size-extensive / super-linear; predicts FULL alpha,
        #                        no DFT baseline). See src/model/thole.py.
        self.alpha_head = str(alpha_head).strip().lower()

        node_irreps = Irreps(node_out_irreps)

        # Validate required channels
        has_1o = any(ir.l == 1 and ir.p == -1 for _, ir in node_irreps)
        has_2e = any(ir.l == 2 and ir.p ==  1 for _, ir in node_irreps)
        if not has_1o:
            raise ValueError(
                f'PropertyHead: node_out_irreps must contain l=1o for dipole. '
                f'Got: {node_irreps}. Configure your backbone to output 1o channels.'
            )
        if not has_2e:
            raise ValueError(
                f'PropertyHead: node_out_irreps must contain l=2e for quadrupole/alpha. '
                f'Got: {node_irreps}. Configure your backbone to output 2e channels.'
            )

        n_1o = sum(mul for mul, ir in node_irreps if ir.l == 1 and ir.p == -1)
        n_2e = sum(mul for mul, ir in node_irreps if ir.l == 2 and ir.p ==  1)
        w1o  = min(n_1o, 32)
        w2e  = min(n_2e, 16)

        # ── Scalar heads (node_irreps → 32x0e → tanh → 1x0e) ─────────────
        def _scalar_head(in_irreps):
            return nn.ModuleList([
                o3.Linear(in_irreps, Irreps('32x0e')),
                e3nn_nn.Activation(Irreps('32x0e'), [torch.tanh]),
                o3.Linear(Irreps('32x0e'), Irreps('1x0e')),
            ])

        self.node_E      = _scalar_head(node_irreps)
        self.node_Egap   = _scalar_head(node_irreps)
        self.node_charge = _scalar_head(node_irreps)

        # ── Dipole head (→ 1x1o, 3 components) ───────────────────────────
        self.node_dipole1    = o3.Linear(node_irreps, Irreps(f'32x0e+{w1o}x1o'))
        self.node_dipole_act = e3nn_nn.Activation(Irreps(f'32x0e+{w1o}x1o'), [torch.tanh, None])
        self.node_dipole2    = o3.Linear(Irreps(f'32x0e+{w1o}x1o'), Irreps('1x1o'))

        # ── Quadrupole head (→ 1x0e+1x2e, 6 components) ──────────────────
        self.node_quad1    = o3.Linear(node_irreps, Irreps(f'32x0e+{w2e}x2e'))
        self.node_quad_act = e3nn_nn.Activation(Irreps(f'32x0e+{w2e}x2e'), [torch.tanh, None])
        self.node_quad2    = o3.Linear(Irreps(f'32x0e+{w2e}x2e'), Irreps('1x0e+1x2e'))

        # ── Polarizability head (same structure as quadrupole) ────────────
        self.node_alpha1    = o3.Linear(node_irreps, Irreps(f'32x0e+{w2e}x2e'))
        self.node_alpha_act = e3nn_nn.Activation(Irreps(f'32x0e+{w2e}x2e'), [torch.tanh, None])
        self.node_alpha2    = o3.Linear(Irreps(f'32x0e+{w2e}x2e'), Irreps('1x0e+1x2e'))

        # ── Thole relay-matrix parameters (only used when alpha_head == 'thole') ──
        # The per-atom 0e+2e output above becomes a symmetric tensor S_i; the PD
        # atomic polarizability is matrix_exp(S_i + log_base[elem]*I). log_base is
        # a learnable per-element baseline (init = log of literature atomic alpha)
        # and thole_a is a single learnable Thole screening scalar.
        if self.alpha_head in ('thole', 'thole_aniso', 'relay_sqe'):
            import math
            if self.alpha_head == 'thole_aniso':
                from model.thole import (ISO_MAX_ANISO as _AMAX,
                                         THOLE_A_MIN_ANISO as _AMIN,
                                         THOLE_A_MAX_ANISO as _AMAXX)
                a0 = 0.8                       # init screening in the extrapolating zone
            else:
                from model.thole import (ALPHA_MAX as _AMAX,
                                         THOLE_A_MIN as _AMIN,
                                         THOLE_A_MAX as _AMAXX)
                a0 = 0.7
            self.alpha_max = float(_AMAX)
            self.thole_a_min = float(_AMIN)
            self.thole_a_max = float(_AMAXX)
            elems = element_list or list(_ATOMIC_ALPHA_AU)
            self.alpha_element_list = list(elems)

            def _inv_sigmoid(p):
                p = min(max(p, 1e-3), 1 - 1e-3)
                return math.log(p / (1 - p))

            # Per-element logit baseline so sigmoid(base)*alpha_max equals the
            # literature atomic alpha at zero head output. alpha_i is bounded in
            # (0, alpha_max), preventing the polarization catastrophe.
            base = torch.tensor([
                _inv_sigmoid(_ATOMIC_ALPHA_AU.get(e, 8.0) / self.alpha_max)
                for e in self.alpha_element_list])
            self.alpha_log_base = nn.Parameter(base)
            # Bounded screening a = a_min + (a_max-a_min)*sigmoid(logit); a0 set per mode above.
            frac = (a0 - self.thole_a_min) / (self.thole_a_max - self.thole_a_min)
            self.thole_a_logit = nn.Parameter(torch.tensor(_inv_sigmoid(frac)))

        if self.alpha_head == 'relay_sqe':
            # Charge-flow (SQE) head: per-atom hardness eta (scalar) + bond hardness
            # kappa. alpha = relay(dipole) + SQE(eta, kappa). kappa is FIXED at 0.05
            # (not learnable): a learnable kappa drifts up over long training and
            # turns the charge flow OFF (the failure mode that lost to v8). The
            # per-atom eta head provides the per-environment charge-flow strength.
            self.eta_min, self.eta_max = 0.1, 5.0
            self.kappa_fixed = 0.05
            self.node_hardness = nn.ModuleList([
                o3.Linear(node_irreps, Irreps('32x0e')),
                e3nn_nn.Activation(Irreps('32x0e'), [torch.tanh]),
                o3.Linear(Irreps('32x0e'), Irreps('1x0e')),
            ])
            with torch.no_grad():
                self.node_hardness[2].weight.zero_()        # eta starts at mid-range

        # ── Bond order head ───────────────────────────────────────────────
        self.has_edge_features = edge_out_irreps is not None
        if self.has_edge_features:
            edge_irreps = Irreps(edge_out_irreps)
            self.edge_bond1    = o3.Linear(edge_irreps, Irreps('32x0e'))
            self.edge_bond_act = e3nn_nn.Activation(Irreps('32x0e'), [torch.tanh])
            self.edge_bond2    = o3.Linear(Irreps('32x0e'), Irreps('1x0e'))
        else:
            # Fallback: extract node scalars equivariantly, then form edge pairs
            # o3.Linear projects from node_irreps → 32x0e (equivariant, keeps only 0e paths)
            self.node_to_scalar = o3.Linear(node_irreps, Irreps('32x0e'))
            self.bond_fallback  = nn.Sequential(
                nn.Linear(64, 32),  # cat [scalar_src, scalar_dst] → 32
                nn.Tanh(),
                nn.Linear(32, 1),
            )

        # ── Zero-initialize all output projection layers ──────────────────
        with torch.no_grad():
            for head in [self.node_E, self.node_Egap, self.node_charge]:
                head[2].weight.zero_()
            self.node_dipole2.weight.zero_()
            self.node_quad2.weight.zero_()
            self.node_alpha2.weight.zero_()
            if self.has_edge_features:
                self.edge_bond2.weight.zero_()
            else:
                nn.init.zeros_(self.bond_fallback[-1].weight)
                nn.init.zeros_(self.bond_fallback[-1].bias)

    def forward(self, backbone_out: dict) -> dict:
        """Predict per-atom/edge properties from backbone output features.

        Args:
            backbone_out: dict from BackboneBase.forward() with keys:
                'node_features', 'edge_features' (or None),
                'batch', 'edge_src', 'edge_dst'

        Returns dict with same keys as V_theta.forward():
            atom_E, atom_Egap, atom_charge, atom_dipole, atom_quad, atom_alpha,
            edge_bond, batch, edge_src, edge_dst
        """
        nf       = backbone_out['node_features']
        ef       = backbone_out.get('edge_features')
        batch    = backbone_out['batch']
        edge_src = backbone_out['edge_src']
        edge_dst = backbone_out['edge_dst']

        def _apply_scalar(head, x):
            return head[2](head[1](head[0](x)))

        atom_E      = _apply_scalar(self.node_E,      nf)
        atom_Egap   = _apply_scalar(self.node_Egap,   nf)
        atom_charge = _apply_scalar(self.node_charge, nf)
        atom_dipole = self.node_dipole2(self.node_dipole_act(self.node_dipole1(nf)))
        atom_quad   = self.node_quad2(self.node_quad_act(self.node_quad1(nf)))
        atom_alpha  = self.node_alpha2(self.node_alpha_act(self.node_alpha1(nf)))
        if self.alpha_head == 'relay_sqe':
            atom_hardness = _apply_scalar(self.node_hardness, nf)   # raw per-atom eta logit

        if self.has_edge_features:
            edge_bond = self.edge_bond2(self.edge_bond_act(self.edge_bond1(ef)))
        else:
            # Extract l=0e scalars from each atom, concatenate for each edge, predict bond
            node_scalar = self.node_to_scalar(nf)   # [N_atoms, 32]
            edge_pair   = torch.cat([node_scalar[edge_src], node_scalar[edge_dst]], dim=-1)
            edge_bond   = self.bond_fallback(edge_pair)   # [N_edges, 1]

        out = {
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
        if self.alpha_head == 'relay_sqe':
            out['atom_hardness'] = atom_hardness     # raw; bounded in direct_model
        # NOTE: the Thole relay alpha is assembled in DirectPropertyModel.forward
        # (which has pos/f_in/batch) using the head's parameters directly, so all
        # parameter use stays inside the DDP-wrapped forward (DDP-safe). The head
        # does NOT emit thole_a/alpha_log_base into `out`.
        return out

    def thole_a_value(self):
        """Bounded Thole screening a = a_min + (a_max-a_min)*sigmoid(logit)."""
        return self.thole_a_min + (self.thole_a_max - self.thole_a_min) * \
            torch.sigmoid(self.thole_a_logit)

    def kappa_value(self):
        """SQE bond hardness kappa -- FIXED at 0.05 (charge flow stays ON)."""
        return self.kappa_fixed

    def eta_from_raw(self, raw_hardness):
        """Map raw per-atom hardness logit -> bounded eta in (eta_min, eta_max)."""
        return self.eta_min + (self.eta_max - self.eta_min) * torch.sigmoid(raw_hardness)

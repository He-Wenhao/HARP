import torch
from e3nn.o3 import Irreps
from model.backbone.base import BackboneBase


class MACEBackbone(BackboneBase):
    """MACE equivariant GNN backbone (mace-torch library).

    Install: pip install mace-torch

    MACE uses e3nn internally, so its output node_feats are directly compatible
    with e3nn-based PropertyHead.

    IMPORTANT: hidden_irreps MUST have UNIFORM multiplicity across all irrep types
    (e.g. '64x0e+64x1o+64x2e', NOT '64x0e+32x1o+16x2e'). Non-uniform multiplicities
    cause a shape mismatch in MACE's SymmetricContraction block.

    MACE concatenates node features from all interaction layers, so node_out_irreps
    is computed automatically after model construction.

    Default hidden_irreps '64x0e+64x1o+64x2e' includes l=0,1,2 with uniform
    multiplicity, satisfying all PropertyHead requirements (dipole l=1o,
    quadrupole/polarizability l=0e+l=2e).

    Reference: Batatia et al., "MACE: Higher Order Equivariant Message Passing
    Neural Networks for Fast and Accurate Force Fields" (NeurIPS 2022).

    Args:
        num_species      : number of element types (e.g. 9 for H,C,N,O,F,Si,P,S,Cl)
        device           : torch device
        r_max            : atomic interaction cutoff in Angstrom (default 5.0)
        num_radial_basis : number of Bessel radial basis functions (default 8)
        max_ell          : max angular momentum for internal edge features (default 3)
        num_interactions : number of MACE interaction blocks / GNN layers (default 2)
        hidden_irreps    : e3nn Irreps string — MUST have uniform multiplicity
                           (default '64x0e+64x1o+64x2e')
        MLP_irreps       : Irreps for MACE's internal MLP readout gate (default '16x0e')
        correlation      : body order of MACE ACE expansion (default 3)
    """

    # Default atomic numbers for H,C,N,O,F,Si,P,S,Cl (our standard element list)
    _DEFAULT_Z = [1, 6, 7, 8, 9, 14, 15, 16, 17]

    def __init__(
        self,
        num_species: int,
        device,
        r_max: float = 5.0,
        num_radial_basis: int = 8,
        max_ell: int = 3,
        num_interactions: int = 2,
        hidden_irreps: str = '64x0e+64x1o+64x2e',
        MLP_irreps: str = '16x0e',
        correlation: int = 3,
        atomic_numbers: list = None,
    ) -> None:
        super().__init__()
        try:
            from mace.modules import MACE
            from mace.modules.blocks import RealAgnosticResidualInteractionBlock
            import torch.nn.functional as F
        except ImportError:
            raise ImportError(
                'mace-torch is required for MACEBackbone. '
                'Install with: pip install mace-torch'
            )
        import numpy as np

        self.device = device
        self.r_max  = r_max

        z_list = atomic_numbers if atomic_numbers is not None else self._DEFAULT_Z[:num_species]

        self._mace = MACE(
            r_max=r_max,
            num_bessel=num_radial_basis,
            num_polynomial_cutoff=5,
            max_ell=max_ell,
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticResidualInteractionBlock,
            num_interactions=num_interactions,
            num_elements=num_species,
            hidden_irreps=Irreps(hidden_irreps),
            MLP_irreps=Irreps(MLP_irreps),
            atomic_energies=np.zeros(num_species),
            avg_num_neighbors=10.0,
            atomic_numbers=z_list,
            correlation=correlation,
            gate=F.silu,
        )

        # MACE concatenates node_feats from ALL interaction layers (see models.py:
        # node_feats_out = torch.cat(node_feats_concat, dim=-1)).
        # Compute the true node_out_irreps by summing the output irreps of each product block.
        concat_irreps = Irreps(
            sum(
                (prod.linear.irreps_out for prod in self._mace.products),
                Irreps([]),
            )
        ).simplify()
        self.node_out_irreps = concat_irreps
        self.edge_out_irreps = None  # MACE does not expose per-edge features

    def forward(self, minibatch: dict) -> dict:
        """Build MACE-specific graph from positions and run MACE forward.

        MACE builds its own radius graph with r_max, separate from the internal
        backbone's 2 Å cutoff used for sh/rnorm. The edge_src/dst returned here
        are from the MACE graph (r_max cutoff), not the internal graph.

        Note on MACE edge convention: radius_graph returns (src, dst) where
        edge goes src→dst. MACE typically uses (receiver, sender) convention
        internally; verify with your mace-torch version if predictions look wrong.
        """
        from torch_cluster import radius_graph

        pos   = minibatch['pos']    # [N_atoms, 3]
        f_in  = minibatch['f_in']   # [N_atoms, num_species]  one-hot
        batch = minibatch['batch']  # [N_atoms]

        # Build MACE graph (no self-loops; MACE handles this internally)
        edge_index = radius_graph(pos, r=self.r_max, batch=batch, loop=False)
        # edge_index[0]=src, edge_index[1]=dst  (torch_cluster convention)

        shifts      = torch.zeros(edge_index.shape[1], 3, device=pos.device)
        unit_shifts = torch.zeros(edge_index.shape[1], 3, device=pos.device)

        n_mol = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=n_mol)
        ptr    = torch.zeros(n_mol + 1, dtype=torch.long, device=batch.device)
        ptr[1:] = torch.cumsum(counts, dim=0)

        # MACE 0.3.x expects a dict with specific keys (see prepare_graph in mace/modules/utils.py)
        # cell is [n_mol, 3, 3]; set to zeros for non-periodic (molecular) systems
        cell = torch.zeros(n_mol, 3, 3, dtype=pos.dtype, device=pos.device)
        data = {
            'positions':    pos,
            'node_attrs':   f_in,
            'edge_index':   edge_index,
            'shifts':       shifts,
            'unit_shifts':  unit_shifts,
            'batch':        batch,
            'ptr':          ptr,
            'cell':         cell,
        }

        # MACE forward: disable force/virial/stress to prevent internal autograd.grad
        # calls that would consume the computation graph before our external backward.
        out = self._mace(data, compute_force=False, compute_virials=False, compute_stress=False)
        node_features = out['node_feats']

        return {
            'node_features': node_features,
            'edge_features': None,
            'batch':         batch,
            'edge_src':      edge_index[0],
            'edge_dst':      edge_index[1],
        }

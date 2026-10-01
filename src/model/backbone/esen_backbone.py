import os
import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from model.backbone.base import BackboneBase


class ESENBackbone(BackboneBase):
    """EquiformerV2 backbone (Meta's fairchem library, EquiformerV2Backbone).

    Install: pip install 'fairchem-core==1.10.0' --no-deps

    EquiformerV2 uses SO(3) equivariant representations (no parity distinction)
    with sphere_channels per angular momentum l.  Output node embeddings have
    shape [N_atoms, num_coefficients, sphere_channels] in 'mul_ir' layout where
    num_coefficients = sum(2l+1 for l in 0..lmax).

    We flatten this to [N_atoms, num_coefficients * sphere_channels] and project
    to an e3nn-compatible Irreps with parity assigned as:
        l=0 → 0e,  l=1 → 1o,  l=2 → 2e,  l=3 → 3o,  l=4 → 4e  (alternating)

    This parity convention matches the physical properties:
        dipole=1o,  quadrupole=2e,  polarizability=2e.

    Args:
        num_species      : number of element types (max atomic number to expect)
        device           : torch device
        r_max            : cutoff radius in Angstrom (default 6.0)
        max_neighbors    : max edges per atom (default 20)
        num_layers       : number of transformer blocks (default 4)
        sphere_channels  : channels per angular momentum l (default 128)
        attn_hidden_channels : hidden channels in attention (default 64)
        num_heads        : attention heads (default 8)
        lmax_list        : list of max l per resolution (default [4])
        mmax_list        : list of max m per resolution, ≤ lmax (default: same as lmax_list)
    """

    def __init__(
        self,
        num_species: int,
        device,
        r_max: float = 6.0,
        max_neighbors: int = 20,
        num_layers: int = 4,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 64,
        num_heads: int = 8,
        lmax_list=None,
        mmax_list=None,
        pretrain_checkpoint: str = '',
        norm_type: str = 'rms_norm_sh',
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        use_gate_act: bool = True,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        weight_init: str = 'uniform',
        activation_checkpoint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if lmax_list is None:
            lmax_list = [4]
        if mmax_list is None:
            mmax_list = list(lmax_list)

        try:
            from fairchem.core.models.equiformer_v2.equiformer_v2 import (
                EquiformerV2Backbone,
            )
        except ImportError:
            raise ImportError(
                "fairchem-core is required for ESENBackbone.\n"
                "Install with: pip install 'fairchem-core==1.10.0' --no-deps"
            )

        self.device = device
        self.r_max = r_max
        self._lmax = max(lmax_list)
        self._sphere_channels = sphere_channels

        model_kwargs = dict(
            use_pbc=False,
            use_pbc_single=False,
            regress_forces=False,
            otf_graph=True,
            max_neighbors=max_neighbors,
            max_radius=r_max,
            max_num_elements=max(num_species, 100),  # cover common element range
            num_layers=num_layers,
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_hidden_channels,
            attn_value_channels=sphere_channels // num_heads,
            ffn_hidden_channels=sphere_channels * 2,
            norm_type=norm_type,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            grid_resolution=None,
            num_sphere_samples=128,
            edge_channels=128,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function='gaussian',
            num_distance_basis=512,
            attn_activation='scaled_silu',
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            ffn_activation='scaled_silu',
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
            drop_path_rate=drop_path_rate,
            proj_drop=proj_drop,
            weight_init=weight_init,
            activation_checkpoint=activation_checkpoint,
        )
        model_kwargs.update(kwargs)
        self._model = EquiformerV2Backbone(**model_kwargs)

        # Optionally load pretrained backbone weights from a fairchem checkpoint.
        # pretrain_checkpoint: fairchem registry name (e.g. 'eSEN-30M-OAM') or local .pt path.
        if pretrain_checkpoint:
            self._load_pretrained(pretrain_checkpoint)

        # Assign alternating parities: l=0→0e, l=1→1o, l=2→2e, ...
        _parity = {l: 1 if l % 2 == 0 else -1 for l in range(self._lmax + 1)}
        irreps_list = []
        for l in range(self._lmax + 1):
            p = _parity[l]
            p_str = 'e' if p == 1 else 'o'
            irreps_list.append(f'{sphere_channels}x{l}{p_str}')
        self.node_out_irreps = Irreps('+'.join(irreps_list))
        self.edge_out_irreps = None  # EquiformerV2 does not expose per-edge features

    @staticmethod
    def _fairchem_coeff_to_e3nn_mul_layout(
        emb: torch.Tensor, lmax: int, sphere_channels: int
    ) -> torch.Tensor:
        """Map Fairchem SO3_Embedding layout to e3nn Irreps flatten order.

        Fairchem stores ``[N, (lmax+1)^2, sphere_channels]``: index 1 is spherical
        coefficients (grouped by l with ``(2l+1)`` rows per block), index 2 is
        ``sphere_channels`` (not batch — batch is ``N``).

        Rotations in Equiformer use ``bmm(W, embedding)`` on that coefficient
        axis, i.e. **left** multiply ``W @ v`` for each length-``(2l+1)`` column
        ``v``, same ``W`` for every channel — see ``SO3_Rotation.rotate``. That
        matches e3nn's column representation; flattened **row** features satisfy
        ``r' = r @ D.T`` for the same ``D = Irreps.D_from_matrix(R)``.

        e3nn ``Irreps('Cx0e+Cx1o+...')`` uses **multiplicity-major** flattening:
        for each l, ``C`` contiguous ``(2l+1)`` blocks (MACE / internal use the
        same). Implemented by ``[N, d, C] -> [N, C, d] -> reshape(C*d)`` per l.

        **Do not** use ``emb.reshape(N, -1)``: that is coefficient-major
        (interleaves m with channel) and mismatches ``node_out_irreps`` vs
        ``D(R)`` by O(1), independent of floating-point noise.
        """
        n_atoms, n_coeff, c = emb.shape
        assert c == sphere_channels and n_coeff == (lmax + 1) ** 2, (
            f'Expected emb [N, {(lmax + 1) ** 2}, {sphere_channels}], got {emb.shape}'
        )
        parts = []
        offset = 0
        for l in range(lmax + 1):
            d = 2 * l + 1
            block = emb[:, offset : offset + d, :]  # [N, d, C]
            # [N, d, C] -> [N, C, d] -> [N, C*d] (mul-major for this l)
            parts.append(block.transpose(1, 2).reshape(n_atoms, sphere_channels * d))
            offset += d
        assert offset == n_coeff
        return torch.cat(parts, dim=-1)

    def _load_pretrained(self, pretrain_checkpoint: str) -> None:
        """Load backbone weights from a fairchem checkpoint.

        Args:
            pretrain_checkpoint: local .pt file path OR a fairchem registry name
                                 (e.g. 'eSEN-30M-OAM', 'EquiformerV2-31M-OMAT24').
        """
        if os.path.isfile(pretrain_checkpoint):
            ckpt_path = pretrain_checkpoint
        else:
            from fairchem.core.models.model_registry import model_name_to_local_file
            cache_dir = os.path.expanduser('~/.cache/fairchem_checkpoints')
            print(f'[ESENBackbone] Downloading pretrained checkpoint: {pretrain_checkpoint}')
            ckpt_path = model_name_to_local_file(pretrain_checkpoint, local_cache=cache_dir)

        print(f'[ESENBackbone] Loading pretrained weights from: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # fairchem saves weights under 'state_dict' key; fall back to raw dict
        sd = ckpt.get('state_dict', ckpt)

        # Strip DDP 'module.' prefix if present
        sd = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in sd.items()}

        result = self._model.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(result.unexpected_keys)
        print(f'[ESENBackbone] Pretrained weights loaded: {n_loaded} matched, '
              f'{len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected')
        if result.missing_keys:
            print(f'  Missing (first 5): {result.missing_keys[:5]}')
        if result.unexpected_keys:
            print(f'  Unexpected (first 5): {result.unexpected_keys[:5]}')

    def forward(self, minibatch: dict) -> dict:
        """Adapt minibatch to fairchem Batch format and run EquiformerV2Backbone.

        Requires torch_geometric (pip install torch-geometric).
        """
        try:
            from torch_geometric.data import Data, Batch
        except ImportError:
            raise ImportError(
                'torch-geometric is required for ESENBackbone. '
                'Install with: pip install torch-geometric'
            )

        pos   = minibatch['pos']    # [N_atoms, 3]
        f_in  = minibatch['f_in']  # [N_atoms, num_species] one-hot
        batch = minibatch['batch'] # [N_atoms]

        # Map one-hot → atomic number index (1-based)
        atomic_numbers = torch.argmax(f_in, dim=1).long() + 1

        n_mol  = int(batch.max().item()) + 1
        counts = torch.bincount(batch, minlength=n_mol)
        offsets = torch.zeros(n_mol + 1, dtype=torch.long, device=batch.device)
        offsets[1:] = torch.cumsum(counts, dim=0)

        data_list = []
        for i in range(n_mol):
            sl = slice(offsets[i].item(), offsets[i + 1].item())
            d = Data(
                pos=pos[sl],
                atomic_numbers=atomic_numbers[sl],
                natoms=counts[i:i+1],
            )
            data_list.append(d)

        batched = Batch.from_data_list(data_list).to(pos.device)
        # Batch.from_data_list does not set natoms correctly as a 1D tensor,
        # so set it explicitly.
        batched.natoms = counts

        result = self._model(batched)
        # result['node_embedding'] is SO3_Embedding with .embedding [N, C_coeffs, C_sphere]
        emb = result['node_embedding'].embedding   # [N_atoms, num_coeffs, sphere_channels]

        # Coefficient-major [N, (2l+1) blocks, C] must become mul-major for node_out_irreps /
        # PropertyHead (reshape(-1) alone interleaves m and channel incorrectly).
        node_features = self._fairchem_coeff_to_e3nn_mul_layout(
            emb, self._lmax, self._sphere_channels
        )

        # Rebuild edge_index from the graph computed inside EquiformerV2
        graph = result['graph']
        edge_index = graph.edge_index  # [2, E]

        return {
            'node_features': node_features,
            'edge_features': None,
            'batch':         batch,
            'edge_src':      edge_index[0],
            'edge_dst':      edge_index[1],
        }

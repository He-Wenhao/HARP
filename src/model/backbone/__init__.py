"""Backbone factory for direct property prediction ablation studies."""

from model.backbone.base import BackboneBase


def _coerce_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_backbone(
    backbone_type: str,
    backbone_config: dict,
    irreps,
    device,
    nodeRDM_flag: bool,
) -> BackboneBase:
    """Instantiate the requested backbone.

    Args:
        backbone_type   : 'internal' | 'mace' | 'esen'
        backbone_config : dict of backbone hyperparameters (from [backbone] config section,
                          with keys prefixed 'backbone_' stripped by load_params)
        irreps          : Irreps_build instance (always needed to get num_species)
        device          : torch device
        nodeRDM_flag    : whether to use nodeRDM (internal backbone only)

    Returns:
        BackboneBase subclass instance.
    """
    bt = backbone_type.lower().strip()

    if bt == 'internal':
        from model.backbone.internal import InternalBackbone
        return InternalBackbone(
            irreps=irreps,
            device=device,
            nodeRDM_flag=nodeRDM_flag,
            ele_emb=backbone_config.get('ele_emb', 10),
            dim_embR=backbone_config.get('dim_embR', 128),
            num_layers=int(backbone_config.get('num_layers', 3)),
        )

    elif bt == 'mace':
        from model.backbone.mace_backbone import MACEBackbone
        return MACEBackbone(
            num_species=irreps.get_input_irreps(),
            device=device,
            r_max=backbone_config.get('r_max', 5.0),
            num_radial_basis=backbone_config.get('num_radial_basis', 8),
            max_ell=backbone_config.get('max_ell', 3),
            num_interactions=backbone_config.get('num_interactions', 2),
            hidden_irreps=backbone_config.get('hidden_irreps', '64x0e+64x1o+64x2e'),
            MLP_irreps=backbone_config.get('MLP_irreps', '16x0e'),
            correlation=backbone_config.get('correlation', 3),
            atomic_numbers=backbone_config.get('atomic_numbers', None),
        )

    elif bt == 'esen':
        from model.backbone.esen_backbone import ESENBackbone
        lmax_raw = backbone_config.get('lmax_list', 4)
        lmax_list = [int(lmax_raw)] if isinstance(lmax_raw, (int, float)) else list(lmax_raw)
        mmax_raw = backbone_config.get('mmax_list', None)
        if mmax_raw is None:
            mmax_list = list(lmax_list)
        elif isinstance(mmax_raw, (int, float)):
            mmax_list = [int(mmax_raw)] * len(lmax_list)
        else:
            mmax_list = list(mmax_raw)
        return ESENBackbone(
            num_species=irreps.get_input_irreps(),
            device=device,
            r_max=backbone_config.get('r_max', 6.0),
            max_neighbors=backbone_config.get('max_neighbors', 20),
            num_layers=backbone_config.get('num_layers', 4),
            sphere_channels=backbone_config.get('sphere_channels', 128),
            attn_hidden_channels=backbone_config.get('attn_hidden_channels', 64),
            num_heads=backbone_config.get('num_heads', 8),
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            pretrain_checkpoint=backbone_config.get('pretrain_checkpoint', ''),
            norm_type=backbone_config.get('norm_type', 'rms_norm_sh'),
            use_s2_act_attn=_coerce_bool(backbone_config.get('use_s2_act_attn', False), False),
            use_attn_renorm=_coerce_bool(backbone_config.get('use_attn_renorm', True), True),
            use_gate_act=_coerce_bool(backbone_config.get('use_gate_act', True), True),
            use_grid_mlp=_coerce_bool(backbone_config.get('use_grid_mlp', False), False),
            use_sep_s2_act=_coerce_bool(backbone_config.get('use_sep_s2_act', True), True),
            alpha_drop=backbone_config.get('alpha_drop', 0.1),
            drop_path_rate=backbone_config.get('drop_path_rate', 0.05),
            proj_drop=backbone_config.get('proj_drop', 0.0),
            weight_init=backbone_config.get('weight_init', 'uniform'),
            activation_checkpoint=_coerce_bool(backbone_config.get('activation_checkpoint', False), False),
        )

    else:
        raise ValueError(
            f"Unknown backbone_type: '{backbone_type}'. "
            f"Expected one of: 'internal', 'mace', 'esen'."
        )


__all__ = ['BackboneBase', 'build_backbone']

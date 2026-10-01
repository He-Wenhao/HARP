import torch.nn as nn
from model.backbone.base import BackboneBase
from model.property_head import PropertyHead


class DirectPropertyModel(nn.Module):
    """Compose a BackboneBase with a PropertyHead for direct property prediction.

    The forward() return dict is identical to V_theta.forward(), making this a
    drop-in replacement wherever train.py calls self.model(minibatch).

    Usage:
        backbone = build_backbone('mace', cfg, irreps, device, nodeRDM_flag)
        head     = PropertyHead(backbone.node_out_irreps, backbone.edge_out_irreps)
        model    = DirectPropertyModel(backbone, head).to(device)
    """

    def __init__(self, backbone: BackboneBase, head: PropertyHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, minibatch: dict) -> dict:
        backbone_out = self.backbone(minibatch)
        out = self.head(backbone_out)
        out['pos'] = minibatch['pos']
        out['f_in'] = minibatch['f_in']
        # Thole relay-matrix alpha is assembled HERE, inside forward(), so every
        # model parameter (head.alpha_log_base, head.thole_a_logit) is used within
        # the DDP-wrapped forward. Assembling it in the trainer instead made DDP
        # mark alpha_log_base "ready twice" (param used outside forward). pos is
        # the same frame the backbone consumed -> the relay tensor T and the
        # equivariant per-atom alpha_i are frame-consistent by construction.
        if self.head.alpha_head in ('thole', 'thole_aniso', 'relay_sqe'):
            from model.thole import molecular_alpha
            eta = kappa = None
            if self.head.alpha_head == 'relay_sqe':
                eta = self.head.eta_from_raw(out['atom_hardness'][:, 0])   # [N] bounded
                kappa = self.head.kappa_value()
            out['alpha_mol'] = molecular_alpha(
                out['atom_alpha'], minibatch['f_in'], minibatch['pos'],
                self.head.thole_a_value(), self.head.alpha_log_base,
                minibatch['batch'], self.head.alpha_head, eta=eta, kappa=kappa)
        return out

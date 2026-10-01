# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:42:41 2023

@author: 17000
"""

from model import V_theta
from model.irreps2mats import to_mat
from model.hamiltonian_model import HamiltonianModel, HamiltonianHead
from train.solve_H import state_solver
import model.Fock2obs as obs_fn
from util_basis.Irreps import Irreps_build
import numpy as np
import torch
import math
from model.xyz2graph import sampler;
from torch.nn.parallel import DistributedDataParallel as DDP;
from train.logger import start_iter_nvml, record_iter_nvml
import os;
import pdb
import torch.distributed as dist
import time
from util_basis.util_fns import mse_loss_over_list

# ── Spherical-to-Cartesian conversion for rank-2 symmetric tensors ────────────
# Input v: [..., 6] where v[...,0] = l=0e (trace/√3),
#   v[...,1:6] = l=2e components in e3nn order m = -2,-1,0,1,2
#   (Y_2^{-2}~xy, Y_2^{-1}~yz, Y_2^{0}~(2z²-x²-y²), Y_2^{1}~xz, Y_2^{2}~(x²-y²))
_S2 = math.sqrt(2); _S3 = math.sqrt(3); _S6 = math.sqrt(6)

def _sph_to_cart(v):
    """v: [..., 6] → (xx, yy, zz, xy, yz, xz) each [..., 1]"""
    s = v[..., 0:1]; d = v[..., 1:]   # d: [..., 5]  m=-2,-1,0,1,2
    xx = s/_S3 - d[...,2:3]*_S6/6 + d[...,4:5]/_S2
    yy = s/_S3 - d[...,2:3]*_S6/6 - d[...,4:5]/_S2
    zz = s/_S3 + d[...,2:3]*_S6/3
    xy = d[...,0:1]/_S2
    yz = d[...,1:2]/_S2
    xz = d[...,3:4]/_S2
    return xx, yy, zz, xy, yz, xz

# e3nn l=1o component ordering: m=-1,0,1  →  y, z, x
_DIPOLE_XYZ_IDX = (2, 0, 1)   # x=col2, y=col0, z=col1

TEST_TIME = False

class trainer():

    def __init__(self, device, nodeRDM_flag) -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        self.device = device
        #self.sampler = sampler(data_in, labels, device);
        #self.n_molecules = len(data_in);
        #self.op_matrices = op_matrices; # dictionary of ['x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'xz', 'yz', 'F']
        #self.charge_matrices = charge_matrices
        self.nodeRDM_flag = nodeRDM_flag;
        self._iter_times = [];
        self.loss_dropout_enabled = False
        self.loss_dropout_default_rate = 0.0
        self.loss_dropout_rates = {}
        # Aggregation mode for the E_gap head. Default 'sum' preserves the
        # historical (v7_bp86 and earlier) behaviour — backwards compatible.
        # 'mean' makes the predicted gap intensive (size-independent in the
        # large-N limit), which is the physically correct choice for an
        # intensive property like HOMO-LUMO gap. See src/train/train.py's
        # direct_predict_properties for the actual aggregation call.
        self.egap_pooling = 'sum'
        # E_gap rescale mode for the hamiltonian head (G ablation):
        #   'affine'   : Eg = raw_gap*(1+G0) + G1   (default, current h_v8)
        #   'additive' : Eg = raw_gap + G1          (drop multiplicative rescale)
        #   'none'     : Eg = raw_gap               (no learned correction)
        self.gap_rescale = 'affine'
        # alpha head mode: 'sum' (legacy local sum + DFT baseline) or 'thole'
        # (Applequist/Thole relay matrix, predicts full alpha, no baseline).
        self.alpha_head = 'sum'
        # Prediction head mode: 'direct' (per-property heads + sum/mean pooling)
        # or 'hamiltonian' (predict a Fock correction V, diagonalize H=h+V, and
        # read observables off the eigenstates via Fock2obs — size-extensive).
        self.head_type = 'direct'
        self.ham_mode = False
        self.to_mat = None

    def build_irreps(self, element_list = ['H','C','N','O','F']):
        
        self.irreps = Irreps_build(element_list);
        self.irreps.generate_irreps();

        return None;

    def _configure_head_type(self, head_type):
        """Record the prediction-head mode and (for the Hamiltonian path) build
        the irreps->matrix converter shared by inference()/predict_properties()."""
        self.head_type = str(head_type or 'direct').strip().lower()
        self.ham_mode = (self.head_type == 'hamiltonian')
        if self.ham_mode:
            self.to_mat = to_mat(self.device, self.irreps)

    def _build_hamiltonian_inner(self, backbone_type, backbone_config, head_scaling=0.2):
        """Backbone (node features) + HamiltonianHead (Fock correction blocks)."""
        from model.backbone import build_backbone
        cfg      = backbone_config or {}
        backbone = build_backbone(backbone_type, cfg, self.irreps, self.device, self.nodeRDM_flag)
        head     = HamiltonianHead(backbone.node_out_irreps, self.irreps, self.device,
                                   scaling=head_scaling)
        return HamiltonianModel(backbone, head)

    def build_model(self, scaling={'V': 0.2, 'T': 0.01}, element_list=['H', 'C', 'N', 'O', 'F'],
                    backbone_type='internal', backbone_config=None, alpha_head='sum',
                    head_type='direct'):
        self.build_irreps(element_list)
        self.scaling = scaling
        self.alpha_head = str(alpha_head).strip().lower()
        self._configure_head_type(head_type)
        if self.ham_mode:
            self.model = self._build_hamiltonian_inner(backbone_type, backbone_config)
        elif backbone_type == 'internal' and not backbone_config:
            # Backward-compatible path: use V_theta directly
            self.model = V_theta(self.device, self.irreps, nodeRDM_flag=self.nodeRDM_flag)
        else:
            from model.backbone import build_backbone
            from model.property_head import PropertyHead
            from model.direct_model import DirectPropertyModel
            cfg      = backbone_config or {}
            backbone = build_backbone(backbone_type, cfg, self.irreps, self.device, self.nodeRDM_flag)
            head     = PropertyHead(backbone.node_out_irreps, backbone.edge_out_irreps,
                                    alpha_head=self.alpha_head, element_list=element_list)
            self.model = DirectPropertyModel(backbone, head)
        self.model = self.model.to(self.device)

    def build_ddp_model(self, scaling={'V': 0.2, 'T': 0.01}, element_list=['H', 'C', 'N', 'O', 'F'],
                        backbone_type='internal', backbone_config=None, alpha_head='sum',
                        head_type='direct'):
        self.build_irreps(element_list)
        self.scaling = scaling
        self.alpha_head = str(alpha_head).strip().lower()
        self._configure_head_type(head_type)
        if self.ham_mode:
            inner = self._build_hamiltonian_inner(backbone_type, backbone_config)
        elif backbone_type == 'internal' and not backbone_config:
            inner = V_theta(self.device, self.irreps, nodeRDM_flag=self.nodeRDM_flag)
        else:
            from model.backbone import build_backbone
            from model.property_head import PropertyHead
            from model.direct_model import DirectPropertyModel
            cfg      = backbone_config or {}
            backbone = build_backbone(backbone_type, cfg, self.irreps, self.device, self.nodeRDM_flag)
            head     = PropertyHead(backbone.node_out_irreps, backbone.edge_out_irreps,
                                    alpha_head=self.alpha_head, element_list=element_list)
            inner    = DirectPropertyModel(backbone, head)
        self.model = DDP(inner.to(self.device), device_ids=[self.device], find_unused_parameters=True)

    def build_optimizer(self, lr=10**-3):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr);

    def configure_loss_dropout(self, enabled=False, default_rate=0.0, rates=None):
        self.loss_dropout_enabled = bool(enabled)
        self.loss_dropout_default_rate = float(default_rate)
        self.loss_dropout_rates = dict(rates or {})

    def configure_egap_pooling(self, pooling: str = 'sum'):
        """Choose the aggregation mode for the E_gap head.

        'sum'  : per-atom Egap features are summed → extensive (legacy).
        'mean' : per-atom Egap features are averaged → intensive (new
                 default for v8+; physically correct for HOMO-LUMO gap).
        """
        pooling = str(pooling).strip().lower()
        if pooling not in ('sum', 'mean'):
            raise ValueError(f"egap_pooling must be 'sum' or 'mean', got {pooling!r}")
        self.egap_pooling = pooling

    def configure_gap_rescale(self, mode: str = 'affine'):
        """E_gap rescale for the hamiltonian head (G ablation).
        'affine': raw_gap*(1+G0)+G1 ; 'additive': raw_gap+G1 ; 'none': raw_gap."""
        mode = str(mode).strip().lower()
        if mode not in ('affine', 'additive', 'none'):
            raise ValueError(f"gap_rescale must be affine|additive|none, got {mode!r}")
        self.gap_rescale = mode

    def load(self, filename):

        if(os.path.exists(filename)):
            try:
                self.model.load_state_dict(torch.load(filename));
            except:
                try:
                    res = torch.load(filename);
                    for key in list(res.keys()):
                        res[key[7:]] = res[key];
                        del res[key];
                    self.model.load_state_dict(res);
                except:
                    res = torch.load(filename);
                    for key in list(res.keys()):
                        res['module.'+key] = res[key];
                        del res[key];
                    self.model.load_state_dict(res);
        else:
            raise ValueError("file not exists:"+filename)

    def save(self, filename):

        torch.save(self.model.state_dict(), filename);

    def save_checkpoint(self, filename, optimizer, scheduler, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, filename)

    def load_checkpoint(self, filename):
        """Load a full checkpoint. Returns (optimizer_state_dict, scheduler_state_dict, start_epoch).
        If the file is a legacy plain-weights file, returns (None, None, 0)."""
        ckpt = torch.load(filename, map_location='cpu')
        if 'model_state_dict' not in ckpt:
            # legacy plain weight file — load as before, epoch unknown
            self.load(filename)
            return None, None, 0
        self.model.load_state_dict(ckpt['model_state_dict'])
        return ckpt['optimizer_state_dict'], ckpt['scheduler_state_dict'], ckpt['epoch'] + 1

    def inference(self, minibatch):
        
        # Fine-grained timing for inference: model forward and transformer
        rank = int(os.environ.get("RANK", 0))
        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
        _sync(); t0 = time.time();
        V_raw = self.model(minibatch);
        _sync(); t1 = time.time();

        mats_dict = self.to_mat.raw_to_mat(V_raw,minibatch);
        _sync(); t2 = time.time();

        mats_dict['V'] *= self.scaling['V'];
        mats_dict['T'] *= self.scaling['T'];

        if TEST_TIME and rank == 0:
            print(f"[inference] model_forward={t1-t0:.6f}s, transform={t2-t1:.6f}s, total={t2-t0:.6f}s")
        return mats_dict;
    
    # def build_loss_bk(self, loss_calculator, labels, ind, op_names,op_matrices,charge_matrices):
    #     lc = loss_calculator
    #     L_grads = {};
    #     L_ave = np.zeros(len(op_names));
        
    #     # timing helpers
    #     def _sync():
    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
    #     rank = int(os.environ.get("RANK", 0)) 
        
    #     t_ops = { 'V':0.0, 'reg':0.0, 'E':0.0, 'atomic_charge':0.0, 'E_gap':0.0, 'bond_order':0.0,
    #               'alpha':0.0, 'F':0.0, 'other':0.0 };        
    #     for i, op_name in enumerate(op_names):
            
    #         if(op_name == 'V'):
    #             _sync(); t0 = time.time();
    #             #Lv = loss_calculator.V_loss();
    #             Lv = torch.mean(loss_calculator.V**2)
    #             _sync(); t1 = time.time();
    #             t_ops['V'] += (t1 - t0);
    #             L_grads['V'] = Lv;
    #             L_ave[i] += float(Lv.detach().cpu());
                
    #         elif(op_name == 'reg'):
    #             _sync(); t0 = time.time();
    #             L_reg = sum([p.square().sum() for p in self.model.parameters()])
    #             _sync(); t1 = time.time();
    #             t_ops['reg'] += (t1 - t0);
    #             L_grads[op_name] = L_reg;
    #             L_ave[i] += float(L_reg.detach().cpu());
                
    #         elif(op_name == 'E'):
    #             _sync(); t0 = time.time();
    #             # Cast labels to match loop tensors' dtype/device for both implementations
    #             E_labels_val = labels['Ee']
    #             if isinstance(E_labels_val, (list, tuple)):
    #                 E_labels_t_for_both = torch.tensor(E_labels_val, dtype=loss_calculator.co[0].dtype, device=loss_calculator.co[0].device)
    #             else:
    #                 E_labels_t_for_both = E_labels_val.to(device=loss_calculator.co[0].device, dtype=loss_calculator.co[0].dtype)
    #             #LE = loss_calculator.E_loss(E_labels_t_for_both);
    #             E_l =  obs_fn.Energy()(lc.H, lc.eo, lc.co, lc.ne)
    #             LE = mse_loss_over_list(E_l, E_labels_t_for_both);
    #             #LE = measure_memory(
    #             #    lambda: loss_calculator.E_loss(E_labels_t_for_both)
    #             #)

    #             _sync(); t1 = time.time();
    #             t_ops['E'] += (t1 - t0);
    #             L_grads['E'] = LE;
    #             L_ave[i] += float(LE.detach().cpu());

    #         elif(op_name == 'atomic_charge'):
    #             _sync(); t0 = time.time();
                
    #             C_mat = [charge_matrices[k1] for k1 in ind];
    #             #LC = loss_calculator.C_loss(C, C_mat);
    #             C_l = obs_fn.Charge()(lc.H, lc.eo, lc.co, lc.ne, C_mat, lc.cv, lc.epsilon_ij)
    #             LC = mse_loss_over_list(C_l, labels['atomic_charge']);
    #             _sync(); t1 = time.time();
    #             t_ops['atomic_charge'] += (t1 - t0);
    #             L_grads['atomic_charge'] = LC;
    #             L_ave[i] += float(LC.detach().cpu());

    #         elif(op_name == 'E_gap'):
    #             _sync(); t0 = time.time();
    #             #Egap = labels['E_gap'];
    #             #Lgap_grad = loss_calculator.Eg_loss(Egap);
    #             Eg_l_raw = obs_fn.EnergyGap()(lc.H, lc.eo, lc.co, lc.ev, lc.cv, lc.ne)
    #             Eg_l = [Eghat_raw * (1 + lc.G[i,0]) + lc.G[i,1] for i, Eghat_raw in enumerate(Eg_l_raw)]
    #             LEg = mse_loss_over_list(Eg_l, labels['E_gap']);
    #             _sync(); t1 = time.time();
    #             t_ops['E_gap'] += (t1 - t0);
    #             L_grads['E_gap'] = LEg;
    #             L_ave[i] += float(LEg.detach().cpu());

    #         elif(op_name == 'bond_order'):
    #             _sync(); t0 = time.time();
    #             B_mat = [charge_matrices[k1] for k1 in ind];
    #             mask_l = [(B_label != 0) for B_label in labels['B']]
    #             B_l = obs_fn.Bond()(lc.H, lc.eo, lc.co, lc.ne, B_mat, lc.cv, lc.epsilon_ij)
    #             LB = mse_loss_over_list(B_l, labels['B'],mask=mask_l);
    #             _sync(); t1 = time.time();
    #             t_ops['bond_order'] += (t1 - t0);
    #             L_grads['bond_order'] = LB;
    #             L_ave[i] += float(LB.detach().cpu());
        
    #         elif(op_name == 'alpha'):
    #             _sync(); t0 = time.time();
    #             #alpha = labels['alpha'];
    #             r_mats = [torch.stack([op_matrices[ii][key] for key in ['x','y','z']]) \
    #                        for ii in ind];
    #             T_mats = [t for t in lc.T];
    #             bare_polar_l = obs_fn.BarePolarizability()(lc.H, lc.eo, lc.co, lc.ne, r_mats, lc.cv, lc.epsilon_ij, lc.epsilon_jk, lc.epsilon_ik)
    #             polar_l = obs_fn.Polarizability()(bare_polar_l, T_mats)
    #             Lpolar = mse_loss_over_list(polar_l, labels['alpha']);
    #             _sync(); t1 = time.time();
    #             t_ops['alpha'] += (t1 - t0);
    #             L_grads['alpha'] = Lpolar;
    #             L_ave[i] += float(Lpolar.detach().cpu());
 
 
    #         else:
    #             _sync(); t0 = time.time();
    #             O = labels[op_name]
    #             O_mat = [op_matrices[k1][op_name] for k1 in ind];
    #             # LO = loss_calculator.O_loss(O, O_mat);
    #             M_l = obs_fn.Multipole()(lc.H, lc.eo, lc.co, lc.ne, O_mat, lc.cv, lc.epsilon_ij)
    #             LO = mse_loss_over_list(M_l, O);
    #             _sync(); t1 = time.time();
    #             t_ops['other'] += (t1 - t0);
    #             L_grads[op_name] = LO;
    #             L_ave[i] += float(LO.detach().cpu());
    #     # Optional batched-vs-original check
        
    #     L = sum([op_names[key]*L_grads[key] \
    #                 for key in op_names])/len(ind) ;
    #     L_total = sum([op_names[list(op_names)[i]]*L_ave[i] \
    #                 for i in range(len(op_names))])/len(ind) ;

    #     # print per-op breakdown for build_loss (rank 0 only)
    #     if TEST_TIME and rank == 0:
    #         total_bl = sum(t_ops.values());
    #         # only show ops that were present or non-zero
    #         parts = []
    #         for k in ['V','reg','E','atomic_charge','E_gap','bond_order','alpha','F','other']:
    #             if t_ops[k] > 0:
    #                 parts.append(f"{k}={t_ops[k]:.6f}s")
    #         print("[build_loss_detail] " + ", ".join(parts) + f" | total={total_bl:.6f}s")


    #     return L, L_ave, L_total;

    def predict_properties(self, state_calculator, ind, op_names, op_matrices=[],charge_matrices=[], labels=None):
        lc = state_calculator
        property_dic = {};
        # Total-energy baseline for delta-learning (v8 convention): the BP86
        # electronic energy E_BP86 = energy - E_nn, loaded as labels['baseline_E'].
        # Energy is then E = E_BP86 + ΔE_net (NN per-atom scalar, summed). This is
        # physically meaningful, unlike summing occupied orbital energies (which
        # double-counts Hartree/XC). Falls back to the orbital-energy sum e_h only
        # when no baseline is present (delta_learning off).
        baseline_E = labels.get('baseline_E') if labels is not None else None
        # timing helpers
        def _sync(if_test = TEST_TIME):
            if torch.cuda.is_available() and if_test:
                torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
        rank = int(os.environ.get("RANK", 0)) 
        
        t_ops = { 'V':0.0, 'reg':0.0, 'E':0.0, 'atomic_charge':0.0, 'E_gap':0.0, 'bond_order':0.0,
                  'alpha':0.0, 'F':0.0, 'other':0.0 }; 
        for i, op_name in enumerate(op_names):
            
            _sync(); t0 = time.time();
            if(op_name == 'V'):
                property_dic[op_name] = [v for v in lc.V]
            elif(op_name == 'E'):
                # v8-style delta-learning: BP86 total electronic energy + NN delta.
                if baseline_E is not None:
                    property_dic[op_name] = [baseline_E[i] + lc.delta_E[i]
                                             for i in range(len(lc.delta_E))];
                else:
                    # Fallback (no baseline): orbital-energy sum 2*Σε_occ of h.
                    property_dic[op_name] = [lc.e_h[i] + lc.delta_E[i]
                                             for i in range(len(lc.e_h))];
            elif(op_name == 'atomic_charge'):
                C_mat = [charge_matrices[k1] for k1 in ind];
                property_dic[op_name] = obs_fn.Charge()(lc.H, lc.eo, lc.co, lc.ne, C_mat, lc.cv, lc.epsilon_ij);
            elif(op_name == 'E_gap'):
                Eg_l_raw = obs_fn.EnergyGap()(lc.H, lc.eo, lc.co, lc.ev, lc.cv, lc.ne)
                if self.gap_rescale == 'none':
                    Eg_l = [Eghat_raw for Eghat_raw in Eg_l_raw]
                elif self.gap_rescale == 'additive':
                    Eg_l = [Eghat_raw + lc.G[i,1] for i, Eghat_raw in enumerate(Eg_l_raw)]
                else:  # 'affine' (default)
                    Eg_l = [Eghat_raw * (1 + lc.G[i,0]) + lc.G[i,1] for i, Eghat_raw in enumerate(Eg_l_raw)]
                property_dic[op_name] = Eg_l;
            elif(op_name == 'bond_order'):
                B_mat = [charge_matrices[k1] for k1 in ind];
                property_dic[op_name] = obs_fn.Bond()(lc.H, lc.eo, lc.co, lc.ne, B_mat, lc.cv, lc.epsilon_ij);
            elif(op_name == 'alpha'):
                r_mats = [torch.stack([op_matrices[ii][key] for key in ['x','y','z']]) \
                           for ii in ind];
                T_mats = [t for t in lc.T];
                bare_polar_l = obs_fn.BarePolarizability()(lc.H, lc.eo, lc.co, lc.ne, r_mats, lc.cv, lc.epsilon_ij, lc.epsilon_jk, lc.epsilon_ik)
                polar_l = obs_fn.Polarizability()(bare_polar_l, T_mats)
                property_dic[op_name] = polar_l;
            elif (op_name in ['xx','yy','zz','xy','xz','yz']) or (op_name in ['x','y','z']):
                O_mat = [op_matrices[k1][op_name] for k1 in ind];
                property_dic[op_name] = obs_fn.Multipole()(lc.H, lc.eo, lc.co, lc.ne, O_mat, lc.cv, lc.epsilon_ij);
            else:
                raise ValueError("Unknown operator name:"+op_name)
            _sync(); t1 = time.time();
            t_ops[op_name] = (t1 - t0);
        if TEST_TIME and rank == 0:
            total_bl = sum(t_ops.values());
            # only show ops that were present or non-zero
            parts = []
            for k in ['V','reg','E','atomic_charge','E_gap','bond_order','alpha','F','other']:
                if t_ops[k] > 0:
                    parts.append(f"{k}={t_ops[k]:.6f}s")
            print("[build_loss_detail] " + ", ".join(parts) + f" | total={total_bl:.6f}s")

        return property_dic;
    
    def direct_predict_properties(self, raw, labels, op_names):
        """Aggregate per-atom/edge GNN outputs into per-molecule observable predictions.

        GNN predicts a correction (delta) relative to a delta-learning baseline
        (B3LYP, BP86, …) stored in labels as 'baseline_<op>' keys. If no
        baseline key is present the raw delta is used as the prediction,
        i.e. delta-learning from zero.

        raw      : dict returned by V_theta.forward (direct mode)
        labels   : batch_labels dict from sampler
        op_names : dict of {obs_name: weight}
        Returns property_dic compatible with weighted_mse_loss.
        """
        batch    = raw['batch']        # [N_atoms]
        edge_src = raw['edge_src']     # [N_edges]
        edge_dst = raw['edge_dst']     # [N_edges]
        n_mol    = int(batch.max().item()) + 1
        dev      = batch.device

        counts  = torch.bincount(batch, minlength=n_mol)
        offsets = torch.zeros(n_mol + 1, dtype=torch.long, device=dev)
        offsets[1:] = torch.cumsum(counts, dim=0)

        def _atom_sum(feat):
            return [feat[offsets[i]:offsets[i+1]].sum(0).squeeze(-1) for i in range(n_mol)]

        def _atom_mean(feat):
            return [feat[offsets[i]:offsets[i+1]].mean(0).squeeze(-1) for i in range(n_mol)]

        def _baseline(key):
            """Return list of delta-learning baselines for 'key', or None if absent."""
            return labels.get('baseline_' + key)

        def _add_baseline(deltas, b3, dtype=None):
            """Add scalar or tensor baseline to list of delta tensors."""
            if b3 is None:
                return deltas
            result = []
            for i, d in enumerate(deltas):
                v = b3[i]
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v, dtype=d.dtype, device=dev)
                else:
                    v = v.to(device=dev, dtype=d.dtype)
                result.append(v + d)
            return result

        property_dic = {}

        for op in op_names:
            if op == 'E':
                property_dic['E'] = _add_baseline(_atom_sum(raw['atom_E']), _baseline('E'))

            elif op == 'E_gap':
                # Pick aggregator based on trainer config. 'sum' (legacy) keeps
                # the v7_bp86 behaviour; 'mean' (v8+) gives an intensive
                # prediction that matches the physical scaling of a HOMO-LUMO
                # gap. The BP86 baseline is itself intensive, so adding a mean-
                # pooled delta keeps the final prediction intensive too.
                egap_agg = _atom_mean if self.egap_pooling == 'mean' else _atom_sum
                property_dic['E_gap'] = _add_baseline(
                    egap_agg(raw['atom_Egap']), _baseline('E_gap'))

            elif op == 'atomic_charge':
                deltas = [raw['atom_charge'][offsets[i]:offsets[i+1], 0] for i in range(n_mol)]
                property_dic['atomic_charge'] = _add_baseline(deltas, _baseline('atomic_charge'))

            elif op in ('x', 'y', 'z'):
                if 'dipole' not in property_dic:
                    mol_dip = [raw['atom_dipole'][offsets[i]:offsets[i+1]].sum(0)
                               for i in range(n_mol)]
                    property_dic['dipole'] = mol_dip
                col = _DIPOLE_XYZ_IDX[('x','y','z').index(op)]
                deltas = [d[col] for d in property_dic['dipole']]
                property_dic[op] = _add_baseline(deltas, _baseline(op))

            elif op in ('xx', 'yy', 'zz', 'xy', 'yz', 'xz'):
                if 'quad_cart' not in property_dic:
                    mol_q = [raw['atom_quad'][offsets[i]:offsets[i+1]].sum(0)
                             for i in range(n_mol)]
                    stacked = torch.stack(mol_q, dim=0)   # [n_mol, 6]
                    comps = _sph_to_cart(stacked)          # tuple of 6 tensors [n_mol, 1]
                    keys = ('xx','yy','zz','xy','yz','xz')
                    property_dic['quad_cart'] = {k: comps[j].squeeze(-1)
                                                 for j, k in enumerate(keys)}
                deltas = list(property_dic['quad_cart'][op])
                property_dic[op] = _add_baseline(deltas, _baseline(op))

            elif op == 'alpha':
                if self.alpha_head in ('thole', 'thole_aniso', 'relay_sqe'):
                    # Size-extensive relay matrix alpha, assembled inside the model
                    # forward (model/direct_model.py -> thole.molecular_alpha) so
                    # all parameter use stays within the DDP-wrapped forward. Full
                    # alpha, no DFT baseline; bounded per-atom alpha_i keeps the
                    # relay positive-definite (no polarization catastrophe).
                    property_dic['alpha'] = raw['alpha_mol']
                else:
                    mol_a = [raw['atom_alpha'][offsets[i]:offsets[i+1]].sum(0)
                             for i in range(n_mol)]
                    stacked = torch.stack(mol_a, dim=0)
                    xx, yy, zz, xy, yz, xz = _sph_to_cart(stacked)
                    xx = xx.squeeze(-1); yy = yy.squeeze(-1); zz = zz.squeeze(-1)
                    xy = xy.squeeze(-1); yz = yz.squeeze(-1); xz = xz.squeeze(-1)
                    delta_alpha = [
                        torch.stack([
                            torch.stack([xx[i], xy[i], xz[i]]),
                            torch.stack([xy[i], yy[i], yz[i]]),
                            torch.stack([xz[i], yz[i], zz[i]])
                        ]) for i in range(n_mol)]
                    property_dic['alpha'] = _add_baseline(delta_alpha, _baseline('alpha'))

            elif op == 'bond_order':
                n_total = counts.sum().item()
                B_full = torch.zeros(n_total, n_total, device=dev)
                bvals  = raw['edge_bond'][:, 0]
                B_full.index_put_((edge_src, edge_dst), bvals, accumulate=True)
                B_full = (B_full + B_full.t()) / 2
                b3_B = _baseline('B')
                property_dic['bond_order'] = [
                    (_add_baseline(
                        [B_full[offsets[i]:offsets[i+1], offsets[i]:offsets[i+1]]],
                        [b3_B[i]] if b3_B is not None else None))[0]
                    for i in range(n_mol)]

        property_dic.pop('dipole', None)
        property_dic.pop('quad_cart', None)
        return property_dic

    def _sample_loss_dropout_weights(self, op_names):
        """Sample per-property loss weights for training-time loss dropout.

        Returns a dict matching ``op_names`` where each value is the multiplier
        applied to that property's loss term. Inverted dropout scaling keeps the
        expected contribution unchanged for retained properties.
        """
        eff_weights = {op: float(w) for op, w in op_names.items()}
        if not self.loss_dropout_enabled:
            return eff_weights

        active_ops = []
        kept_any = False
        for op, base_w in eff_weights.items():
            if base_w == 0.0:
                eff_weights[op] = 0.0
                continue
            rate = float(self.loss_dropout_rates.get(op, self.loss_dropout_default_rate))
            rate = min(max(rate, 0.0), 1.0)
            keep_prob = 1.0 - rate
            if keep_prob <= 0.0:
                eff_weights[op] = 0.0
                active_ops.append((op, base_w, keep_prob))
                continue
            mask = float(torch.bernoulli(torch.tensor([keep_prob], device=self.device)).item())
            eff_weights[op] = base_w * mask / keep_prob
            active_ops.append((op, base_w, keep_prob))
            kept_any = kept_any or (mask > 0.0)

        # Avoid all-zero loss on a training step when dropout is enabled.
        if active_ops and not kept_any:
            forced_op, base_w, keep_prob = active_ops[int(torch.randint(len(active_ops), (1,), device=self.device).item())]
            if keep_prob > 0.0:
                eff_weights[forced_op] = base_w / keep_prob
            else:
                eff_weights[forced_op] = base_w
        return eff_weights

    def weighted_mse_loss(self,property_dic,labels,op_names, apply_loss_dropout=False):
        L_ave = np.zeros(len(op_names));
        L_dict = {};
        for op_name in op_names:
            if(op_name == 'V'):
                #Lv = torch.mean(property_dic[op_name]**2)
                label_V = [torch.zeros_like(v) for v in property_dic[op_name]]
                Lv = mse_loss_over_list(property_dic[op_name], label_V);
                L_dict[op_name] = Lv;
                #L_ave[i] += float(Lv.detach().cpu());
            elif(op_name == 'E'):
                LE = mse_loss_over_list(property_dic[op_name], labels['Ee']);
                L_dict[op_name] = LE;
                #L_ave[i] += float(LE.detach().cpu());
            elif(op_name == 'atomic_charge'):
                LC = mse_loss_over_list(property_dic[op_name], labels[op_name]);
                L_dict[op_name] = LC;
                #L_ave[i] += float(LC.detach().cpu());
            elif(op_name == 'E_gap'):
                LEg = mse_loss_over_list(property_dic[op_name], labels[op_name]);
                L_dict[op_name] = LEg;
                #L_ave[i] += float(LEg.detach().cpu());
            elif(op_name == 'bond_order'):
                mask_l = [(B_label != 0) for B_label in labels['B']]
                LB = mse_loss_over_list(property_dic[op_name], labels['B'],mask=mask_l);
                L_dict[op_name] = LB;
                #L_ave[i] += float(LB.detach().cpu());
            elif(op_name == 'alpha'):
                Lpolar = mse_loss_over_list(property_dic[op_name], labels[op_name]);
                L_dict[op_name] = Lpolar;
                #L_ave[i] += float(Lpolar.detach().cpu());
            elif (op_name in ['xx','yy','zz','xy','xz','yz']) or (op_name in ['x','y','z']):
                LO = mse_loss_over_list(property_dic[op_name], labels[op_name]);
                L_dict[op_name] = LO;
                #L_ave[i] += float(LO.detach().cpu());
            else:
                raise ValueError("Unknown operator name:"+op_name)
        loss_weights = self._sample_loss_dropout_weights(op_names) if apply_loss_dropout else op_names
        L = sum([loss_weights[op]*L_dict[op] for op in op_names]) ;
        L_total = float(L.detach().cpu());
        L_ave = [float(l.detach().cpu()) for l in L_dict.values()]
        return L, L_ave, L_total;

    def get_nodeRDM(self,minibatch):

        h = minibatch['h'];
        ne = minibatch['ne'];
        norbs = minibatch['norbs'];
        #epsilon, phi = torch.linalg.eigh(h);
        t0 = time.time();
        epsilon, phi = np.linalg.eigh(h.tolist());
        t1 = time.time();
        if TEST_TIME:
            print(f"[trainer] CPU eigh (nodeRDM) took {t1-t0:.4f} s")
        phi = torch.tensor(phi, dtype=torch.get_default_dtype()).to(self.device);
        rdms = []
        for i in range(len(phi)):
            nb = norbs[i]
            n = ne[i]
            p = phi[i,:nb,:n].to(self.device).float()
            rdm = torch.einsum("ij,kj->ik",[p,p])
            rdms.append(rdm)
        nodeRDM = self.to_mat.nodeRDM(rdms,minibatch,aligned=True)
            
        return nodeRDM

    def pop_iter_timers(self):
        timers = self._iter_times;
        self._iter_times = [];
        return timers;

    def run_model(self, sampler, steps=10, N_batches = 1,
                    op_names=[], op_matrices=[],charge_matrices=[],do_backward=True,time_stat=False) -> float:

        # Train the model using given data points
        # M: number of data points, N: number of atoms, B: number of basis
        # pos_list: MxNx3 list of coordinates of input configurations
        # elements_list: MxN list of atomic species, 'C' or 'H'
        # E_list: M list of energy, unit Hartree
        # S_list: MxBxB list of overlap matrix <phi_i|phi_j>
        # N_list: MxBxB list of density matrix <phi_i|N|phi_j>
        # steps: steps to train using this dataset.
        # This method implements gradient descend to the contained model
        # and return the average loss
        
        n_molecules = len(sampler)
        operators_electric = [key for key in list(op_names.keys()) \
                              if key in ['x','y','z','xx','yy',
                                         'zz','xy','xz','yz']];
        L_ave = np.zeros(len(op_names));
        L_total = 0
        batch_size = -(-n_molecules // N_batches);
        # Every rank must run the SAME number of batches or the DDP gradient
        # all-reduce desyncs and hangs, so fail loudly instead of clamping.
        if n_molecules < N_batches:
            raise ValueError(
                f"N_batches={N_batches} exceeds the {n_molecules} molecules on this rank; "
                "reduce N_batches (empty batches would also desync DDP).");
        #print('batch size:',batch_size)
        #N_batchs = max(1,int(round(self.n_molecules/batch_size)));
        #assert N_batchs == 1
        rank = int(os.environ.get("RANK", 0)) 
        for _ in range(steps):  # outer loop of training steps
            
            ########### forward calculations ################
            # apply the NN-model to get K-S potential

            for batch_ind in range(N_batches):  # inner loop of batch size
                #if rank == 0:
                    #print(torch.matmul(alpha_0,T_mats[i]))
                #    pdb.set_trace()
                iter_t0 = time.time();
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
                #self.optim.zero_grad()  # clear gradient
                for param in self.model.parameters():
                    param.grad = None

                # data: sample batch (and implicit H2D if any), labels: dictionary of ['atomic_charge','E_gap','B','alpha','F'] + operator_electric
                data_t0 = time.time();
                minibatch, labels, ind = sampler.sample(batch_ind = batch_ind, batch_size=batch_size,
                                                        irreps=self.irreps, op_names=operators_electric,
                                                        n_batches=N_batches);
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
                data_t1 = time.time();

                # fwd + bwd: including optional preprocessing (nodeRDM), model, loss, backward, optimizer
                fwd_t0 = data_t1;
                # Start NVML sampling via module helper
                epoch_id = int(os.environ.get('EPOCH', -1))
                output_root = os.environ.get('OUTPUT_DIR', 'output')
                iter_nvml_file = output_root + '/loss/iter_nvml.txt'
                nvml_state = start_iter_nvml(self.device)

                # fine-grained timings
                node_t = 0.0; inf_t = 0.0; losses_init_t = 0.0; build_loss_t = 0.0; backward_t = 0.0; optim_t = 0.0
                if self.nodeRDM_flag:
                    _t0 = time.time();
                    nodeRDM = self.get_nodeRDM(minibatch)
                    minibatch["nodeRDM"] = torch.cat(nodeRDM,dim=0)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
                    node_t = time.time() - _t0
                
                if self.ham_mode:
                    # ── Hamiltonian path: forward -> Fock correction V -> H=h+V
                    #    -> diagonalize -> observables via Fock2obs (Feynman-Hellmann
                    #    analytic backward). Size-extensive by construction. ──
                    _t0 = time.time();
                    if do_backward:
                        mats_dict = self.inference(minibatch)
                    else:
                        with torch.no_grad():
                            mats_dict = self.inference(minibatch)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
                    inf_t = time.time() - _t0

                    _t0 = time.time();
                    h          = minibatch['h'];
                    ne         = minibatch['ne'];
                    norbs      = minibatch['norbs'];
                    alpha_bp86 = minibatch['alpha_bp86'];
                    r_mats = [torch.stack([op_matrices[ii][key] for key in ['x','y','z']])
                              for ii in ind];
                    info = {'alpha_bp86': alpha_bp86, 'r_mats': r_mats};
                    state_calculator = state_solver(h, mats_dict, ne, norbs, self.device, info=info);
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
                    losses_init_t = time.time() - _t0

                    _t0 = time.time();
                    property_dic = self.predict_properties(
                        state_calculator, ind, op_names,
                        op_matrices=op_matrices, charge_matrices=charge_matrices,
                        labels=labels);
                    loss_grad, loss_out, l_total = self.weighted_mse_loss(
                        property_dic, labels, op_names);
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
                    build_loss_t = time.time() - _t0
                else:
                    _t0 = time.time();
                    # Direct GNN forward (no H matrix, no diagonalization)
                    if do_backward:
                        raw = self.model(minibatch)
                    else:
                        with torch.no_grad():
                            raw = self.model(minibatch)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
                    inf_t = time.time() - _t0

                    losses_init_t = 0.0

                    _t0 = time.time();
                    property_dic = self.direct_predict_properties(raw, labels, op_names)
                    loss_grad, loss_out, l_total = self.weighted_mse_loss(
                        property_dic, labels, op_names, apply_loss_dropout=do_backward
                    );
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
                    build_loss_t = time.time() - _t0
                
                #with torch.autograd.detect_anomaly():
                _t0 = time.time();
                if do_backward:
                    loss_grad.backward()  # calculate the gradient
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
                backward_t = time.time() - _t0

                _t0 = time.time();
                if do_backward:
                    self.optim.step()  # implement gradient descend
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device if isinstance(self.device, int) else None)
                optim_t = time.time() - _t0
                fwd_t1 = time.time();
                
                rank = int(os.environ.get("RANK", 0)) 
                #if TEST_TIME and rank == 0:
                #    print(V)
                #    pdb.set_trace()
                #print(f"[Rank {rank}] Before barrier", flush=True)
                #dist.barrier()

                L_ave += loss_out;
                L_total += l_total;

                # compute per-iteration timing segments
                iter_total = fwd_t1 - iter_t0;
                data_s = data_t1 - data_t0;
                fwd_bwd_s = fwd_t1 - fwd_t0;
                other_s = max(iter_total - data_s - fwd_bwd_s, 0.0);
                self._iter_times.append((data_s, fwd_bwd_s, other_s, iter_total));

                # print fine-grained fwd+bwd breakdown (rank 0 only)
                if time_stat and rank == 0:
                    fine_sum = node_t + inf_t + losses_init_t + build_loss_t + backward_t + optim_t
                    print(("[fwd+bwd] nodeRDM=%0.6fs, inference=%0.6fs, LossesInit=%0.6fs, "
                           "build_loss=%0.6fs, backward=%0.6fs, optim=%0.6fs | sum=%0.6fs ~= fwd_bwd=%0.6fs") %
                          (node_t, inf_t, losses_init_t, build_loss_t, backward_t, optim_t, fine_sum, fwd_bwd_s))

                # persist per-iteration NVML averages using module helper
                if self.device == 'cpu':
                    pass
                elif time_stat:
                    record_iter_nvml(iter_nvml_file, epoch_id, batch_ind, nvml_state)
                else:
                    # Always stop the NVML thread to prevent thread accumulation across epochs
                    if nvml_state is not None:
                        nvml_state['running'] = False
                        if nvml_state['thread'] is not None:
                            nvml_state['thread'].join(timeout=0.5)

        return L_ave/steps/N_batches, L_total/steps/N_batches;

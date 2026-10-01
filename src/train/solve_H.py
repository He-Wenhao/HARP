# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:16:14 2024

@author: haota
"""

import torch;
import numpy as np;
import pdb
import os
import torch.distributed as dist
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp  # Use numpy as fallback
    HAS_CUPY = False
import model.Fock2obs as obs_fn
from util_basis.util_fns import mse_loss_over_list, torch_lexsort

# solve a batch of Hamiltonians
def solve_H_batch(H, device):
    # The eigendecomposition is intentionally OUTSIDE the autograd graph: the
    # eigenpairs (epsilon, phi) are treated as constants and every observable's
    # gradient w.r.t. H is supplied analytically (Feynman-Hellmann) by the custom
    # autograd Functions in model/Fock2obs.py. So we detach H before solving.
    if HAS_CUPY and device != 'cpu':
        # Zero-copy to CuPy via DLPack (fast path when cupy is available).
        cp_H = cp.from_dlpack(torch.utils.dlpack.to_dlpack(H.detach()))
        epsilon_unsorted, phi_unsorted = cp.linalg.eigh(cp_H);
        epsilon_unsorted = torch.as_tensor(epsilon_unsorted, dtype=torch.float32).to(device);
        phi_unsorted = torch.as_tensor(phi_unsorted, dtype=torch.float32).to(device);
    else:
        # torch fallback (no cupy): works on CPU or CUDA. eigh on the detached
        # Hamiltonian keeps the eigenpairs as graph-free constants.
        epsilon_unsorted, phi_unsorted = torch.linalg.eigh(H.detach().to(torch.float32))
        epsilon_unsorted = epsilon_unsorted.to(torch.float32).to(device)
        phi_unsorted = phi_unsorted.to(torch.float32).to(device)
    return epsilon_unsorted, phi_unsorted

# devide orbitals into occupied and unoccupied orbitals
def divide_occ_unocc(epsilon, phi, ne, norbs):
    co = []; # occupied eigenstates
    cv = []; # unoccupied eigenstates
    eo = []; # occupied energy levels
    ev = []; # unoccupied energy levels
    for i, n in enumerate(ne):
        nb = norbs[i]

        eo.append(epsilon[i][ :n]);
        co.append(phi[i][ :nb, :n]);
        ev.append(epsilon[i][ n:nb]);
        cv.append(phi[i][ :nb, n:nb]);
    return eo, ev, co, cv

# calculate energy denominators for perturbation usage
def calc_energy_denominators(epsilon, ne, norbs, smear=1E-3):
    epsilon_ij = []; # E_{ij} = 1 / (E_i - E_j); E_i is occupied orbitals, E_j is unoccupied orbitals
    epsilon_jk = []; # E_{jk} = 1 / (E_j - E_k); E_j is is unoccupied orbitals, E_k is all orbitals
    epsilon_ik = []; # E_{ik} = 1 / (E_i - E_k); E_i is occupied orbitals, E_k is all orbitals
    for i, n in enumerate(ne):
        nb = norbs[i]
    
        ij = epsilon[i][:n][:,None]-epsilon[i][n:nb][None,:];
        epsilon_ij.append(ij**-1)
    
        epsilon_jk_i = epsilon[i][n:nb][:, None] - epsilon[i][ None, :nb];
        epsilon_jk.append(epsilon_jk_i/(epsilon_jk_i**2 + smear**2));
    
        epsilon_ik_i = epsilon[i][:n][:, None] - epsilon[i][ None, :nb];
        epsilon_ik.append(epsilon_ik_i/(epsilon_ik_i**2 + smear**2));
    return epsilon_ij, epsilon_jk, epsilon_ik


class state_solver(object):
    
    def __init__(self, h, mats_dict, ne, norbs, device, smear = 1E-3,info = {}):
        
        V, T, G, delta_E = mats_dict['V'], mats_dict['T'], mats_dict['G'], mats_dict['delta_E'];
        self.delta_E = delta_E
        self.device = device;
        self.ne = ne; # ne is # of occupied orbitals
        self.V = V; # neural newtork Fock correction
        self.h = h; # Fock matrix of BP86 functional 
        #self.T = T; # screening matrix
        self.G = G; # another corrector
        H = h+V; # final Fock matrix
        self.norbs = norbs; # norbs is # of all orbitals
        # Zero-copy to CuPy via DLPack
        epsilon_unsorted, phi_unsorted = solve_H_batch(H, device);
        #rho0 = self.rho0(h,norbs,ne);
        #self.epsilon, phi = self.states_sort(epsilon_unsorted, phi_unsorted,rho0,norbs);
        self.epsilon, phi = epsilon_unsorted, phi_unsorted
        # epsilon is eigenvalues, energy levels
        # phi is eigenvectors, eigenstates
        #self.epsilon = torch.as_tensor(self.epsilon, dtype=torch.float32).to(device);
        #phi = torch.as_tensor(phi, dtype=torch.float32).to(device);
        
        self.H = [Hi[:nb,:nb] for Hi, nb in zip(H, norbs)];

        self.eo, self.ev, self.co, self.cv = divide_occ_unocc(self.epsilon, phi, ne, norbs);
        self.epsilon_ij, self.epsilon_jk, self.epsilon_ik = calc_energy_denominators(self.epsilon, ne, norbs, smear=smear);
        
        # construct initial screening matrix T
        if info  == {} or info['alpha_bp86'] == None:
            T0 = 0.
        else:
            T0 = self.calc_T0(info['alpha_bp86'], info['r_mats'],h);
        self.T = T0 + 0.01*T;
        
        # just for debug
        # calculate energy from h
        epsilon_h, phi_h = solve_H_batch(h, device);
        self.e_h = [2*epsilon_h[i][:ne[i]].sum() for i in range(len(ne))];
        
    def rho0(self,h,norbs,ne):
        rho0_l = [];
        epsilon_h, phi_h = solve_H_batch(h, self.device);
        for i,n in enumerate(ne):
            nb = norbs[i]
            coh_i = phi_h[i, :nb, :n]
            P = 2*torch.einsum('ik,jk->ij', [coh_i, coh_i]);
            rho0_l.append(P);
        return rho0_l;
    
    def states_sort(self,epsilon_unsorted, phi_unsorted,rho0,norbs):
        # sort epsilon_unsorted and phi_unsorted according to 
        # 1. average value on rho0
        # 2. energy (epsilon_unsorted)
        rho0_avg = [];
        for i in range(epsilon_unsorted.shape[0]):
            nb = norbs[i]
            phi_i = phi_unsorted[i,:nb,:nb];
            P = rho0[i];
            avg_i = torch.einsum('ik,ij,jk->k', [phi_i, P, phi_i]);
            rho0_avg.append(avg_i);
            
        epsilon = [];
        phi = [];
        for i in range(epsilon_unsorted.shape[0]):
            nb = norbs[i]
            avg_i = rho0_avg[i];
            eps_i = epsilon_unsorted[i][:nb];
            phi_i = phi_unsorted[i];
            sort_idx = torch_lexsort((eps_i.clone(), -avg_i.clone()));
            epsilon.append(eps_i[sort_idx]);
            phi.append(phi_i[:, sort_idx]);
        return epsilon, phi;
  
    # def E(self, E_labels):
    #     E_l =  obs_fn.Energy()(self.H, self.eo, self.co, self.ne)
    #     #LE = mse_loss_over_list(E_l, E_labels);
    #     return E_l
    
    # def O(self, O_labels, O_mats):
    #     # O is dipole moment
    #     M_l = obs_fn.Multipole()(self.H, self.eo, self.co, self.ne, O_mats, self.cv, self.epsilon_ij)
    #     #LO = mse_loss_over_list(M_l, O_labels);
    #     return M_l
    
    # def C(self, C_labels, C_mats):
    #     # C is charge
    #     C_l = obs_fn.Charge()(self.H, self.eo, self.co, self.ne, C_mats, self.cv, self.epsilon_ij)
    #     #LC = mse_loss_over_list(C_l, C_labels);

    #     return C_l
        
    # def B(self, B_labels, B_mats):
    #     # B is Mayers bond order
    #     mask_l = [(B_label != 0) for B_label in B_labels]
    #     B_l = obs_fn.Bond()(self.H, self.eo, self.co, self.ne, B_mats, self.cv, self.epsilon_ij)
    #     #LB = mse_loss_over_list(B_l, B_labels,mask=mask_l);
            
    #     return B_l

    # def Eg(self, Eg_labels):
    #     # Eg is Energy gap
    #     Eg_l_raw = obs_fn.EnergyGap()(self.H, self.eo, self.co, self.ev, self.cv, self.ne)
    #     Eg_l = [Eghat_raw * (1 + self.G[i,0]) + self.G[i,1] for i, Eghat_raw in enumerate(Eg_l_raw)]
    #     #LEg = mse_loss_over_list(Eg_l, Eg_labels);
    #     return Eg_l
    
    # def polar(self, alpha_labels, r_mats):
    #     # polar is polarizibility
    #     T_mats = [t for t in self.T];
    #     bare_polar_l = obs_fn.BarePolarizability()(self.H, self.eo, self.co, self.ne, r_mats, self.cv, self.epsilon_ij, self.epsilon_jk, self.epsilon_ik)
    #     polar_l = obs_fn.Polarizability()(bare_polar_l, T_mats)
    #     #Lpolar = mse_loss_over_list(polar_l, alpha_labels);
        
    #     return polar_l
    
    def calc_T0(self,alpha_bp86,r_mats, h):
        # polar is polarizibility
        BarePolarizability_fn = obs_fn.BarePolarizability();
        epsilon_h, phi_h = solve_H_batch(h, self.device);
        T0_l = [];
        eo_l, ev_l, co_l, cv_l = divide_occ_unocc(epsilon_h, phi_h, self.ne, self.norbs);
        h_l = [Hi[:nb,:nb] for Hi, nb in zip(h, self.norbs)];
        epsilon_h_ij_l, _, _ = calc_energy_denominators(epsilon_h, self.ne, self.norbs, smear=1E-3);
        bare_polar_l = obs_fn.BarePolarizability()(h_l, eo_l, co_l, self.ne, r_mats, cv_l, epsilon_h_ij_l, None, None)
        assert all(bp.requires_grad == False for bp in bare_polar_l)
        T0_l = [torch.linalg.inv(alpha_bp86[i]) - torch.linalg.inv(bare_polar_l[i]) for i in range(len(bare_polar_l))]
        return torch.stack(T0_l);

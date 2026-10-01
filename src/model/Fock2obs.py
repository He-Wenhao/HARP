import torch
from torch import nn

def ragged_apply(func, *args):
    """
    Apply `func` either to a single sample or to a ragged batch (list).

    Args:
        func:          a forward_single-like function
        *args:         inputs (some may be lists, others scalars/tensors)

    Returns:
        If batched:    a list of outputs
        If single:     func(*args)
    """

    # Check if ANY argument is a list → treat as ragged batch
    is_batched = any(isinstance(a, list) for a in args)

    # Single input → call directly
    if is_batched:
        assert all(isinstance(a, list) or a == None for a in args)
    else:
        return func(*args)

    # Ensure ALL arguments that should be batched are lists of same length
    batch_sizes = [len(a) for a in args if isinstance(a, list)]
    assert len(set(batch_sizes)) == 1, f"Ragged batch size mismatch: {batch_sizes}"
    B = batch_sizes[0]

    # Apply func to each sample
    outputs = []
    for i in range(B):
        elem_args = [a[i] if isinstance(a, list) else a for a in args]
        outputs.append(func(*elem_args))
    return outputs


class EnergyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, eo, co, ne):
        """
        ctx: context to save tensors for backward
        H:  Hamiltonian (matrix)
        eo: orbital energies or some other quantity
        co: coefficients (wavefunction)
        ne: electron number (can be int or tensor)
        
        WARNING: This function only supports first-order derivatives via 
        Feynman-Hellmann theorem. Second-order derivatives are not available.
        
        Function E(H)
        """
        # Example forward — modify as you need
        E = 2 * torch.sum(eo)
        
        # Save tensors for backward use
        ctx.save_for_backward(H, eo, co)
        ctx.ne = ne
        return E

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """
        Backward of: 2 * torch.einsum('ij,kj,ik->', [co, co, H])
        with respect to H (co is treated as constant)
        
        dE/dH[i,k] = 2 * sum_j(co[i,j] * co[k,j])
        """
        _, _, co = ctx.saved_tensors
        
        # Gradient w.r.t. H
        grad_H = 2 * torch.einsum('ij,kj->ik', co, co)
        
        # Return gradients w.r.t. (H, eo, co, ne)
        return grad_output * grad_H, None, None, None

class Energy(nn.Module):
    def __init__(self):
        super(Energy, self).__init__()

    def forward_single(self, Ham, eo, co, ne):
        """
        Wrapper that calls the custom autograd function
        
        Args:
            Ham: Hamiltonian matrix
            eo: orbital energies
            co: coefficients
            ne: number of electrons
        """
        return EnergyFunction.apply(Ham, eo, co, ne)

    # wrapper for ragged batched input
    def forward(self, Ham_l, eo_l, co_l, ne_l):
        return ragged_apply(self.forward_single, Ham_l, eo_l, co_l, ne_l)
    
    
    
    
    
class MultipoleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, eo, co, ne, O_mats, cv, epsilon_ij):
        """
        ctx: context to save tensors for backward
        H:  Hamiltonian (matrix)
        eo: orbital energies or some other quantity
        co: coefficients (wavefunction)
        ne: electron number (can be int or tensor)
        
        WARNING: This function only supports first-order derivatives via 
        Feynman-Hellmann theorem. Second-order derivatives are not available.
        
        Function E(H)
        """
        # Example forward — modify as you need
        Ohat = 2 * torch.einsum('bui,buv,bvi->b', [co, O_mats, co]);
        
        # Save tensors for backward use
        ctx.save_for_backward(H, eo, co, cv, O_mats, epsilon_ij)
        ctx.ne = ne
        return Ohat

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """
        Backward of: 2 * torch.einsum('ij,kj,ik->', [co, co, H])
        with respect to H (co is treated as constant)
        
        dE/dH[i,k] = 2 * sum_j(co[i,j] * co[k,j])
        """
        _, _, co, cv, O_mats, epsilon_ij  = ctx.saved_tensors
        
        # Gradient w.r.t. H
        O_ij = torch.einsum('buk,buv,bvi->bik', [cv, O_mats, co]) 
        #braket_ijkl = torch.einsum('bui,bvk->bikuv', [co, cv])
        #grad_O = torch.einsum("bij,bij,bijkl->bkl", [epsilon_ij,O_ij,  braket_ijkl])
        grad_O = torch.einsum("bij,bij,bki,blj ->bkl", [epsilon_ij,O_ij,  co, cv])
        
        # Return gradients w.r.t. (H, eo, co, ne)
        return grad_output * grad_O, None, None, None, None, None, None

class Multipole(nn.Module):
    def __init__(self):
        super(Multipole, self).__init__()

    def forward_single(self, H, eo, co, ne, O_mats, cv, epsilon_ij):
        """
        Wrapper around MultipoleFunction that supports both
        single-sample and batched inputs.

        Inputs:
            H:          [B, n, n] or [n, n]
            eo:         [B, ...] or [...]
            co, cv:     [B, n, m] or [n, m]
            ne:         scalar or [B]
            O_mats:     [B, n, n] or [n, n]
            epsilon_ij: [B, n, n] or [n, n]
        Returns:
            Ohat:       [B] if batched, scalar if single
        """
        # Detect if input has batch dimension
        batched = O_mats.dim() == 3

        # If not batched, add dummy batch dim
        if not batched:
            H = H.unsqueeze(0)
            eo = eo.unsqueeze(0) if torch.is_tensor(eo) else eo
            co = co.unsqueeze(0)
            cv = cv.unsqueeze(0)
            O_mats = O_mats.unsqueeze(0)
            epsilon_ij = epsilon_ij.unsqueeze(0)
            if torch.is_tensor(ne):
                ne = ne.unsqueeze(0)

        # Call autograd function
        Ohat = MultipoleFunction.apply(H, eo, co, ne, O_mats, cv, epsilon_ij)

        # If single sample, remove batch dim
        if not batched:
            Ohat = Ohat.squeeze(0)

        return Ohat
    
    def forward(self, H, eo, co, ne, O_mats, cv, epsilon_ij):
        # allow ragged batch input via list
        return ragged_apply(self.forward_single, H, eo, co, ne, O_mats, cv, epsilon_ij)
    
    
    
    

class ChargeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, eo, co, ne, O_mats, cv, epsilon_ij):
        """
        ctx: context to save tensors for backward
        H:  Hamiltonian (matrix)
        eo: orbital energies or some other quantity
        co: coefficients (wavefunction)
        ne: electron number (can be int or tensor)
        
        WARNING: This function only supports first-order derivatives via 
        Feynman-Hellmann theorem. Second-order derivatives are not available.
        
        Function E(H)
        """
        # Example forward — modify as you need
        Ohat = 2 * torch.einsum('ui,buv,vi->b', [co, O_mats, co]);
        
        # Save tensors for backward use
        ctx.save_for_backward(H, eo, co, cv, O_mats, epsilon_ij)
        ctx.ne = ne
        return Ohat

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """
        Backward of: 2 * torch.einsum('ij,kj,ik->', [co, co, H])
        with respect to H (co is treated as constant)
        
        dE/dH[i,k] = 2 * sum_j(co[i,j] * co[k,j])
        """
        _, _, co, cv, O_mats, epsilon_ij  = ctx.saved_tensors
        
        # Gradient w.r.t. H
        O_ij = torch.einsum('uk,buv,vi->bik', [cv, O_mats, co]) 
        #braket_ijkl = torch.einsum('ui,vk->ikuv', [co, cv])
        #grad_O = torch.einsum("ij,bij,ijkl->bkl", [epsilon_ij,O_ij,  braket_ijkl])
        #braket_ijkl = torch.einsum('bui,bvk->bikuv', [co, cv])
        #grad_O = torch.einsum("bij,bij,bijkl->bkl", [epsilon_ij,O_ij,  braket_ijkl])
        grad_O = torch.einsum("ij,bij,ki,lj ->bkl", [epsilon_ij,O_ij,  co, cv])
        
        # Return gradients w.r.t. (H, eo, co, ne)
        return grad_output[:,None,None] * grad_O, None, None, None, None, None, None

class Charge(nn.Module):
    def __init__(self):
        super(Charge, self).__init__()

    def forward_single(self, H, eo, co, ne, O_mats, cv, epsilon_ij):
        # Detect if input has batch dimension
        assert O_mats.dim() == 3

        # Call autograd function
        Ohat = ChargeFunction.apply(H, eo, co, ne, O_mats, cv, epsilon_ij)

        # If single sample, remove batch dim
        #if not batched:
        #    Ohat = Ohat.squeeze(0)

        return Ohat
    
    def forward(self, H, eo, co, ne, O_mats, cv, epsilon_ij):
        # allow ragged batch input via list
        return ragged_apply(self.forward_single, H, eo, co, ne, O_mats, cv, epsilon_ij)
    
    

class BondFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, eo, co, ne, O_mats, cv, epsilon_ij):
        """
        ctx: context to save tensors for backward
        H:  Hamiltonian (matrix)
        eo: orbital energies or some other quantity
        co: coefficients (wavefunction)
        ne: electron number (can be int or tensor)
        
        WARNING: This function only supports first-order derivatives via 
        Feynman-Hellmann theorem. Second-order derivatives are not available.
        
        Function E(H)
        """
        # Example forward — modify as you need
        P = torch.einsum('jk,lk->jl', [co, co]);
        Ohat = 4*torch.einsum('jk,ukl,lm,vmj->uv', [P, O_mats, P, O_mats]);
        
        # Save tensors for backward use
        ctx.save_for_backward(H, eo, co, cv, O_mats, epsilon_ij,P)
        ctx.ne = ne
        return Ohat

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """
        Backward of: 2 * torch.einsum('ij,kj,ik->', [co, co, H])
        with respect to H (co is treated as constant)
        
        dE/dH[i,k] = 2 * sum_j(co[i,j] * co[k,j])
        """
        _, _, co, cv, O_mats, epsilon_ij,P  = ctx.saved_tensors
        term2 = torch.einsum('mj,ni,ij->ijmn', [cv,  co, epsilon_ij]);
        term31 = torch.einsum('mi,amn,nw,bwq,qj->abij', [co, O_mats, P, O_mats, cv]);
        term32 = torch.einsum('mj,amn,nw,bwq,qi->abij', [cv, O_mats, P, O_mats, co]);
        grad_O = torch.einsum('ijmn,abij->abmn', [term2, term31+term32])
        # Gradient w.r.t. H
        #O_ij = torch.einsum('uk,buv,vi->bik', [cv, O_mats, co]) 
        #braket_ijkl = torch.einsum('ui,vk->ikuv', [co, cv])
        #grad_O = torch.einsum("ij,bij,ijkl->bkl", [epsilon_ij,O_ij,  braket_ijkl])
        
        # Return gradients w.r.t. (H, eo, co, ne)
        return grad_output[:,:,None,None] * grad_O, None, None, None, None, None, None

class Bond(nn.Module):
    def __init__(self):
        super(Bond, self).__init__()

    def forward_single(self, H, eo, co, ne, O_mats, cv, epsilon_ij):
        # Detect if input has batch dimension
        assert O_mats.dim() == 3

        # Call autograd function
        Ohat = BondFunction.apply(H, eo, co, ne, O_mats, cv, epsilon_ij)

        # If single sample, remove batch dim
        #if not batched:
        #    Ohat = Ohat.squeeze(0)

        return Ohat
    
    def forward(self, H, eo, co, ne, O_mats, cv, epsilon_ij):
        # allow ragged batch input via list
        return ragged_apply(self.forward_single, H, eo, co, ne, O_mats, cv, epsilon_ij)
    
    

class EnergyGapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, eo, co, ev, cv, ne):
        """
        ctx: context to save tensors for backward
        H:  Hamiltonian (matrix)
        eo: orbital energies or some other quantity
        co: coefficients (wavefunction)
        ne: electron number (can be int or tensor)
        
        WARNING: This function only supports first-order derivatives via 
        Feynman-Hellmann theorem. Second-order derivatives are not available.
        
        Function E(H)
        """
        # Example forward — modify as you need
        #E = 2 * torch.sum(eo)
        Eg = ev[0] - eo[-1]
        
        # Save tensors for backward use
        ctx.save_for_backward(H, eo, co,ev,cv)
        ctx.ne = ne
        return Eg

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """
        Backward of: 2 * torch.einsum('ij,kj,ik->', [co, co, H])
        with respect to H (co is treated as constant)
        
        dE/dH[i,k] = 2 * sum_j(co[i,j] * co[k,j])
        """
        _, _, co, _, cv = ctx.saved_tensors
        
        # Gradient w.r.t. H
        grad_H = torch.einsum('i,k->ik', cv[:,0], cv[:,0]) - torch.einsum('i,k->ik', co[:,-1], co[:,-1])
        
        # Return gradients w.r.t. (H, eo, co, ne)
        return grad_output * grad_H, None, None, None, None, None

class EnergyGap(nn.Module):
    def __init__(self):
        super(EnergyGap, self).__init__()

    def forward_single(self, Ham, eo, co, ev, cv, ne):
        """
        Wrapper that calls the custom autograd function
        
        Args:
            Ham: Hamiltonian matrix
            eo: orbital energies
            co: coefficients
            ne: number of electrons
        """
        return EnergyGapFunction.apply(Ham, eo, co, ev, cv, ne)
    def forward(self, Ham_l, eo_l, co_l, ev_l, cv_l, ne_l):
        return ragged_apply(self.forward_single, Ham_l, eo_l, co_l, ev_l, cv_l, ne_l)
    



class BarePolarizabilityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, eo, co, ne, O_mats, cv, epsilon_ij,epsilon_jk,epsilon_ik):
    
        rij = torch.einsum('mi, xmn, nj -> xij', 
                                [co, O_mats, cv]);

        alpha_0 = - 4*torch.einsum('xij, yij, ij -> xy', [rij, rij, epsilon_ij]);
        
        ctx.save_for_backward(H, eo, co, cv, O_mats, epsilon_ij,epsilon_jk,epsilon_ik,rij)
        ctx.ne = ne
        return alpha_0

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        """
        Backward of: 2 * torch.einsum('ij,kj,ik->', [co, co, H])
        with respect to H (co is treated as constant)
        
        dE/dH[i,k] = 2 * sum_j(co[i,j] * co[k,j])
        """
        H, eo, co, cv, O_mats, epsilon_ij,epsilon_jk,epsilon_ik, rij  = ctx.saved_tensors
        
        phi = torch.cat([co, cv], dim=-1);
        rik = torch.einsum('mi, xmn, nj -> xij',[co, O_mats, phi]); 
        rjk = torch.einsum('mi, xmn, nj -> xij',[cv, O_mats, phi]); 
        
        
        Vik = torch.einsum('mi,   nj -> ijmn',[co,  phi]);
        Vjk = torch.einsum('mi, nj -> ijmn',[cv,  phi]);
        Vkk_o = torch.einsum('mi, ni -> imn',[co,  co]);
        Vkk_v = torch.einsum('mi,ni -> imn',[cv,  cv]);
        V_diff = Vkk_o.unsqueeze(1) - Vkk_v.unsqueeze(0);
        
        grad_term_1 = 4*torch.einsum('xij, yij, ij, ijmn -> xymn', 
                        [rij,rij,epsilon_ij**2, V_diff]);

        grad_term_2 = -4*torch.einsum('xij, yik, jkmn, ij, jk -> xymn', 
                    [rij, rik, Vjk, epsilon_ij, epsilon_jk]);
        grad_term_3 = -4*torch.einsum('xij, yjk, ikmn, ij, ik -> xymn', 
                    [rij, rjk, Vik, epsilon_ij, epsilon_ik]);
        grad_O = grad_term_1 + grad_term_2+ grad_term_2.transpose(0,1) + grad_term_3 + grad_term_3.transpose(0,1)
        # Return gradients w.r.t. (H, eo, co, ne)
        return grad_O*grad_output[:,:,None,None], None, None, None, None, None, None,None, None

class BarePolarizability(nn.Module):
    def __init__(self):
        super(BarePolarizability, self).__init__()

    def forward_single(self, H, eo, co, ne, O_mats, cv, epsilon_ij,epsilon_jk,epsilon_ik):
        Ohat = BarePolarizabilityFunction.apply(H, eo, co, ne, O_mats, cv,  epsilon_ij,epsilon_jk,epsilon_ik)
        return Ohat
    def forward(self, H_l, eo_l, co_l, ne_l, O_mats_l, cv_l, epsilon_ij_l,epsilon_jk_l,epsilon_ik_l):
        return ragged_apply(self.forward_single, H_l, eo_l, co_l, ne_l, O_mats_l, cv_l, epsilon_ij_l,epsilon_jk_l,epsilon_ik_l)
    
    

class PolarizabilityFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha_0, T):
        device = alpha_0.device
        denominator = torch.linalg.inv(torch.eye(3).to(device) +alpha_0 @ T); 
        alpha_hat = torch.matmul(denominator, alpha_0);
        ctx.save_for_backward(alpha_hat,alpha_0, T, denominator)
        return alpha_hat

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        
        alpha_hat, alpha_0, T, denominator = ctx.saved_tensors
        device = alpha_0.device
        grad_alpha0 = torch.einsum('ij,kl -> iljk', [denominator, (torch.eye(3).to(device)-T @ alpha_hat)])
        grad_T = - torch.einsum('ij,kl -> iljk', [alpha_hat , alpha_hat])
        return grad_alpha0 * grad_output[:,:,None,None], grad_T * grad_output[:,:,None,None]
        

    
class Polarizability(nn.Module):
    def __init__(self):
        super(Polarizability, self).__init__()

    def forward_single(self, alpha_0, T):
        Ohat = PolarizabilityFunction.apply(alpha_0, T)
        return Ohat  
    
    def forward(self, alpha_0_l, T_l):
        return ragged_apply(self.forward_single, alpha_0_l, T_l)
    
    
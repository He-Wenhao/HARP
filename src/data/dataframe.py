#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:39:10 2023

@author: ubuntu
"""

import json;
import torch;
import scipy;
from util_basis.Irreps import Irreps_build;

def load_data(molecules, device, path = 'data', load_obs_mat = True, 
              ind_list = [], op_names = [], element_list = ['C','H']):

    data_in = [];
    labels = [];
    obs_mats = [];
    irreps = Irreps_build(element_list);
    nbasis = irreps.get_nbasis();
    ne_dic = irreps.get_ne();
    
    cm_inverse_to_hartree = 4.55633528*1E-6;
    
    for molecule in molecules:
        with open(path+'/'+molecule+'_data.json', 'r') as file:

            data = json.load(file);

        frame_idx = ind_list if (ind_list != []) else list(range(len(data['coordinates'])));

        pos = torch.tensor(data['coordinates'])[frame_idx].to(device);
        pos[:,:,[0,1,2]] = pos[:,:,[1,2,0]];
        elements = data['elements'][0];
        nuclearCharge = [ne_dic[ele] for ele in elements];
        nuclearCharge = torch.tensor(nuclearCharge, dtype=torch.float).to(device);
        
        ne = int(round(sum([ne_dic[ele] for ele in elements])/2));
        norbs = int(round(sum([nbasis[ele] for ele in elements])));
        nframe = len(pos);

        data_in.append({'pos':pos,'elements':elements,
                   'properties':{'ne':ne, 'norbs':norbs,'nframe':nframe}});

        h = torch.tensor(data['h'])[frame_idx].to(device);
        S_mhalf = [scipy.linalg.fractional_matrix_power(data['S'][i], (-1/2)).tolist() for i in frame_idx]
        S_mhalf = torch.tensor(S_mhalf).to(device);
        h = torch.matmul(torch.matmul(S_mhalf, h),S_mhalf);
        
        B_labels = torch.zeros([nframe, len(elements), 
                                len(elements)]).to(device);
        for u in range(nframe):
            i_global = frame_idx[u];
            for dp in data['bond_order'][i_global]:
                B_labels[u, dp[0], dp[1]] = dp[2];
                B_labels[u, dp[1], dp[0]] = dp[2];
                
        label = {'S': torch.tensor(data['S'])[frame_idx].to(device),
                 'Smhalf': S_mhalf,
                 'h': h,
                 'E': torch.tensor(data['energy'])[frame_idx].to(device),
                 'E_nn':torch.tensor(data['Enn'])[frame_idx].to(device),
                 'atomic_charge': -torch.tensor(data['atomic_charge'])[frame_idx].to(device) + \
                      nuclearCharge[None,:],
                 'E_gap': cm_inverse_to_hartree * \
                      torch.tensor([u[0] for u in data['Ee']])[frame_idx].to(device),
                 'B': B_labels,
                 'alpha': torch.tensor(data['alpha'])[frame_idx].to(device)};
            
        for op_name in op_names:
            label[op_name] = torch.tensor(data[op_name])[frame_idx].to(device)
        labels.append(label);
        
        if(load_obs_mat):
            with open(path+'/'+molecule+'_obs_mat.json', 'r') as file:
                data_obs = json.load(file);                  

            if op_names is None:
                op_names = set(data_obs.keys())

            def _obs_tensor(op):
                t = torch.Tensor(data_obs[op]).to(device);
                if ind_list != []:
                    t = t[frame_idx]
                return t

            obs_mats.append({op: torch.matmul(torch.matmul(S_mhalf, _obs_tensor(op)),S_mhalf)
                                                            for op in op_names})
                
    return data_in, labels, obs_mats;



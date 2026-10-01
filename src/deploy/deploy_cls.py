#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 21:54:56 2023

@author: ubuntu
"""

from util_basis.integral import integrate
from model import V_theta
#from deploy.predictor import predict_fns
import numpy as np
import torch
from model.irreps2mats import to_mat;
from train.train import trainer;
from train.solve_H import state_solver
import os;
import json;
import matplotlib;
import matplotlib.pyplot as plt;

class estimator_test(trainer):

    def __init__(self, device, nodeRDM_flag, output_folder='output/test') -> None:

        # Initialize a neural network model, an optimizer,
        # and set training parameters.
        # Device: 'cpu' or 'gpu', the device to train the model
        # kn: the weight of the electron density term in the loss function
        # lr: learning rate

        trainer.__init__(self, device, nodeRDM_flag);

        if(not os.path.exists(output_folder)):
            os.mkdir(output_folder);
    
    def evaluate_properties(self, calc, ind, op_names, op_matrices=[],charge_matrices=[]):
        properties_dic = {};
        with torch.no_grad():
            properties_dic = trainer.predict_properties(self, calc, ind, op_names, op_matrices=op_matrices,charge_matrices=charge_matrices)
            
        # convert torch tensor to floats or lists
        def torch2listORfloat(tensor):
            if(len(tensor.shape)==0):
                return float(tensor);
            else:
                return tensor.tolist();
        properties_dic_list = {key:[torch2listORfloat(item) for item in properties_dic[key]] for key in properties_dic};
        return properties_dic_list;

    def solve(self, my_sampler, N_batches=1, op_names=[], op_matrices=[],charge_matrices=[]):

        n_molecules = len(my_sampler);

        operators_electric = [key for key in list(op_names.keys()) \
                        if key in ['x','y','z','xx','yy',
                                    'zz','xy','xz','yz']];

        properties_dic = {};
        
        batch_size = -(-n_molecules // N_batches);

        for batch_ind in range(N_batches):
            minibatch, labels, ind = my_sampler.sample(batch_ind = batch_ind, batch_size=batch_size,
                                                    irreps=self.irreps, op_names=operators_electric,
                                                    n_batches=N_batches);
            if self.nodeRDM_flag:
                nodeRDM = self.get_nodeRDM(minibatch)
                minibatch["nodeRDM"] = torch.cat(nodeRDM,dim=0)
                
            # return matrices including fock matrix
            mats_dict = self.inference(minibatch);
            
            # number of occupied orbitals
            h = minibatch['h'];
            ne = minibatch['ne'];
            norbs = minibatch['norbs'];
            state_calc = state_solver(h, mats_dict, ne, norbs, self.device);
            
            properties = self.evaluate_properties(state_calc, ind, op_names, op_matrices=op_matrices,charge_matrices=charge_matrices);
            for key in properties:
                if(key not in properties_dic):
                    properties_dic[key] = {'pred':[],'label':[]};
                properties_dic[key]['pred'] += properties[key];
                if(key=='bond_order'):
                    properties_dic[key]['label'] += [u.tolist() for u in labels['B']];
                elif(key == 'E'):
                    properties_dic[key]['label'] += labels['Ee'];
                elif(key in ['atomic_charge','alpha','F']):
                    properties_dic[key]['label'] += [u.tolist() for u in labels[key]];
                elif key == 'V':
                    continue
                else:
                    properties_dic[key]['label'] += labels[key];

            print('complete ' + str(batch_ind) + ' of ' + str(N_batches));

        return properties_dic;

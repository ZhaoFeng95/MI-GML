# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:30:36 2020

@author: Administrator
"""



import torch
import torch.nn as nn
from MI_GML_learner import MetaLearner
from torch import optim
import torch.nn.functional as F
import numpy as np
from VGAE_modules import VGAE
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
import math


class Film(nn.Module):
    def __init__(self, args):
        super(Film, self).__init__()
        self.node_num = (args.k_spt + args.k_qry) * args.n_way
        self.w1 = nn.Parameter(torch.Tensor(self.node_num, args.n_GCN[1]))
        self.w2 = nn.Parameter(torch.Tensor(self.node_num, args.n_GCN[2]))
        self.fc1 = nn.Linear(args.n_GCN[1], args.n_GCN[2])
        # self.fc1 = nn.Linear(256, 128)
        # self.fc2 = nn.Linear(128, 64)
        # self.fc3 = nn.Linear(64, 16)
        # self.fc4 = nn.Linear(16, 8)
        # self.fc5 = nn.Linear(8, 2)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w1.size(1))
        self.w1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.w2.size(1))
        self.w2.data.uniform_(-stdv, stdv)
        
    def forward(self, feats):
        tao_1 = torch.mm(feats.t(), self.w1)
        h = self.fc1(feats)
        tao_2 = torch.mm(h.t(), self.w2)
        return tao_1, tao_2
    
class MetaGCN(nn.Module):
    def __init__(self, args, activation):
        super(MetaGCN, self).__init__()
        self.alpha = 0.8
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.GAE_lr = args.GAE_lr
        self.Film_lr = args.Film_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        
        self.Local_GAE = VGAE(args.n_GCN[0], args.n_GCN[1], args.dropout, isGlobal = False)
        self.Global_GAE = VGAE(args.n_GCN[0], args.n_GCN[1], args.dropout, isGlobal = True)
        
        self.Local_Film = Film(args)
        self.Global_Film = Film(args)
        
        self.net = MetaLearner(args.n_layer, args.n_GCN, activation, args.dropout)
        
        
        self.meta_optim = optim.Adam([{'params': self.net.parameters(), 'lr':self.meta_lr},
                                      {'params': self.Local_GAE.parameters(), 'lr':self.GAE_lr},
                                      {'params': self.Global_GAE.parameters(), 'lr':self.GAE_lr},
                                      {'params': self.Local_Film.parameters(), 'lr':self.Film_lr},
                                      {'params': self.Global_Film.parameters(), 'lr':self.Film_lr}])
        
    def forward(self, g, feats, spt_idx, qry_idx, labels_local, adj, PPMI):
        task_num = self.task_num
        querysz = self.n_way * self.k_qry

        losses_q = [0 for _ in range(self.update_step + 1)]
        corrects = [0 for _ in range(self.update_step + 1)]
        
        with torch.no_grad():
            Local_g = g
            Global_g = g
        
        adj_logits_L, h_out_L = self.Local_GAE.forward(Local_g, feats, PPMI)
        adj_logits_G, h_out_G = self.Global_GAE.forward(Global_g, feats, PPMI)
        loss_link = BCELoss(adj_logits_L, adj)
        loss_KL = 0.5/ adj_logits_L.size(0) * (1 + 2*self.Local_GAE.logstd - self.Local_GAE.mean**2 - torch.exp(self.Local_GAE.logstd)**2).sum(1).mean()
        loss_link += BCELoss(adj_logits_G, PPMI)
        loss_KL += 0.5/ adj_logits_G.size(0) * (1 + 2*self.Global_GAE.logstd - self.Global_GAE.mean**2 - torch.exp(self.Global_GAE.logstd)**2).sum(1).mean()
        
        loss_gae = loss_link - loss_KL
        
        
        for i in range(task_num):
            h_input_L = torch.cat((h_out_L[spt_idx[i]], h_out_L[qry_idx[i]]), dim = 0)
            h_input_G = torch.cat((h_out_G[spt_idx[i]], h_out_G[qry_idx[i]]), dim = 0)
            
            tao_1_L, tao_2_L = self.Local_Film(h_input_L)
            tao_1_G, tao_2_G = self.Global_Film(h_input_G)
            
            logits = self.net(g, feats, tao_1_L, tao_2_L, tao_1_G, tao_2_G, PPMI, net_vars=None)
            
            loss = F.cross_entropy(logits[spt_idx[i]], labels_local[spt_idx[i]])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))
            
            with torch.no_grad():
                logits_q = self.net(g, feats, tao_1_L, tao_2_L, tao_1_G, tao_2_G, PPMI, self.net.parameters())
                # logits_q = self.net(g, feats, tao_1, tao_2, self.net.parameters())
                loss_q = F.cross_entropy(logits_q[qry_idx[i]], labels_local[qry_idx[i]])
                losses_q[0] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q[qry_idx[i]], labels_local[qry_idx[i]]).sum().item()
                corrects[0] = corrects[0] + correct

            with torch.no_grad():
                logits_q = self.net(g, feats, tao_1_L, tao_2_L, tao_1_G, tao_2_G, PPMI, fast_weights)
                # logits_q = self.net(g, feats, tao_1, tao_2, fast_weights)
                loss_q = F.cross_entropy(logits_q[qry_idx[i]], labels_local[qry_idx[i]])
                losses_q[1] += loss_q
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q[qry_idx[i]], labels_local[qry_idx[i]]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                logits = self.net(g, feats, tao_1_L, tao_2_L, tao_1_G, tao_2_G, PPMI, fast_weights)
                # logits = self.net(g, feats, tao_1, tao_2, fast_weights)
                loss = F.cross_entropy(logits[spt_idx[i]], labels_local[spt_idx[i]])
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                logits_q = self.net(g, feats, tao_1_L, tao_2_L, tao_1_G, tao_2_G, PPMI, fast_weights)
                # logits_q = self.net(g, feats, tao_1, tao_2, fast_weights)
                loss_q = F.cross_entropy(logits_q[qry_idx[i]], labels_local[qry_idx[i]])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q[qry_idx[i]], labels_local[qry_idx[i]]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        loss_q = (losses_q[-1] / task_num) + self.alpha * loss_gae
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()
        accs = np.array(corrects) / (querysz * task_num)
        return accs
        
        


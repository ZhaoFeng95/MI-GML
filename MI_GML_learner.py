# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:43:02 2020

@author: Administrator
"""
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    return {'m': msg}

def gcn_reduce(node):
    accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
    return {'h': accum}

def reset_parameters(x, x_size):
    stdv = 1. / math.sqrt(x_size)
    return x.data.uniform_(-stdv, stdv) 


class MetaLearner(nn.Module):
    def  __init__(self,
                  n_layer,
                  n_GCN,
                  activation,
                  dropout):
        super(MetaLearner, self).__init__()
        self.n_layer = n_layer
        self.activation = activation
        self.vars = nn.ParameterList()
        
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        for _ in range(2):
            for i in range(self.n_layer):
                w = torch.Tensor(n_GCN[i], n_GCN[i+1])
                b = torch.Tensor(n_GCN[i+1])
                w = reset_parameters(w,w.size(1))
                b = reset_parameters(b,w.size(0))
                self.vars.append(nn.Parameter(w))
                self.vars.append(nn.Parameter(b))
        W_a = torch.Tensor(n_GCN[2]*2, 2)
        W_a = reset_parameters(W_a, W_a.size(1))
        self.vars.append(nn.Parameter(W_a))
        W_c = torch.Tensor(n_GCN[2], n_GCN[3])
        W_c = reset_parameters(W_c, W_c.size(1))
        b_c = torch.Tensor(n_GCN[3])
        b_c = reset_parameters(b_c, b_c.size(0))
        self.vars.append(nn.Parameter(W_c))
        self.vars.append(nn.Parameter(b_c))
        
    def Bi_GCN(self, g, feats, tao_1, tao_2, net_vars, PPMI):
        if PPMI == None:
            idx = 0
        else:
            idx = 4
        h = feats
        for i in range(self.n_layer):
            w, b = net_vars[idx], net_vars[idx + 1]
            if i == 0:
                w = torch.mm(w,tao_1)
            elif i == 1:
                w = torch.mm(w,tao_2)
            if self.dropout:
                h = self.dropout(h)
            if PPMI != None:
                h = torch.mm(h, w)
                h = torch.mm(PPMI, h)
                h = h + b
                if self.activation[i]:
                    h = self.activation[i](h)
                idx += 2
            else:
                g.ndata['h'] = torch.mm(h, w)
                g.update_all(gcn_msg, gcn_reduce)
                h = g.ndata.pop('h')
                h = h + b
                if self.activation[i]:
                    h = self.activation[i](h)
                idx += 2
        return h
    def Attention(self, H_L, H_G, net_vars):
        Attn_input = torch.cat([H_L, H_G],1)
        Attn = torch.mm(Attn_input, net_vars[8])
        Attn = F.softmax(Attn)
        Z = Attn[:,0] * H_L.t() + Attn[:,1] * H_G.t()
        return Z.t()
    
    def forward(self, g, feats, tao_1_L, tao_2_L, tao_1_G, tao_2_G, PPMI, net_vars=None):
        if net_vars is None:
            net_vars = self.vars
        H_L = self.Bi_GCN(g, feats, tao_1_L, tao_2_L, net_vars, None)
        H_G = self.Bi_GCN(g, feats, tao_1_G, tao_2_G, net_vars, PPMI)
        Z = self.Attention(H_L, H_G, net_vars)
        
        Y = torch.mm(Z, net_vars[9]) + net_vars[10]
        
        
        return Y
    
    def parameters(self):
        return self.vars
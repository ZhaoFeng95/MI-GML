# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 17:09:04 2020

@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:45:36 2020

@author: Administrator
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gcn_msg(edge):
    msg = edge.src['h'] * edge.src['norm']
    return {'m': msg}


def gcn_reduce(node):
    accum = torch.sum(node.mailbox['m'], 1) * node.data['norm']
    return {'h': accum}
    

class NodeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}
    
    
class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, g, h, PPMI):
        if self.dropout:
            h = self.dropout(h)
        g.ndata['h'] = torch.mm(h, self.weight)
        g.update_all(gcn_msg, gcn_reduce, self.node_update)
        h = g.ndata.pop('h')
        return h

class GlobalGCNLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dims,
                 activation,
                 dropout):
        super(GlobalGCNLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        self.w = nn.Parameter(torch.Tensor(in_dim, out_dims))
        self.b = nn.Parameter(torch.Tensor(out_dims))
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.b.size(0))
        self.b.data.uniform_(-stdv, stdv)
    
    def forward(self, g, X, PPMI):
        h = self.dropout(X)
        h = torch.mm(h, self.w)
        h = h + self.b
        h = torch.mm(PPMI, h)
        if self.activation:
            h = self.activation(h)
            
        return h
class VGAE(nn.Module):
    def __init__(self, 
                 in_dim,
                 hidden_dims,
                 dropout,
                 isGlobal = False):
        super(VGAE, self).__init__()
        self.isGlobal = isGlobal
        if isGlobal == False:
            self.base_gcn = GCNLayer(in_dim, 128, F.relu, dropout)
            self.gcn_mean = GCNLayer(128, hidden_dims, None, dropout)
            self.gcn_logstddev = GCNLayer(128, hidden_dims, None, dropout)
        else:
            self.base_gcn = GlobalGCNLayer(in_dim, 128, F.relu, dropout)
            self.gcn_mean = GlobalGCNLayer(128, hidden_dims, None, dropout)
            self.gcn_logstddev = GlobalGCNLayer(128, hidden_dims, None, dropout)
            
    def encode(self, g, X, PPMI):
        hidden = self.base_gcn(g, X, PPMI)
        self.mean = self.gcn_mean(g, hidden, PPMI)
        self.logstd = self.gcn_logstddev(g, hidden, PPMI)
        gaussian_noise = torch.randn(X.size(0), self.mean.size(1)).cuda()
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z
        
    def forward(self, g, X, PPMI):
        
        Z = self.encode(g, X, PPMI)
        A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
        return A_pred, Z


    
    
    
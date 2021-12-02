# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:52:35 2020

@author: Administrator
"""
from itertools import combinations
import dgl
import networkx as nx
import numpy as np
from dgl.data import *
from collections import Counter
import torch
import argparse
import torch.nn.functional as F
from dgl import DGLGraph
from MI_GML_utils import data_generator, get_norm, get_PPMI
from MI_GML_meta import MetaGCN


def main(args):
    
    if args.gpu != -1:
        device = torch.device("cuda:" + str(args.gpu))
    else:
        device = torch.device("cpu")
    
        
    if args.datasets == 'citeseer':
        data = citegrh.load_citeseer()
        node_num = 3327
        class_label = [0, 1, 2, 3, 4, 5]
        combination = list(combinations(class_label, 2))
    elif args.datasets == 'cora':
        data = dgl.data.CoraDataset()
        node_num = 2708
        class_label = [0, 1, 2, 3, 4, 5, 6]
        combination = list(combinations(class_label, 2))
    args.node_num = node_num
    
    g = data.graph
    g.remove_edges_from(nx.selfloop_edges(g))
    
    g.add_edges_from(zip(g.nodes(), g.nodes()))
    g = DGLGraph(data.graph)
    adj = g.adjacency_matrix().to_dense()
    PPMI, _ = get_PPMI(adj, 10)
    feats = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    norm = get_norm(g)
    
    args.n_GCN[0] = feats.size(1)
    args.n_GCN[3] = args.n_way
    activation = [F.relu, None]
    
    
    if args.gpu != -1:
        feats =  feats.to(device)
        labels = labels.to(device)
        norm = norm.to(device)
        adj = adj.to(device)
        PPMI = PPMI.to(device)
        g = g.to(device)
        
    g.ndata['norm'] = norm.unsqueeze(1)
    
    
    for i in range(len(combination)):
        model = MetaGCN(args, activation).cuda()
        model.train()
        test_label = list(combination[i])
        train_label = [n for n in class_label if n not in test_label]
        spt_idx, qry_idx, labels_local = data_generator(args, feats, labels, node_num, train_label)
        
        for epoch in range(args.epochs):
            acc = model(g, feats, spt_idx, qry_idx, labels_local, adj, PPMI)
            print("Epoch: {:02d} | Train_Acc: {}"
                  .format(epoch, acc.astype(np.float16)))
            if (epoch+1) % 50 == 0:
                print('\n')
                torch.save(model.state_dict(), 'maml.pkl')
                meta_test_acc = []
                for k in range(args.step):
                    model_meta_trained = MetaGCN(args, activation).cuda()
                    model_meta_trained.load_state_dict(torch.load('maml.pkl'))
                    model_meta_trained.eval()
                    spt_idx, qry_idx, labels_local = data_generator(args, feats, labels, node_num, test_label)
                    accs = model_meta_trained(g, feats, spt_idx, qry_idx, labels_local, adj, PPMI)
                    meta_test_acc.append(accs)
                if args.datasets == 'citeseer':
                    print('Test_Acc: {}'.format(np.array(meta_test_acc).mean(axis=0)))
                    with open('citeseer.txt', 'a') as f:
                        f.write('Cross Validation:{}, Step: {}, Meta-Test_Accuracy: {}'.format(i+1, epoch, np.array(meta_test_acc).mean(axis=0).astype(np.float16)))
                        f.write('\n')
                        f.close()
                elif args.datasets == 'cora':
                    print('Test_Acc: {}'.format(np.array(meta_test_acc).mean(axis=0)))
                    with open('cora.txt', 'a') as f:
                        f.write('Cross Validation:{}, Step: {}, Meta-Test_Accuracy: {}'.format(i+1, epoch, np.array(meta_test_acc).mean(axis=0).astype(np.float16)))
                        f.write('\n')
                        f.close()
      
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--datasets', type=str, default='citeseer', help="Dataset to use")
    parser.add_argument('--node_num', type=int, default=0, help="number of node")
    parser.add_argument("--n_layer", type=int, default=2, help="num of layer")
    parser.add_argument("--task_num", type=int, default=2, help="num of task")
    parser.add_argument("--n_way", type=int, default=2, help="n way")
    parser.add_argument('--k_spt', type=int, default=1, help='k shot for support set')
    parser.add_argument('--k_qry', type=int, default=19, help='k shot for query set')
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
    parser.add_argument('--GAE_lr', type=float, help='GAE learning rate', default=0.01)
    parser.add_argument('--Film_lr', type=float, help='Film learning rate', default=0.003)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--epochs", type=int, default=900, help="number of training epochs")
    parser.add_argument("--n_GCN", type=int, default=[128, 32, 16, 2], help="number of GCN layers")
    parser.add_argument('--step', type=int, default=10, help='How many times to random select node to test')
    
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    
    main(args)

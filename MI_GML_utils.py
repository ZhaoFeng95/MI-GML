# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:40:46 2020

@author: Administrator
"""


import random
import torch
import numpy as np
from collections import Counter


def norm(counter):
    s = sum(counter.values())
    new_counter = Counter()
    for a, count in counter.items():
        new_counter[a] = counter[a] / s
    return new_counter
    

def get_PPMI(adj, path_len):

    adj = np.array(adj)
    PPMI = np.zeros([np.size(adj,0), np.size(adj,0)])
    walk_counters = {}
    for a in range(len(adj)):
        
        current_path_len = np.random.randint(0, path_len)
        # a_neighbors = np.where(adj[a,:] == 1)[0]
        # if len(a_neighbors) == 0:
        #         break
        # for current_a in a_neighbors:
        #     for _ in range(current_path_len):
        #         neighbors = np.where(adj[current_a,:] == 1)[0]
        #         b = np.random.choice(neighbors, 1)
        #         if a in walk_counters:
        #             walk_counter = walk_counters[int(a)]
        #         else:
        #             walk_counter = Counter()
        #             walk_counters[int(a)] = walk_counter
        #         walk_counter[int(b)] += 1
        #         current_a = int(b)
        
        current_a = a
        for _ in range(current_path_len):
            neighbors = np.where(adj[current_a,:] == 1)[0]
            if len(neighbors) == 0:
                break
            b = np.random.choice(neighbors, 1)
            if a in walk_counters:
                walk_counter = walk_counters[int(a)]
            else:
                walk_counter = Counter()
                walk_counters[int(a)] = walk_counter
            walk_counter[int(b)] += 1
            current_a = int(b)
    
    normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}
    prob_sums = Counter()
    for a, normed_walk_counter in normed_walk_counters.items():
        for b, prob in normed_walk_counter.items():
            prob_sums[b] += prob

    for a, normed_walk_counter in normed_walk_counters.items():
        for b, prob in normed_walk_counter.items():
            ppmi = np.log(prob / prob_sums[b] * len(prob_sums) / path_len)
            PPMI[a,b] = ppmi
    PPMI[PPMI<0] = 0
    D_inv = np.diag(np.sum(PPMI,axis=0))
    D_inv = np.power(D_inv, -0.5)
    D_inv[np.isnan(D_inv)] = 0.0
    D_inv[np.isinf(D_inv)] = 0.0
    D_inv[np.isneginf(D_inv)] = 0.0
    PPMI = np.dot(np.dot(D_inv,  PPMI), D_inv)
    return torch.Tensor(PPMI), PPMI
    


def data_generator(args, features, labels, node_num, select_array):
    spt_idx = []
    qry_idx = []
    i = 0
    
    labels_local = labels.clone().detach()
    unselect_class = select_array
    while i < len(select_array):
        class1_idx = []
        class2_idx = []
        if len(unselect_class) >= 2:
            select_class = random.sample(unselect_class, args.n_way)
            unselect_class = [n for n in unselect_class if n not in select_class]
        i += 3 
        for j in range(node_num):
            if (labels_local[j] == select_class[0]):
                class1_idx.append(j)
                labels_local[j] = 0
            elif (labels_local[j] == select_class[1]):
                class2_idx.append(j)
                labels_local[j] = 1
        
        for t in range(args.task_num):
            class1_train = random.sample(class1_idx, args.k_spt)
            class2_train = random.sample(class2_idx, args.k_spt)
            class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
            class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
            class1_test = random.sample(class1_test, args.k_qry)
            class2_test = random.sample(class2_test, args.k_qry)
            train_idx = class1_train + class2_train
            random.shuffle(train_idx)
            test_idx = class1_test + class2_test
            random.shuffle(test_idx)
        
            spt_idx.append(torch.LongTensor(train_idx))
            qry_idx.append(torch.LongTensor(test_idx))
        
    return spt_idx, qry_idx, labels_local

def get_norm(g):
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    return norm


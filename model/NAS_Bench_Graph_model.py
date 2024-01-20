import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.autograd import Variable
from torch_geometric.nn.conv import *

import json

gnn_list = [
    "gat",  # GAT
    "gcn",  # GCN
    "gin",  # GIN
    "cheb",  # chebnet
    "sage",  # sage
    "arma",
    "graph",  
    "fc",  # fully-connected
    "skip"  # skip connection
]

def get_num_features(dataset):
    if dataset == 'Computers':
      return 767
    elif dataset == 'Photo':
      return 745
    elif dataset == 'CS':
      return 6805
    elif dataset == 'Physics':
      return 8415
    elif dataset == 'Cora':
      return 1433
    elif dataset == 'CiteSeer':
      return 3703
    elif dataset == 'PubMed':
      return 500
    else:
       print('No supported dataset')
    return None

def get_num_classes(dataset):
    if dataset == 'Computers':
      return 10
    elif dataset == 'Photo':
      return 8
    elif dataset == 'CS':
      return 15
    elif dataset == 'Physics':
      return 5
    elif dataset == 'Cora':
      return 7
    elif dataset == 'CiteSeer':
      return 6
    elif dataset == 'PubMed':
      return 3
    else:
       print('No supported dataset')
    return None

class HP:
    def __repr__(self):
        return json.dumps(self, default=lambda obj: obj.__dict__)

def get_NAS_Bench_graph_HP():
    hpdict = {"dropout": 0.0, "dim": 256, "num_cells": 1, "num_pre": 1, "num_pro": 1, "lr": 0.01, "wd": 5e-4, "optimizer": "Adam", "num_epochs": 500}

    hp = HP()
    for key in hpdict:
        setattr(hp, key, hpdict[key])
    return hp

def gnn_map(gnn_name, in_dim, out_dim, norm=True, bias=True) -> Module:
    '''
    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    '''
    if gnn_name == "gat":
        return GATConv(in_dim, out_dim // 4, 4, bias=bias, concat = True, add_self_loops=norm)
    elif gnn_name == "gcn":
        return GCNConv(in_dim, out_dim, add_self_loops=True, normalize=norm)
    elif gnn_name == "gin":
        return GINConv(torch.nn.Linear(in_dim, out_dim))
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "monet":
        return GMMConv(in_dim, out_dim, dim = out_dim, kernel_size = 1)
    elif gnn_name == "graph":
        return GraphConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "fc":
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "skip":
        return SkipConv(in_dim, out_dim, bias=bias)
    else:
        raise ValueError("No such GNN name") 

class LinearConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        return self.linear(x)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class SkipConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(SkipConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_weight=None):
        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class Cell(nn.Module):
    def __init__(self, ops, link, hp, in_dim, out_dim, dname):
        super(Cell, self).__init__()
        self.link = link
        self.hp = hp
        self.dname = dname
        self.ops = nn.ModuleList()
        self.out = [True for i in range(len(link) + 1)]
        # dim is in_dim in res
        self.res_in_dim = [False for i in range(len(link) + 1)]
        self.queue_ori_in_dim(ops)
        for i in link:
            self.out[i] = False

        if self.hp.num_pro:
            self.fc = nn.Linear(sum(self.out) * hp.dim, hp.dim)
        else:
            self.fc = nn.Linear(sum(self.out) * hp.dim, out_dim)
        self.bns = nn.ModuleList()
        for op, lk in zip(ops, link):
            if self.res_in_dim[lk]:
                self.ops.append(gnn_map(op, in_dim, hp.dim, dname != 'ogbn-proteins'))
            else:
                self.ops.append(gnn_map(op, hp.dim, hp.dim, dname != 'ogbn-proteins'))
            self.bns.append(nn.BatchNorm1d(hp.dim))

    def queue_ori_in_dim(self, ops):
        if self.hp.num_pre:
            return
        self.res_in_dim[0] = True
        for op, lk, i in zip(ops, self.link, range(len(ops))):
            if self.res_in_dim[lk] and op == 'skip':
                self.res_in_dim[i + 1] = True

    def forward(self, x, data):
        res = [x]
        for op, link, bn in zip(self.ops, self.link, self.bns):
            inp = res[link]
            if self.dname != 'ogbn-proteins' or isinstance(op, GCNConv):
                adjs = data.edge_index
            else:
                adjs = data.adj_t
            if not self.res_in_dim[link]:
                inp = bn(inp)
                inp = F.relu(inp)
                inp = F.dropout(inp, p=self.hp.dropout, training=self.training)
            res.append(op(inp, adjs))  # call the gcn module
        res = sum([[res[i]] if out else [] for i, out in enumerate(self.out)], [])
        fin = torch.cat(res, 1)
        fin = F.relu(fin)
        fin = self.fc(fin)
        return fin

class Net(nn.Module):
    def __init__(self, ops, link, hp, in_dim, out_dim, dname):
        super().__init__()
        self.hp = hp
        self.dname = dname
        self.prep = nn.ModuleList()
        for i in range(hp.num_pre):
            idim = in_dim if i == 0 else hp.dim             
            self.prep.append(LinearConv(idim, hp.dim))

        self.prop = nn.ModuleList()
        for i in range(hp.num_pro):
            odim = out_dim if i == hp.num_pro - 1 else hp.dim             
            self.prop.append(LinearConv(hp.dim, odim))

        self.cells = nn.ModuleList()
        for i in range(hp.num_cells):
            cell = Cell(ops, link, hp, in_dim, out_dim, dname)
            self.cells.append(cell)

    def forward(self, data):
        x, adjs = data.x, data.edge_index
        x = F.dropout(x, p=self.hp.dropout, training=self.training)
        for i, prep in enumerate(self.prep):
            x = prep(x, adjs)
            if i == self.hp.num_pre - 2:
                x = F.elu(x)

        for cell in self.cells:
            x = cell(x, data)

        for i, prop in enumerate(self.prop):
            x = prop(x, adjs)
            if i == self.hp.num_pro - 2:
                x = F.elu(x)
        return x

from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.transforms import RandomNodeSplit, Compose, LargestConnectedComponents, ToSparseTensor, BaseTransform
from torch_geometric.data import Data, HeteroData

from torch_geometric.data.storage import NodeStorage

from torch_geometric.utils import *

class SemiSplit(BaseTransform):
    r"""Performs a node-level random split by adding :obj:`train_mask`,
    :obj:`val_mask` and :obj:`test_mask` attributes to the
    :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData` object
    (functional name: :obj:`random_node_split`).

    Args:
        split (string): The type of dataset split (:obj:`"train_rest"`,
            :obj:`"test_rest"`, :obj:`"random"`).
            If set to :obj:`"train_rest"`, all nodes except those in the
            validation and test sets will be used for training (as in the
            `"FastGCN: Fast Learning with Graph Convolutional Networks via
            Importance Sampling" <https://arxiv.org/abs/1801.10247>`_ paper).
            If set to :obj:`"test_rest"`, all nodes except those in the
            training and validation sets will be used for test (as in the
            `"Pitfalls of Graph Neural Network Evaluation"
            <https://arxiv.org/abs/1811.05868>`_ paper).
            If set to :obj:`"random"`, train, validation, and test sets will be
            randomly generated, according to :obj:`num_train_per_class`,
            :obj:`num_val` and :obj:`num_test` (as in the `"Semi-supervised
            Classification with Graph Convolutional Networks"
            <https://arxiv.org/abs/1609.02907>`_ paper).
            (default: :obj:`"train_rest"`)
        num_splits (int, optional): The number of splits to add. If bigger
            than :obj:`1`, the shape of masks will be
            :obj:`[num_nodes, num_splits]`, and :obj:`[num_nodes]` otherwise.
            (default: :obj:`1`)
        num_train_per_class (int, optional): The number of training samples
            per class in case of :obj:`"test_rest"` and :obj:`"random"` split.
            (default: :obj:`20`)
        num_val (int or float, optional): The number of validation samples.
            If float, it represents the ratio of samples to include in the
            validation set. (default: :obj:`500`)
        num_test (int or float, optional): The number of test samples in case
            of :obj:`"train_rest"` and :obj:`"random"` split. If float, it
            represents the ratio of samples to include in the test set.
            (default: :obj:`1000`)
        key (str, optional): The name of the attribute holding ground-truth
            labels. By default, will only add node-level splits for node-level
            storages in which :obj:`key` is present. (default: :obj:`"y"`).
    """
    def __init__(
        self,
        num_splits: int = 1,
        num_train_per_class: int = 20,
        num_val_per_class: int = 30,
        key: Optional[str] = "y",
        lcc: bool = False,
    ):
        self.split = "hxy"
        self.num_splits = num_splits
        self.num_train_per_class = num_train_per_class
        self.num_val_per_class = num_val_per_class
        self.key = key
        self.lcc = lcc

    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            if self.key is not None and not hasattr(store, self.key):
                continue

            train_masks, val_masks, test_masks = zip(
                *[self._split(store) for _ in range(self.num_splits)])

            store.train_mask = torch.stack(train_masks, dim=-1).squeeze(-1)
            store.val_mask = torch.stack(val_masks, dim=-1).squeeze(-1)
            store.test_mask = torch.stack(test_masks, dim=-1).squeeze(-1)

        return data

    def _split(self, store: NodeStorage) -> Tuple[Tensor, Tensor, Tensor]:
        num_nodes = store.num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        y = getattr(store, self.key)
        num_classes = int(y.max().item()) + 1
        indices = []
        data = store
        if self.lcc:
            data_ori = data
            data_nx = to_networkx(data_ori)
            data_nx = data_nx.to_undirected()
            data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
            lcc_mask = list(data_nx.nodes)
            for i in range(num_classes):
                index = (data.y[lcc_mask] == i).nonzero().view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)
        else:
            for i in range(num_classes):
                index = (data.y == i).nonzero().view(-1)
                index = index[torch.randperm(index.size(0))]
                indices.append(index)

        train_index = torch.cat([i[:self.num_train_per_class] for i in indices], dim=0)
        val_index = torch.cat([i[self.num_train_per_class:self.num_train_per_class + self.num_val_per_class] for i in indices], dim=0)

        rest_index = torch.cat([i[self.num_train_per_class + self.num_val_per_class:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=data.num_nodes)
        val_mask = index_to_mask(val_index, size=data.num_nodes)
        test_mask = index_to_mask(rest_index, size=data.num_nodes)

        return train_mask, val_mask, test_mask

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(split={self.split})'

def add_mask(dname, dataset):
    split_idx = dataset.get_idx_split() 
    dat = dataset[0]
    if dname == 'ogbn-proteins':
        dat.x = dat.adj_t.mean(dim=1)
        dat.adj_t.set_value_(None)
        #dat.adj_t = dat.adj_t.set_diag()

        # Pre-compute GCN normalization.
        adj_t = dat.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)

        setattr(dat, "edge_index", adj_t)
    elif dname == 'ogbn-arxiv':
        setattr(dat, "edge_index", torch.cat((dat.edge_index, dat.edge_index[[1,0]]), dim = 1))
    setattr(dat, "y", dat.y.squeeze())
    setattr(dat, "train_mask", index_to_mask(split_idx["train"], size=dat.num_nodes))
    setattr(dat, "val_mask", index_to_mask(split_idx["valid"], size=dat.num_nodes))
    setattr(dat, "test_mask", index_to_mask(split_idx["test"], size=dat.num_nodes))
    return dat


def get_dataset(dname,datadir = '~/data'):
    if dname[:3] == 'ogb':
        if dname == 'ogbn-arxiv':
            dataset = PygNodePropPredDataset(root=datadir, name=dname)
        elif dname == 'ogbn-proteins':
            dataset = PygNodePropPredDataset(root=datadir, name=dname, transform = ToSparseTensor(attr = 'edge_attr'))
        n_class = dataset.num_classes
        dataset = add_mask(dname, dataset)
    elif dname in ["CS", "Physics"]:
        dataset = Coauthor(root=datadir, name=dname, pre_transform = SemiSplit(num_train_per_class=20, num_val_per_class = 30, lcc = False))
        n_class = dataset.num_classes
        dataset = dataset[0]
    elif dname in ["Photo", "Computers"]:
        dataset = Amazon(root=datadir, name=dname, pre_transform = Compose([LargestConnectedComponents(), SemiSplit(num_train_per_class=20, num_val_per_class = 30, lcc = False)]))
        n_class = dataset.num_classes
        dataset = dataset[0]
    else: # PubMed
        dataset = Planetoid(root=datadir, name=dname)
        n_class = dataset.num_classes
        dataset = dataset[0]
    setattr(dataset, "num_classes", n_class)
    return dataset 
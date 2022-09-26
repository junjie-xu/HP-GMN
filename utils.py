import os

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse as sp
# from sklearn.metrics import roc_auc_score, f1_score
from torch_sparse import SparseTensor


def rand_train_test_idx(label, train_prop=.5, valid_prop=.25, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    labeled_nodes = torch.where(label != -1)[0]

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return {'train': train_idx.numpy(), 'valid': valid_idx.numpy(), 'test': test_idx.numpy()}

def index_to_mask(splits_lst, num_nodes):
    mask_len = len(splits_lst)
    train_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)
    val_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)
    test_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)

    for i in range(mask_len):
        train_mask[i][splits_lst[i]['train']] = True
        val_mask[i][splits_lst[i]['valid']] = True
        test_mask[i][splits_lst[i]['test']] = True

    return train_mask.T, val_mask.T, test_mask.T


# def eval_rocauc(y_true, y_pred):
#     """ adapted from ogb
#     https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
#     rocauc_list = []
#     y_true = y_true.detach().cpu().numpy()
#     if y_true.shape[1] == 1:
#         # use the predicted class for single-class classification
#         y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
#     else:
#         y_pred = y_pred.detach().cpu().numpy()

#     for i in range(y_true.shape[1]):
#         # AUC is only defined when there is at least one positive data.
#         if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
#             is_labeled = y_true[:, i] == y_true[:, i]
#             score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

#             rocauc_list.append(score)

#     if len(rocauc_list) == 0:
#         raise RuntimeError(
#             'No positively labeled data available. Cannot compute ROC-AUC.')

#     return sum(rocauc_list) / len(rocauc_list)


def eval_acc(y_true, y_pred):

    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)

from typing import Optional, Callable

import os.path as osp

import torch
import numpy as np

from torch_geometric.utils import to_undirected
from torch_geometric.data import InMemoryDataset, download_url, Data


class WikipediaNetwork2(InMemoryDataset):
    r"""The Wikipedia networks introduced in the
    `"Multi-scale Attributed Node Embedding"
    <https://arxiv.org/abs/1909.13021>`_ paper.
    Nodes represent web pages and edges represent hyperlinks between them.
    Node features represent several informative nouns in the Wikipedia pages.
    The task is to predict the average daily traffic of the web page.
    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"chameleon"`,
            :obj:`"crocodile"`, :obj:`"squirrel"`).
        geom_gcn_preprocess (bool): If set to :obj:`True`, will load the
            pre-processing data as introduced in the `"Geom-GCN: Geometric
            Graph Convolutional Networks" <https://arxiv.org/abs/2002.05287>_`,
            in which the average monthly traffic of the web page is converted
            into five categories to predict.
            If set to :obj:`True`, the dataset :obj:`"crocodile"` is not
            available.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    raw_url = 'https://graphmining.ai/datasets/ptg/wiki'
    processed_url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
                     'geom-gcn/master')

    def __init__(self, root: str, name: str, geom_gcn_preprocess: bool = True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.geom_gcn_preprocess = geom_gcn_preprocess
        assert self.name in ['chameleon', 'crocodile', 'squirrel']
        if geom_gcn_preprocess and self.name == 'crocodile':
            raise AttributeError("The dataset 'crocodile' is not available in "
                                 "case 'geom_gcn_preprocess=True'")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'raw')
        else:
            return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        if self.geom_gcn_preprocess:
            return osp.join(self.root, self.name, 'geom_gcn', 'processed')
        else:
            return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        if self.geom_gcn_preprocess:
            return (['out1_node_feature_label.txt', 'out1_graph_edges.txt'] +
                    [f'{self.name}_split_0.6_0.2_{i}.npz' for i in range(10)])
        else:
            return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        if self.geom_gcn_preprocess:
            for filename in self.raw_file_names[:2]:
                url = f'{self.processed_url}/new_data/{self.name}/{filename}'
                download_url(url, self.raw_dir)
            for filename in self.raw_file_names[2:]:
                url = f'{self.processed_url}/splits/{filename}'
                download_url(url, self.raw_dir)
        else:
            download_url(f'{self.raw_url}/{self.name}.npz', self.raw_dir)

    def process(self):
        if self.geom_gcn_preprocess:
            with open(self.raw_paths[0], 'r') as f:
                data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)
            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

            with open(self.raw_paths[1], 'r') as f:
                data = f.read().split('\n')[1:-1]
                data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            # edge_index = to_undirected(edge_index, num_nodes=x.size(0))
            print('test')
            train_masks, val_masks, test_masks = [], [], []
            for filepath in self.raw_paths[2:]:
                f = np.load(filepath)
                train_masks += [torch.from_numpy(f['train_mask'])]
                val_masks += [torch.from_numpy(f['val_mask'])]
                test_masks += [torch.from_numpy(f['test_mask'])]
            train_mask = torch.stack(train_masks, dim=1).to(torch.bool)
            val_mask = torch.stack(val_masks, dim=1).to(torch.bool)
            test_mask = torch.stack(test_masks, dim=1).to(torch.bool)

            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)

        else:
            data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
            x = torch.from_numpy(data['features']).to(torch.float)
            edge_index = torch.from_numpy(data['edges']).to(torch.long)
            edge_index = edge_index.t().contiguous()
            # edge_index = to_undirected(edge_index, num_nodes=x.size(0))
            y = torch.from_numpy(data['label']).to(torch.float)
            train_mask = torch.from_numpy(data['train_mask']).to(torch.bool)
            test_mask = torch.from_numpy(data['test_mask']).to(torch.bool)
            val_mask = torch.from_numpy(data['val_mask']).to(torch.bool)
            data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                        val_mask=val_mask, test_mask=test_mask)
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

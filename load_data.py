import torch.nn.functional as F
from utils import rand_train_test_idx, index_to_mask, WikipediaNetwork2
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork, Actor
import scipy.io
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data
import torch_geometric.transforms as T
import scipy.io
import numpy as np
import scipy.sparse
import torch
import csv
import json


def load_dataset(dataname, train_prop=.5, valid_prop=.25, num_masks=5):
    assert dataname in ('Cora', 'Citeseer', 'Pubmed', 'texas', 'wisconsin', 'cornell', 'squirrel',
                        'chameleon', 'crocodile', 'computers', 'photo'), 'Invalid dataset'

    if dataname in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='./data/', name=dataname)
        data = dataset[0]
        data.train_mask = torch.unsqueeze(data.train_mask, dim=1)
        data.val_mask = torch.unsqueeze(data.val_mask, dim=1)
        data.test_mask = torch.unsqueeze(data.test_mask, dim=1)

        # splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop)
        #               for _ in range(num_masks)]
        # data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['texas', 'wisconsin', 'cornell']:
        dataset = WebKB(root='./data/', name=dataname)
        data = dataset[0]

    elif dataname in ['squirrel', 'chameleon']:
        dataset = WikipediaNetwork(root='./data/', name=dataname, geom_gcn_preprocess=True)
        data = dataset[0]

    elif dataname in ['crocodile']:
        dataset = WikipediaNetwork2(root='./data/', name=dataname, geom_gcn_preprocess=False)
        data = dataset[0]
        data.y = data.y.long()

        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['computers', 'photo']:
        dataset = Amazon(root='./data/', name=dataname)
        data = dataset[0]
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)


    data.n_id = torch.arange(data.num_nodes)

    return data






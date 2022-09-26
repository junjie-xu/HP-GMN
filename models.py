import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import APPNP
from torch_geometric.nn import GCNConv
from torch_scatter import scatter




class MLP_simple(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=.3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.lin1 = nn.Linear(self.in_channels, self.hidden_channels)
        self.lin2 = nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x



class GMN_ORTHO(torch.nn.Module):
    def __init__(self, x, edge_index, num_nodes, num_features, num_classes, y_pseudo, centers, cluster_labels,
                 K, hidden, dropout, ppr_alpha, local_stat_num, memory_hidden, device):
        super(GMN_ORTHO, self).__init__()
        self.K = K
        self.hidden_dim = hidden
        self.dropout = dropout
        self.ppr_alpha = ppr_alpha
        self.local_stat_num = local_stat_num
        self.device = device
        self.memory_hidden = memory_hidden

        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes

        self.lin1 = Linear(self.num_features, self.hidden_dim)
        self.lin2 = Linear(self.num_classes, self.hidden_dim)
        self.lin3 = Linear(self.num_features * self.num_classes, self.hidden_dim)
        self.lin4 = Linear(self.num_nodes, self.hidden_dim)

        self.lin5 = Linear(self.hidden_dim, self.hidden_dim)
        self.lin6 = Linear(self.hidden_dim, self.hidden_dim)
        self.lin7 = Linear(self.hidden_dim, self.hidden_dim)
        self.lin8 = Linear(self.hidden_dim, self.hidden_dim)

        self.lin9 = Linear(self.hidden_dim * self.local_stat_num, self.memory_hidden)
        self.lin10 = Linear(self.memory_hidden, self.memory_hidden)

        self.lin11 = Linear(self.memory_hidden, self.memory_hidden)
        self.lin12 = Linear(self.memory_hidden, self.memory_hidden)


        self.lin13 = Linear(self.memory_hidden * 2, self.memory_hidden)
        self.lin14 = Linear(self.memory_hidden * 2, self.num_classes)


        self.memory = torch.rand((self.K, self.memory_hidden)).to(self.device).requires_grad_(True)
        torch.nn.init.xavier_uniform_(self.memory)


        self.neighbor_num_labelwise, self.neighbor_features_labelwise = self.num_neighbor_classes(x, edge_index, y_pseudo)
        self.diffusion_matrix = self.diffusion(edge_index)


    def num_neighbor_classes(self, x, edge_index, y_pseudo):
        neighbor_num_labelwise = torch.zeros((self.num_nodes, self.num_classes), device=self.device)
        neighbor_features_labelwise = torch.zeros((self.num_nodes, self.num_classes * self.num_features), device=self.device)
        for i in range(self.num_nodes):
            neighbors = edge_index[1][edge_index[0] == i]
            neighbors_class = y_pseudo[neighbors]
            neighbor_num = scatter(torch.ones(neighbors.size(0), device=self.device), neighbors_class, dim=0, reduce="sum")
            neighbor_num_labelwise[i][:neighbor_num.size(0)] = neighbor_num
            neighbor_features = scatter(x[neighbors], neighbors_class, dim=0, reduce="mean")
            neighbor_features = neighbor_features.view(-1)
            neighbor_features_labelwise[i][:neighbor_features.size(0)] = neighbor_features

        return neighbor_num_labelwise, neighbor_features_labelwise

    def ask_memory(self, query):
        attention_score = F.softmax(torch.mm(self.memory, query.t()), dim=0)

        memory_weight = attention_score.mean(dim=1)
        memory_weight = memory_weight / memory_weight.sum()
        memory_entropy = torch.special.entr(memory_weight).sum()

        value = torch.mm(attention_score.t(), self.memory)
        return value, -memory_entropy

    def kpattern_distance(self, query):
        dis_matrix = torch.cdist(self.memory, query)
        dis, _ = torch.min(dis_matrix, dim=1)
        return torch.mean(dis)


    def diffusion(self, edge_index):
        gdc = T.GDC(self_loop_weight=1, normalization_in='sym', normalization_out='col',
                    diffusion_kwargs=dict(method='ppr', alpha=self.ppr_alpha))
        diffusion_matrix = gdc.diffusion_matrix_exact(edge_index, torch.ones(edge_index.size(1), device=self.device),
                                       self.num_nodes, method='ppr', alpha=self.ppr_alpha)
        return diffusion_matrix


    def forward(self, x, edge_index):
        x_transformed = self.lin5(F.dropout(F.relu(self.lin1(x)), p=self.dropout, training=self.training))
        neighbor_num_labelwise = self.lin6(F.dropout(F.relu(self.lin2(self.neighbor_num_labelwise)), p=self.dropout, training=self.training))
        neighbor_features_labelwise = self.lin7(F.dropout(F.relu(self.lin3(self.neighbor_features_labelwise)), p=self.dropout, training=self.training))
        diffusion_matrix = self.lin8(F.dropout(F.relu(self.lin4(self.diffusion_matrix)), p=self.dropout, training=self.training))

        query = torch.cat([x_transformed, neighbor_num_labelwise, neighbor_features_labelwise, diffusion_matrix], dim=1)
        query = self.lin9(query)
        # query = self.lin10(F.dropout(F.relu(query), p=self.dropout, training=self.training))


        value, entropy_loss = self.ask_memory(query)
        value = self.lin11(value)
        # value = self.lin12(F.dropout(F.relu(value), p=self.dropout, training=self.training))


        h = torch.cat([query, value], dim=1)
        # h = self.lin13(h)
        h = self.lin14(F.relu(F.dropout(h, p=self.dropout, training=self.training)))

        kpattern_loss = self.kpattern_distance(query)
        regu_loss = torch.norm(self.memory)

        return F.log_softmax(h, dim=1), kpattern_loss, entropy_loss, regu_loss



# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv, GraphConv, TransformerConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, activation_name):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.activation = get_activation(activation_name)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(
                GCNConv(
                    first_channels,
                    second_channels,
                    normalize=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x = self.activation(x)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, activation_name):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.activation = get_activation(activation_name)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(first_channels, second_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x = self.activation(x)
        return x


class WSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, activation_name):
        super(WSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.activation = get_activation(activation_name)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(GraphConv(first_channels, second_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x = self.activation(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, activation_name):
        super(Transformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.activation = get_activation(activation_name)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(TransformerConv(first_channels, second_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        x = self.activation(x)
        return x


class MLPPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, activation_name):
        super(MLPPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.activation = get_activation(activation_name)
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.lins.append(torch.nn.Linear(first_channels, second_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class MLPCatPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, activation_name):
        super(MLPCatPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.activation = get_activation(activation_name)
        in_channels = 2 * in_channels
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.lins.append(torch.nn.Linear(first_channels, second_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x1 = torch.cat([x_i, x_j], dim=-1)
        x2 = torch.cat([x_j, x_i], dim=-1)
        for lin in self.lins[:-1]:
            x1, x2 = lin(x1), lin(x2)
            x1, x2 = self.activation(x1), self.activation(x2)
            x1 = F.dropout(x1, p=self.dropout, training=self.training)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x1 = self.lins[-1](x1)
        x2 = self.lins[-1](x2)
        x = (x1 + x2)/2
        return x


class MLPDotPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, activation_name):
        super(MLPDotPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.activation = get_activation(activation_name)
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(
                torch.nn.Linear(
                    hidden_channels,
                    hidden_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        for lin in self.lins:
            x_i, x_j = lin(x_i), lin(x_j)
            x_i, x_j = self.activation(x_i), self.activation(x_j)
            x_i, x_j = F.dropout(x_i, p=self.dropout, training=self.training), \
                F.dropout(x_j, p=self.dropout, training=self.training)
        x = torch.sum(x_i * x_j, dim=-1)
        return x


class MLPBilPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 dropout, activation_name):
        super(MLPBilPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.activation = get_activation(activation_name)
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.lins.append(
                torch.nn.Linear(
                    hidden_channels,
                    hidden_channels))
        self.bilin = torch.nn.Linear(
            hidden_channels, hidden_channels, bias=False)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        self.bilin.reset_parameters()

    def forward(self, x_i, x_j):
        for lin in self.lins:
            x_i, x_j = lin(x_i), lin(x_j)
            x_i, x_j = self.activation(x_i), self.activation(x_j)
            x_i, x_j = F.dropout(x_i, p=self.dropout, training=self.training), \
                F.dropout(x_j, p=self.dropout, training=self.training)
        x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        return x


class DotPredictor(torch.nn.Module):
    def __init__(self):
        super(DotPredictor, self).__init__()

    def reset_parameters(self):
        return

    def forward(self, x_i, x_j):
        x = torch.sum(x_i * x_j, dim=-1)
        return x


class BilinearPredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BilinearPredictor, self).__init__()
        self.bilin = torch.nn.Linear(
            hidden_channels, hidden_channels, bias=False)

    def reset_parameters(self):
        self.bilin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.sum(self.bilin(x_i) * x_j, dim=-1)
        return x


def get_activation(activation_name):
    if activation_name.lower() == 'elu':
        return torch.nn.ELU()
    else:
        return torch.nn.ReLU()


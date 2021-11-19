# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        if num_layers < 2:
            self.convs.append(
                GCNConv(
                    in_channels,
                    out_channels,
                    normalize=False))
        else:
            self.convs.append(
                GCNConv(
                    in_channels,
                    hidden_channels,
                    normalize=False))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(
                        hidden_channels,
                        hidden_channels,
                        normalize=False))
            self.convs.append(
                GCNConv(
                    hidden_channels,
                    out_channels,
                    normalize=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class DIRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(DIRGCN, self).__init__()
        self.in_convs = torch.nn.ModuleList()
        self.out_convs = torch.nn.ModuleList()
        if num_layers < 2:
            self.in_convs.append(
                GCNConv(
                    in_channels,
                    out_channels,
                    normalize=False))
            self.out_convs.append(
                GCNConv(
                    in_channels,
                    out_channels,
                    normalize=False))
        else:
            self.in_convs.append(
                GCNConv(
                    in_channels,
                    hidden_channels,
                    normalize=False))
            self.out_convs.append(
                GCNConv(
                    in_channels,
                    hidden_channels,
                    normalize=False))
            for _ in range(num_layers - 2):
                self.in_convs.append(
                    GCNConv(
                        hidden_channels,
                        hidden_channels,
                        normalize=False))
                self.out_convs.append(
                    GCNConv(
                        hidden_channels,
                        hidden_channels,
                        normalize=False))
            self.in_convs.append(
                GCNConv(
                    hidden_channels,
                    out_channels,
                    normalize=False))
            self.out_convs.append(
                GCNConv(
                    hidden_channels,
                    out_channels,
                    normalize=False))
        self.dropout = dropout

    def reset_parameters(self):
        for in_conv, out_conv in zip(self.in_convs, self.out_convs):
            in_conv.reset_parameters()
            out_conv.reset_parameters()

    def forward(self, x, in_adj_t, out_adj_t):
        for in_conv, out_conv in zip(self.in_convs[:-1], self.out_convs[:-1]):
            h_in, h_out = in_conv(x, in_adj_t), out_conv(x, out_adj_t)
            h_in, h_out = F.relu(h_in), F.relu(h_out)
            h_in, h_out = F.dropout(h_in, p=self.dropout, training=self.training), \
                F.dropout(h_out, p=self.dropout, training=self.training)
        h_in, h_out = self.in_convs[-1](h_in, in_adj_t), \
            self.out_convs[-1](h_out, out_adj_t)
        h = torch.cat((h_in, h_out), dim=-1)
        return h


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        if num_layers < 2:
            self.convs.append(SAGEConv(in_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class MLPPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLPPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
        if num_layers < 2:
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        else:
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(
                    torch.nn.Linear(
                        hidden_channels,
                        hidden_channels))
            self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class MLPDotPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLPDotPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
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
            x_i, x_j = F.relu(x_i), F.relu(x_j)
            x_i, x_j = F.dropout(x_i, p=self.dropout, training=self.training), \
                F.dropout(x_j, p=self.dropout, training=self.training)
        x = torch.sum(x_i * x_j, dim=-1)
        return x


class MLPBilPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLPBilPredictor, self).__init__()
        self.lins = torch.nn.ModuleList()
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
            x_i, x_j = F.relu(x_i), F.relu(x_j)
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

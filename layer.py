from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch

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
                self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
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

class DotPredictor(torch.nn.Module):
    def __init__(self):
        super(DotPredictor, self).__init__()

    def reset_parameters(self):
        return

    def forward(self, x_i, x_j):
        x = torch.sum(x_i * x_j, dim = -1)
        return x

class BilinearPredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(BilinearPredictor, self).__init__()
        self.lin = torch.nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = torch.sum(self.lin(x_i) * x_j, dim = -1)
        return x
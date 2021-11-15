# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from ogbk.layer import *
from ogbk.negative_sample import *
from ogbk.loss import *


class Model(object):
    """
        Parameters
        ----------
        lr : double
            Learning rate
        dropout : double
            dropout probability for gnn and mlp layers
        gnn_num_layers : int
            number of gnn layers
        mlp_num_layers : int
            number of gnn layers
        hidden_channels : int
            dimension of hidden
        batch_size : int
            batch size.
        num_nodes : int
            number of graph nodes
        num_node_features : int
            dimension of raw node features
        num_neg : int
            number of negative samples for one positive sample
        gnn_encoder : str
            gnn encoder name
        predictor: str
            link predictor name
        loss_func: str
            loss function name
        neg_sampler: str
            negative sampling strategy name
        optimizer: str
            optimization method name
        device: str
            device name: gpu or cpu
        use_node_features: bool
            whether to use raw node features as input
        train_node_emb:
            whether to train node embeddings based on node id
    """

    def __init__(self, lr, dropout, gnn_num_layers, mlp_num_layers, hidden_channels, batch_size, num_nodes, num_node_features,
                 num_neg, gnn_encoder, predictor, loss_func, neg_sampler, optimizer, device, use_node_features, train_node_emb):
        self.lr = lr
        self.dropout = dropout
        self.gnn_num_layers = gnn_num_layers
        self.mlp_num_layers = mlp_num_layers
        self.optimizer_name = optimizer
        self.hidden_channels = hidden_channels
        self.encoder_name = gnn_encoder
        self.predictor_name = predictor
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.loss_func_name = loss_func
        self.neg_sampler_name = neg_sampler
        self.num_nodes = num_nodes
        self.use_node_features = use_node_features
        self.num_node_features = num_node_features
        self.train_node_emb = train_node_emb
        self.device = device

        if self.use_node_features:
            self.input_dim = self.num_node_features
            if self.train_node_emb:
                self.emb = torch.nn.Embedding(
                    self.num_nodes,
                    self.hidden_channels).to(
                    self.device)
                self.input_dim += self.hidden_channels
            else:
                self.emb = None
        else:
            self.emb = torch.nn.Embedding(
                self.num_nodes,
                self.hidden_channels).to(
                self.device)
            self.input_dim = self.hidden_channels

        if self.encoder_name == 'GCN':
            self.encoder = GCN(self.input_dim, self.hidden_channels,
                               self.hidden_channels, self.gnn_num_layers,
                               self.dropout).to(self.device)
        else:
            self.encoder = SAGE(self.input_dim, self.hidden_channels,
                                self.hidden_channels, self.gnn_num_layers,
                                self.dropout).to(self.device)

        if self.predictor_name == 'DOT':
            self.predictor = DotPredictor().to(self.device)
        elif self.predictor_name == 'BIL':
            self.predictor = BilinearPredictor(
                self.hidden_channels).to(
                self.device)
        else:
            self.predictor = MLPPredictor(
                self.hidden_channels,
                self.hidden_channels,
                1,
                self.mlp_num_layers,
                self.dropout).to(
                self.device)

    def param_init(self):
        self.para_list = list(self.encoder.parameters()) + \
            list(self.predictor.parameters())
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()
        if not self.use_node_features:
            torch.nn.init.xavier_uniform_(self.emb.weight)
            self.para_list += list(self.emb.parameters())
        if self.optimizer_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.para_list, lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.para_list, lr=self.lr)

    def train(self, data, split_edge):
        self.encoder.train()
        self.predictor.train()

        pos_train_edge = split_edge['train']['edge']
        if self.neg_sampler_name == 'local':
            neg_train_edge = local_random_neg_sample(
                pos_train_edge,
                num_nodes=self.num_nodes,
                num_neg=self.num_neg).to(
                self.device)
        elif self.neg_sampler_name == 'global':
            neg_train_edge = global_neg_sample(
                data.edge_index,
                num_nodes=self.num_nodes,
                num_samples=pos_train_edge.size(0),
                num_neg=self.num_neg).to(
                self.device)
        else:
            neg_train_edge = global_perm_neg_sample(
                data.edge_index,
                num_nodes=self.num_nodes,
                num_samples=pos_train_edge.size(0),
                num_neg=self.num_neg).to(
                self.device)

        neg_train_edge = torch.reshape(neg_train_edge, (-1, self.num_neg, 2))

        if self.use_node_features:
            if self.train_node_emb:
                input_feat = torch.cat(
                    [self.emb.weight, data.x.to(self.device)], dim=-1)
            else:
                input_feat = data.x.to(self.device)
        else:
            input_feat = self.emb.weight

        total_loss = total_examples = 0
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size,
                               shuffle=True):
            self.optimizer.zero_grad()
            h = self.encoder(input_feat, data.adj_t)
            pos_edge = pos_train_edge[perm].t()
            neg_edge = torch.reshape(neg_train_edge[perm], (-1, 2)).t()
            pos_out = self.predictor(h[pos_edge[0]], h[pos_edge[1]])
            neg_out = self.predictor(h[neg_edge[0]], h[neg_edge[1]])
            if self.loss_func_name == 'CE':
                loss = ce_loss(pos_out, neg_out)
            elif self.loss_func_name == 'InfoNCE':
                loss = info_nce_loss(pos_out, neg_out, self.num_neg)
            elif self.loss_func_name == 'LogRank':
                loss = log_rank_loss(pos_out, neg_out, self.num_neg)
            else:
                loss = auc_loss(pos_out, neg_out, self.num_neg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    @torch.no_grad()
    def test(self, data, split_edge, evaluator):
        self.encoder.eval()
        self.predictor.eval()

        if self.use_node_features:
            if self.train_node_emb:
                input_feat = torch.cat(
                    [self.emb.weight, data.x.to(self.device)], dim=-1)
            else:
                input_feat = data.x.to(self.device)
        else:
            input_feat = self.emb.weight

        h = self.encoder(input_feat, data.adj_t)

        pos_train_edge = split_edge['train']['edge'].to(self.device)
        pos_valid_edge = split_edge['valid']['edge'].to(self.device)
        neg_valid_edge = split_edge['valid']['edge_neg'].to(self.device)
        pos_test_edge = split_edge['test']['edge'].to(self.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(self.device)

        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size):
            edge = pos_train_edge[perm].t()
            pos_train_preds += [self.predictor(h[edge[0]],
                                               h[edge[1]]).squeeze().cpu()]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), self.batch_size):
            edge = pos_valid_edge[perm].t()
            pos_valid_preds += [self.predictor(h[edge[0]],
                                               h[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), self.batch_size):
            edge = neg_valid_edge[perm].t()
            neg_valid_preds += [self.predictor(h[edge[0]],
                                               h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        h = self.encoder(input_feat, data.full_adj_t)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), self.batch_size):
            edge = pos_test_edge[perm].t()
            pos_test_preds += [self.predictor(h[edge[0]],
                                              h[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), self.batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [self.predictor(h[edge[0]],
                                              h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        results = {}
        for topK in [20, 50, 100]:
            evaluator.K = topK
            train_hits = evaluator.eval({
                'y_pred_pos': pos_train_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{topK}']
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{topK}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{topK}']

            results[f'Hits@{topK}'] = (train_hits, valid_hits, test_hits)

        return results

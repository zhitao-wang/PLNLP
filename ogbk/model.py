# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader
from ogbk.layer import *
from ogbk.loss import *
from ogbk.utils import *


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
        elif self.predictor_name == 'MLP':
            self.predictor = MLPPredictor(
                self.hidden_channels,
                self.hidden_channels,
                1,
                self.mlp_num_layers,
                self.dropout).to(
                self.device)
        elif self.predictor_name == 'MLPDOT':
            self.predictor = MLPDotPredictor(
                self.hidden_channels,
                self.hidden_channels,
                1,
                self.mlp_num_layers,
                self.dropout).to(
                self.device)
        elif self.predictor_name == 'MLPBIL':
            self.predictor = MLPBilPredictor(
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

        pos_train_edge, neg_train_edge = get_pos_neg_edges('train', split_edge,
                                                           edge_index=data.edge_index,
                                                           num_nodes=self.num_nodes,
                                                           neg_sampler_name=self.neg_sampler_name,
                                                           num_neg=self.num_neg)
        pos_train_edge = pos_train_edge.to(self.device)
        neg_train_edge = neg_train_edge.to(self.device)

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
    def test(self, data, split_edge, evaluator, eval_metric):
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

        pos_valid_edge, neg_valid_edge = get_pos_neg_edges('valid', split_edge)
        pos_test_edge, neg_test_edge = get_pos_neg_edges('test', split_edge)
        pos_valid_edge = pos_valid_edge.to(self.device)
        neg_valid_edge = neg_valid_edge.to(self.device)
        pos_test_edge = pos_test_edge.to(self.device)
        neg_test_edge = neg_test_edge.to(self.device)

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

        h = self.encoder(input_feat, data.adj_t)

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

        if eval_metric == 'hits':
            results = evaluate_hits(
                evaluator,
                pos_valid_pred,
                neg_valid_pred,
                pos_test_pred,
                neg_test_pred)
        elif eval_metric == 'mrr':
            results = evaluate_mrr(
                evaluator,
                pos_valid_pred,
                neg_valid_pred,
                pos_test_pred,
                neg_test_pred)

        return results

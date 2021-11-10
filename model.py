from layer import *
from negative_sample import *
from loss import *
from torch.utils.data import DataLoader

class Model(object):
    def __init__(self, config):
        self.lr = config.lr
        self.dropout = config.dropout
        self.gnn_num_layers = config.gnn_num_layers
        self.mlp_num_layers = config.mlp_num_layers
        self.optimizer_name = config.optimizer
        self.hidden_channels = config.hidden_channels
        self.encoder_name = config.encoder
        self.predictor_name = config.predictor
        self.batch_size = config.batch_size
        self.num_neg = config.num_neg
        self.loss_func_name = config.loss_func
        self.neg_sampler_name = config.neg_sampler
        self.num_nodes = config.num_nodes
        self.use_node_features = config.use_node_features
        self.num_node_features = config.num_node_features

        self.device = config.device

        if self.use_node_features:
            self.input_dim = self.num_node_features
        else:
            self.emb = torch.nn.Embedding(self.num_nodes, self.hidden_channels).to(self.device)
            self.input_dim = self.hidden_channels

        self.encoder = SAGE(self.input_dim, self.hidden_channels,
                       self.hidden_channels, self.gnn_num_layers,
                       self.dropout).to(self.device)

        if self.predictor_name == 'DOT':
            self.predictor = DotPredictor().to(self.device)
        elif self.predictor_name == 'BIL':
            self.predictor = BilinearPredictor(self.hidden_channels).to(self.device)
        else:
            self.predictor = MLPPredictor(self.hidden_channels, self.hidden_channels, 1,
                                     self.mlp_num_layers, self.dropout).to(self.device)


    def param_init(self):
        self.para_list = list(self.encoder.parameters()) + list(self.predictor.parameters())
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
            neg_train_edge = local_random_neg_sample(pos_train_edge, num_nodes=self.num_nodes, num_neg=self.num_neg)
        elif self.neg_sampler_name == 'global':
            neg_train_edge = global_neg_sample(data.edge_index, num_nodes=self.num_nodes, num_samples=pos_train_edge.size(0), num_neg=self.num_neg)
        else:
            neg_train_edge = global_perm_neg_sample(data.edge_index, num_nodes=self.num_nodes, num_samples=pos_train_edge.size(0), num_neg=self.num_neg)

        neg_train_edge = torch.reshape(neg_train_edge, (-1, self.num_neg, 2))
        pos_train_edge = pos_train_edge.to(self.device)
        neg_train_edge = neg_train_edge.to(self.device)

        if self.use_node_features:
            input_feat = data.x
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
            input_feat = data.x
        else:
            input_feat = self.emb.weight

        h = self.encoder(input_feat, data.adj_t)

        pos_train_edge = split_edge['train']['edge'].to(h.device)
        pos_valid_edge = split_edge['valid']['edge'].to(h.device)
        neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
        pos_test_edge = split_edge['test']['edge'].to(h.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

        pos_train_preds = []
        for perm in DataLoader(range(pos_train_edge.size(0)), self.batch_size):
            edge = pos_train_edge[perm].t()
            pos_train_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_train_pred = torch.cat(pos_train_preds, dim=0)

        pos_valid_preds = []
        for perm in DataLoader(range(pos_valid_edge.size(0)), self.batch_size):
            edge = pos_valid_edge[perm].t()
            pos_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

        neg_valid_preds = []
        for perm in DataLoader(range(neg_valid_edge.size(0)), self.batch_size):
            edge = neg_valid_edge[perm].t()
            neg_valid_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

        h = self.encoder(input_feat, data.full_adj_t)

        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), self.batch_size):
            edge = pos_test_edge[perm].t()
            pos_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)

        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), self.batch_size):
            edge = neg_test_edge[perm].t()
            neg_test_preds += [self.predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        results = {}
        for K in [20, 50, 100]:
            evaluator.K = K
            train_hits = evaluator.eval({
                'y_pred_pos': pos_train_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            valid_hits = evaluator.eval({
                'y_pred_pos': pos_valid_pred,
                'y_pred_neg': neg_valid_pred,
            })[f'hits@{K}']
            test_hits = evaluator.eval({
                'y_pred_pos': pos_test_pred,
                'y_pred_neg': neg_test_pred,
            })[f'hits@{K}']

            results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

        return results
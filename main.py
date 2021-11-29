# -*- coding: utf-8 -*-
import argparse
import time
import torch
import os
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.nn.pool.avg_pool import avg_pool_neighbor_x
from torch_sparse import coalesce, SparseTensor
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from ogbk.logger import Logger
from ogbk.model import BaseModel, NCModel
from ogbk.utils import gcn_normalization


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='GraphSage')
    parser.add_argument('--predictor', type=str, default='DOT')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_func', type=str, default='AUC')
    parser.add_argument('--neg_sampler', type=str, default='global_perm')
    parser.add_argument('--data_name', type=str, default='ogbl-ppa')
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--eval_metric', type=str, default='hits')
    parser.add_argument('--model', type=str, default='base')
    parser.add_argument('--res_dir', type=str, default='')
    parser.add_argument('--pretrain_emb', type=str, default='')
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--mlp_num_layers', type=int, default=1)
    parser.add_argument('--num_hops', type=int, default=1)
    parser.add_argument('--emb_hidden_channels', type=int, default=256)
    parser.add_argument('--gnn_hidden_channels', type=int, default=256)
    parser.add_argument('--mlp_hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--year', type=int, default=-1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--use_node_feats', type=str2bool, default=False)
    parser.add_argument('--use_coalesce', type=str2bool, default=False)
    parser.add_argument('--train_node_emb', type=str2bool, default=False)
    parser.add_argument('--node_feat_trans', type=str2bool, default=False)
    parser.add_argument('--pre_aggregate', type=str2bool, default=False)
    parser.add_argument('--only_neg_train_nodes', type=str2bool, default=False)
    parser.add_argument(
        '--use_valedges_as_input',
        type=str2bool,
        default=False)
    args = parser.parse_args()
    return args


def main():
    args = argument()
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name=args.data_name, root=args.data_path)
    data = dataset[0]

    if hasattr(data, 'edge_weight'):
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)

    data = T.ToSparseTensor()(data)
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack([col, row], dim=0)

    if hasattr(data, 'num_features'):
        num_node_feats = data.num_features
    else:
        num_node_feats = 0

    print(num_node_feats)

    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    else:
        num_nodes = data.adj_t.size(0)

    split_edge = dataset.get_edge_split()
    print(args)
    log_file_name = 'log_' + str(int(time.time())) + '.txt'
    log_file = os.path.join(args.res_dir, log_file_name)
    with open(log_file, 'a') as f:
        f.write(str(args) + '\n')

    if args.data_name == 'ogbl-citation2':
        data.adj_t = data.adj_t.to_symmetric()

    selected_node_ids = None
    if args.data_name == 'ogbl-collab':
        if args.year > 0 and hasattr(data, 'edge_year'):
            selected_year_index = torch.reshape(
                (split_edge['train']['year'] >= args.year).nonzero(
                    as_tuple=False), (-1,))
            split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
            split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
            split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
            train_edge_index = split_edge['train']['edge'].t()
            # create adjacency matrix
            new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight)

        # Use training + validation edges for inference on test set.
        if args.use_valedges_as_input:
            full_edge_index = torch.cat([split_edge['valid']['edge'].t(), split_edge['train']['edge'].t()], dim=-1)
            full_edge_weight = torch.cat([split_edge['train']['weight'], split_edge['valid']['weight']], dim=-1)
            # create adjacency matrix
            new_edges = to_undirected(full_edge_index, full_edge_weight, reduce='add')
            new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
            data.adj_t = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], value=new_edge_weight)

            if args.use_coalesce:
                full_edge_index, full_edge_weight = coalesce(full_edge_index, full_edge_weight, num_nodes, num_nodes)

            split_edge['train']['edge'] = full_edge_index.t()
            split_edge['train']['weight'] = full_edge_weight

        if args.only_neg_train_nodes:
            row, col, _ = data.adj_t.coo()
            selected_node_set = set(row.tolist()).union(set(col.tolist()))
            selected_node_ids = torch.tensor(list(selected_node_set))

    if hasattr(data, 'x'):
        if data.x is not None:
            data.x = data.x.to(torch.float)
        if args.pre_aggregate:
            data = avg_pool_neighbor_x(data)

    data = data.to(device)

    if args.num_hops > 1:
        data.adj_t = data.adj_t.matmul(data.adj_t)

    if args.encoder == 'GCN':
        # Pre-compute GCN normalization.
        data.adj_t = gcn_normalization(data.adj_t)

    model_name = 'NCModel' if args.model.lower() == 'ncmodel' else 'BaseModel'
    model = eval(model_name)(
        lr=args.lr,
        dropout=args.dropout,
        gnn_num_layers=args.gnn_num_layers,
        mlp_num_layers=args.mlp_num_layers,
        emb_hidden_channels=args.emb_hidden_channels,
        gnn_hidden_channels=args.gnn_hidden_channels,
        mlp_hidden_channels=args.mlp_hidden_channels,
        num_nodes=num_nodes,
        num_node_feats=num_node_feats,
        gnn_encoder_name=args.encoder,
        predictor_name=args.predictor,
        loss_func=args.loss_func,
        optimizer_name=args.optimizer,
        device=device,
        use_node_feats=args.use_node_feats,
        train_node_emb=args.train_node_emb,
        pretrain_emb=args.pretrain_emb,
        node_feat_trans=args.node_feat_trans
        )

    evaluator = Evaluator(name=args.data_name)

    if args.eval_metric == 'hits':
        loggers = {
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }
    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.runs, args),
        }

    for run in range(args.runs):
        model.param_init()
        start_time = time.time()
        for epoch in range(1, 1 + args.epochs):
            loss = model.train(data, split_edge,
                               batch_size=args.batch_size,
                               neg_sampler_name=args.neg_sampler,
                               num_neg=args.num_neg,
                               node_ids=selected_node_ids)
            if epoch % args.eval_steps == 0:
                results = model.test(data, split_edge,
                                     batch_size=args.batch_size,
                                     evaluator=evaluator,
                                     eval_metric=args.eval_metric)
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                if epoch % args.log_steps == 0:
                    spent_time = time.time() - start_time
                    for key, result in results.items():
                        valid_res, test_res = result
                        to_print = (f'Run: {run + 1:02d}, '
                                    f'Epoch: {epoch:02d}, '
                                    f'Loss: {loss:.4f}, '
                                    f'Valid: {100 * valid_res:.2f}%, '
                                    f'Test: {100 * test_res:.2f}%')
                        print(key)
                        print(to_print)
                        with open(log_file, 'a') as f:
                            print(key, file=f)
                            print(to_print, file=f)
                    print('---')
                    print(
                        f'Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s')
                    print('---')
                    start_time = time.time()

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)


if __name__ == "__main__":
    main()

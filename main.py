# -*- coding: utf-8 -*-
import argparse
import time
import torch
import torch_geometric.transforms as T
from torch_sparse import coalesce, SparseTensor
from ogbk.logger import Logger
from ogbk.model import Model
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator


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
    parser.add_argument('--gnn_num_layers', type=int, default=2)
    parser.add_argument('--mlp_num_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_neg', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--year', type=int, default=-1)
    parser.add_argument('--num_nodes', type=int)
    parser.add_argument('--num_node_features', type=int)
    parser.add_argument('--use_node_features', type=str2bool, default=False)
    parser.add_argument('--use_coalesce', type=str2bool, default=False)
    parser.add_argument('--train_node_emb', type=str2bool, default=False)
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
    args.device = device

    dataset = PygLinkPropPredDataset(name=args.data_name, root=args.data_path)
    data = dataset[0]

    if hasattr(data, 'edge_weight'):
        if data.edge_weight is not None:
            data.edge_weight = data.edge_weight.view(-1).to(torch.float)

    data = T.ToSparseTensor()(data)
    row, col, _ = data.adj_t.coo()
    data.edge_index = torch.stack([col, row], dim=0)

    if hasattr(data, 'x'):
        if data.x is not None:
            data.x = data.x.to(torch.float)
    if hasattr(data, 'num_features'):
        args.num_node_features = data.num_features
    if hasattr(data, 'num_nodes'):
        args.num_nodes = data.num_nodes
    else:
        args.num_nodes = data.adj_t.size(0)

    split_edge = dataset.get_edge_split()
    print(args)

    if args.data_name == 'ogbl-citation2':
        data.adj_t = data.adj_t.to_symmetric()

    if args.data_name == 'ogbl-collab':
        if args.year > 0 and hasattr(data, 'edge_year'):
            selected_year_index = torch.reshape(
                (split_edge['train']['year'] >= args.year).nonzero(
                    as_tuple=False), (-1,))
            split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
            split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
            split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
            train_edge_index = split_edge['train']['edge'].t()
            data.adj_t = SparseTensor.from_edge_index(
                train_edge_index).t().to_symmetric()

        # Use training + validation edges for inference on test set.
        if args.use_valedges_as_input:
            val_edge_index = split_edge['valid']['edge'].t()
            train_edge_index = split_edge['train']['edge'].t()
            full_edge_index = torch.cat([train_edge_index, val_edge_index], dim=-1)
            data.adj_t = SparseTensor.from_edge_index(full_edge_index).t()
            data.adj_t = data.adj_t.to_symmetric()
            row, col, _ = data.adj_t.coo()
            data.edge_index = torch.stack([col, row], dim=0)

            if args.use_coalesce:
                full_edge_index, _ = coalesce(full_edge_index, torch.ones(
                    [full_edge_index.size(1), 1], dtype=int), args.num_nodes, args.num_nodes)

            split_edge['train']['edge'] = full_edge_index.t()

    data = data.to(device)

    if args.encoder == 'GCN':
        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t


    model = Model(
        lr=args.lr,
        dropout=args.dropout,
        gnn_num_layers=args.gnn_num_layers,
        mlp_num_layers=args.mlp_num_layers,
        hidden_channels=args.hidden_channels,
        batch_size=args.batch_size,
        num_nodes=args.num_nodes,
        num_node_features=args.num_node_features,
        num_neg=args.num_neg,
        gnn_encoder=args.encoder,
        predictor=args.predictor,
        loss_func=args.loss_func,
        neg_sampler=args.neg_sampler,
        optimizer=args.optimizer,
        device=args.device,
        use_node_features=args.use_node_features,
        train_node_emb=args.train_node_emb)

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
            loss = model.train(data, split_edge)
            if epoch % args.eval_steps == 0:
                results = model.test(data, split_edge, evaluator, args.eval_metric)
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                if epoch % args.log_steps == 0:
                    spent_time = time.time() - start_time
                    for key, result in results.items():
                        valid_res, test_res = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Valid: {100 * valid_res:.2f}%, '
                              f'Test: {100 * test_res:.2f}%')
                    print('---')
                    print(
                        f'Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s')
                    print('---')
                    start_time = time.time()

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()

import argparse
from torch_sparse import SparseTensor
import torch_geometric.transforms as T
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from logger import Logger
import time
from model import Model
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='GraphSage')
    parser.add_argument('--predictor', type=str, default='DOT')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--loss_func', type=str, default='AUC')
    parser.add_argument('--neg_sampler', type=str, default='global_perm')
    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    parser.add_argument('--gnn_num_layers', type=int, default=3)
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
    parser.add_argument('--num_nodes', type=int)
    parser.add_argument('--num_node_features', type=int)
    parser.add_argument('--use_node_features', action='store_true')
    parser.add_argument('--use_valedges_as_input', action='store_true')
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    args.device = device

    dataset = PygLinkPropPredDataset(name=args.data_name)
    data = dataset[0]
    edge_index = data.edge_index
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    data = T.ToSparseTensor()(data)
    split_edge = dataset.get_edge_split()
    args.num_nodes = data.num_nodes
    args.num_node_features = data.num_features

    print(args)

    # Use training + validation edges for inference on test set.
    if args.use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        train_edge_index = split_edge['train']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index).t()
        data.full_adj_t = data.full_adj_t.to_symmetric()
        split_edge['train']['edge'] = torch.cat([train_edge_index, val_edge_index], dim=-1).t()
        data.adj_t = data.full_adj_t
        row, col, _ = data.adj_t.coo()
        data.edge_index = torch.stack([col, row], dim=0)
    else:
        data.full_adj_t = data.adj_t

    data = data.to(device)
    model = Model(args)

    evaluator = Evaluator(name=args.data_name)

    loggers = {
        'Hits@20': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.param_init()
        start_time = time.time()
        for epoch in range(1, 1 + args.epochs):
            loss = model.train(data, split_edge)
            if epoch % args.eval_steps == 0:
                results = model.test(data, split_edge, evaluator)
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                if epoch % args.log_steps == 0:
                    spent_time = time.time() - start_time
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')
                    print(f'Training Time Per Epoch: {spent_time / args.eval_steps: .4f} s')
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
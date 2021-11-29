# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch_geometric.utils import negative_sampling, add_self_loops


def global_neg_sample(edge_index, num_nodes, num_samples,
                      num_neg, method='sparse'):
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(new_edge_index, num_nodes=num_nodes,
                                 num_neg_samples=num_samples * num_neg, method=method)
    neg_src = neg_edge[0]
    neg_dst = neg_edge[1]
    if neg_edge.size(1) < num_samples * num_neg:
        k = num_samples * num_neg - neg_edge.size(1)
        rand_index = torch.randperm(neg_edge.size(1))[:k]
        neg_src = torch.cat((neg_src, neg_src[rand_index]))
        neg_dst = torch.cat((neg_dst, neg_dst[rand_index]))
    return torch.reshape(torch.stack(
        (neg_src, neg_dst), dim=-1), (-1, num_neg, 2))


def global_perm_neg_sample(edge_index, num_nodes,
                           num_samples, num_neg, method='sparse'):
    new_edge_index, _ = add_self_loops(edge_index)
    neg_edge = negative_sampling(new_edge_index, num_nodes=num_nodes,
                                 num_neg_samples=num_samples, method=method)
    # neg_src = neg_edge[0]
    # neg_dst = neg_edge[1]
    # if neg_edge.size(1) < num_samples:
    #     k = num_samples - neg_edge.size(1)
    #     rand_index = torch.randperm(neg_edge.size(1))[:k]
    #     neg_src = torch.cat((neg_src, neg_src[rand_index]))
    #     neg_dst = torch.cat((neg_dst, neg_dst[rand_index]))
    # tmp_src = neg_src
    # tmp_dst = neg_dst
    # for i in range(num_neg - 1):
    #     rand_index = torch.randperm(num_samples)
    #     neg_src = torch.cat((neg_src, tmp_src[rand_index]))
    #     neg_dst = torch.cat((neg_dst, tmp_dst[rand_index]))
    # return torch.reshape(torch.stack(
    #     (neg_src, neg_dst), dim=-1), (-1, num_neg, 2))
    return sample_perm_copy(neg_edge, num_samples, num_neg)


def local_random_neg_sample(pos_edges, num_nodes, num_neg):
    neg_src = pos_edges[torch.arange(pos_edges.size(0)), torch.randint(
        0, 2, (pos_edges.size(0), ), dtype=torch.long)]
    neg_src = torch.reshape(neg_src, (-1, 1)).repeat(1, num_neg)
    neg_src = torch.reshape(neg_src, (-1,))
    neg_dst = torch.randint(
        0, num_nodes, (num_neg * pos_edges.size(0),), dtype=torch.long)
    return torch.reshape(torch.stack(
        (neg_src, neg_dst), dim=-1), (-1, num_neg, 2))


def local_neg_sample(pos_edges, num_nodes, num_neg):
    neg_src = pos_edges[:, 0]
    neg_src = torch.reshape(neg_src, (-1, 1)).repeat(1, num_neg)
    neg_src = torch.reshape(neg_src, (-1,))
    neg_dst = torch.randint(
        0, num_nodes, (num_neg * pos_edges.size(0),), dtype=torch.long)
    return torch.reshape(torch.stack(
        (neg_src, neg_dst), dim=-1), (-1, num_neg, 2))


def local_perm_neg_sample(pos_edges, num_nodes, num_neg):
    neg_src = pos_edges[:, 0]
    neg_src = torch.reshape(neg_src, (-1,))
    neg_dst = torch.randint(
        0, num_nodes, (pos_edges.size(0),), dtype=torch.long)
    tmp_src = neg_src
    tmp_dst = neg_dst
    for i in range(num_neg - 1):
        rand_index = torch.randperm(pos_edges.size(0))
        neg_src = torch.cat((neg_src, tmp_src[rand_index]))
        neg_dst = torch.cat((neg_dst, tmp_dst[rand_index]))
    return torch.reshape(torch.stack(
        (neg_src, neg_dst), dim=-1), (-1, num_neg, 2))


def local_dist_neg_sample(pos_edges, num_neg, neg_table):
    neg_src = pos_edges[:, 0]
    neg_src = torch.reshape(neg_src, (-1, 1)).repeat(1, num_neg)
    neg_src = torch.reshape(neg_src, (-1,))
    neg_dst_index = torch.randint(
        0, neg_table.size(0), (num_neg * pos_edges.size(0),), dtype=torch.long)
    neg_dst = neg_table[neg_dst_index]
    return torch.reshape(torch.stack(
        (neg_src, neg_dst), dim=-1), (-1, num_neg, 2))


def global_neg_sample_with_ids(edge_index, num_nodes, num_samples, num_neg, node_ids):
    num_samples = num_samples
    size = node_ids.size(0) * node_ids.size(0)
    row, col = edge_index
    idx = row * num_nodes + col

    # Percentage of edges to oversample so that we are save to only sample once
    # (in most cases).
    alpha = abs(1 / (1 - 1.2 * (edge_index.size(1) / size)))

    pair_ids = torch.randint(
        0, node_ids.size(0), (2, alpha * num_samples), dtype=torch.long)
    sampled_pairs = node_ids[pair_ids]

    perm = sampled_pairs[0] * num_nodes + sampled_pairs[1]
    mask = torch.from_numpy(np.isin(perm, idx.to('cpu'))).to(torch.bool)
    perm = perm[~mask][:num_samples].to(edge_index.device)

    row = perm // num_nodes
    col = perm % num_nodes
    neg_edge = torch.stack([row, col], dim=0).long()

    return sample_perm_copy(neg_edge, num_samples, num_neg)


def sample_perm_copy(edge_index, target_num_sample, num_perm_copy):
    src = edge_index[0]
    dst = edge_index[1]
    if edge_index.size(1) < target_num_sample:
        k = target_num_sample - edge_index.size(1)
        rand_index = torch.randperm(edge_index.size(1))[:k]
        src = torch.cat((src, src[rand_index]))
        dst = torch.cat((dst, dst[rand_index]))
    tmp_src = src
    tmp_dst = dst
    for i in range(num_perm_copy - 1):
        rand_index = torch.randperm(target_num_sample)
        src = torch.cat((src, tmp_src[rand_index]))
        dst = torch.cat((dst, tmp_dst[rand_index]))
    return torch.reshape(torch.stack(
        (src, dst), dim=-1), (-1, num_perm_copy, 2))


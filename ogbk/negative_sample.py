# -*- coding: utf-8 -*-
import torch
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
    neg_src = neg_edge[0]
    neg_dst = neg_edge[1]
    if neg_edge.size(1) < num_samples:
        k = num_samples - neg_edge.size(1)
        rand_index = torch.randperm(neg_edge.size(1))[:k]
        neg_src = torch.cat((neg_src, neg_src[rand_index]))
        neg_dst = torch.cat((neg_dst, neg_dst[rand_index]))
    tmp_src = neg_src
    tmp_dst = neg_dst
    for i in range(num_neg - 1):
        rand_index = torch.randperm(num_samples)
        neg_src = torch.cat((neg_src, tmp_src[rand_index]))
        neg_dst = torch.cat((neg_dst, tmp_dst[rand_index]))
    return torch.reshape(torch.stack(
        (neg_src, neg_dst), dim=-1), (-1, num_neg, 2))


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
    neg_src = torch.reshape(neg_src, (-1, 1)).repeat(1, num_neg)
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

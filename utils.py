# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import scipy.sparse as sp
import random
random.seed(1001)
np.random.seed(121)


def adj_nhood(adj, nhood):
    adj = adj.tocsr()
    mt = sp.eye(adj.shape[0], format='csr')
    for _ in range(nhood):
        mt = mt * adj
    indices = mt.nonzero()
    mt[indices[0], indices[1]] = 1.0
    return mt


def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()
    new_shape = (adj.shape[0]+1, adj.shape[1]+1)
    return indices, adj.data, new_shape


def load_graph(f, directed=False):
    if directed:
        return nx.read_edgelist(f, data=False, create_using=nx.DiGraph())
    else:
        return nx.read_edgelist(f, data=False)


def prepare_batch(adj, edges, num_nodes, nhood, batch_size, is_train=False):
    if is_train:
        edges_reshape = np.reshape(edges, (2, -1, 2))
        edges = edges_reshape[0]
        edges_neg = edges_reshape[1]
    nb_lists = []
    user_batch = []
    mask_batch = []
    nb_lists_neg = []

    count = 0
    adj = adj_nhood(adj, nhood)
    adj = adj.tolil()

    for k in range(edges.shape[0]):
        count += 1
        nb_list = list(set(adj.rows[int(edges[k][0])]).union(set(adj.rows[int(edges[k][1])])) )
        nb_list = [int(edges[k][0]), int(edges[k][1])] + nb_list
        nb_lists.append(nb_list)
        if is_train:
            nb_list_neg = list(set(adj.rows[int(edges_neg[k][0])]).union(set(adj.rows[int(edges_neg[k][1])])) )
            nb_list_neg = [int(edges_neg[k][0]), int(edges_neg[k][1])] + nb_list_neg
            nb_lists_neg.append(nb_list_neg)
        if count == batch_size: # create batch
            if is_train:
                nb_lists = nb_lists + nb_lists_neg
            users, masks = boolean_indexing(nb_lists, num_nodes)
            user_batch.append(users)
            mask_batch.append(-1e9 *(1-masks))
            nb_lists = []
            nb_lists_neg = []
            count = 0
    if nb_lists:
        if is_train:
            nb_lists = nb_lists + nb_lists_neg
        users, masks = boolean_indexing(nb_lists, num_nodes)
        user_batch.append(users)
        mask_batch.append(-1e9 *(1-masks))
    return user_batch, mask_batch


def boolean_indexing(v, num_nodes):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.zeros(mask.shape,dtype=int) + num_nodes
    out[mask] = np.concatenate(v)
    return out, mask*1


def users_masking(adj, users, num_nodes):
    nb_list = adj.rows[users]
    neighbors, n_mask = boolean_indexing(nb_list, num_nodes)
    centers = np.expand_dims(users, -1)
    out = np.concatenate((centers, neighbors), axis=-1)
    c_mask = np.ones(centers.shape)
    mask = np.concatenate((c_mask, n_mask), axis=-1)
    return out, mask, centers


def read_edges_from_file(f, graph_index, directed=False):
    edge_value = []
    edge_source = []
    edge_target = []
    with open(f, 'r') as fr:
        for line in fr:
            edge = line.strip().split()
            if directed:
                edge_source.append(graph_index[edge[0]])
                edge_target.append(graph_index[edge[1]])
                edge_value.append(1)
            else:
                edge_source.append(graph_index[edge[0]])
                edge_target.append(graph_index[edge[1]])
                edge_value.append(1)
                edge_source.append(graph_index[edge[1]])
                edge_target.append(graph_index[edge[0]])
                edge_value.append(1)
    return edge_value, edge_source, edge_target


def process_train_graph(whole_adj, graph_index, val_pos_file, test_pos_file, directed=False):
    val_pos_edge_value, val_pos_source, val_pos_target = read_edges_from_file(val_pos_file, graph_index, directed)
    test_pos_edge_value, test_pos_source, test_pos_target = read_edges_from_file(test_pos_file, graph_index, directed)

    train_adj = whole_adj - \
                sp.coo_matrix((val_pos_edge_value, (val_pos_source, val_pos_target)), shape=whole_adj.shape) - \
                sp.coo_matrix((test_pos_edge_value, (test_pos_source, test_pos_target)), shape=whole_adj.shape)
    train_edges = sp.triu(train_adj, 1).nonzero()
    train_neg_adj = sp.coo_matrix(-(whole_adj.todense() - 1))
    train_non_edges = train_neg_adj
    nns = train_non_edges.tolil().rows

    return train_adj, train_edges, nns,


def get_val_test(val_pos_file, val_neg_file, test_pos_file, test_neg_file, graph_index, directed=False):
    val_pos_edge_value, val_pos_source, val_pos_target = read_edges_from_file(val_pos_file, graph_index, directed)
    test_pos_edge_value, test_pos_source, test_pos_target = read_edges_from_file(test_pos_file, graph_index, directed)
    val_neg_edge_value, val_neg_source, val_neg_target = read_edges_from_file(val_neg_file, graph_index, directed)
    test_neg_edge_value, test_neg_source, test_neg_target = read_edges_from_file(test_neg_file, graph_index, directed)

    val_sources = np.concatenate((np.array(val_pos_source[0::2]), np.array(val_neg_source[0::2])))
    val_targets = np.concatenate((np.array(val_pos_target[0::2]), np.array(val_neg_target[0::2])))
    val_labels = np.concatenate((np.ones(len(val_pos_source[0::2])), np.zeros(len(val_neg_source[0::2]))))

    test_sources = np.concatenate((np.array(test_pos_source[0::2]), np.array(test_neg_source[0::2])))
    test_targets = np.concatenate((np.array(test_pos_target[0::2]), np.array(test_neg_target[0::2])))
    test_labels = np.concatenate((np.ones(len(test_pos_source[0::2])), np.zeros(len(test_neg_source[0::2]))))

    return (val_sources, val_targets, val_labels), (test_sources, test_targets, test_labels)


def negative_sample_from_nns(nns, num_nodes, size):
    train_sources_neg = []
    train_targets_neg = []
    for i in range(size):
        neg_source = random.randint(0, num_nodes-1)
        train_sources_neg.append(neg_source)
        train_targets_neg.append(random.choice(nns[neg_source]))
    return np.array(train_sources_neg), np.array(train_targets_neg)


def creat_index(G):
    graph_index = {}
    i = 0
    for key in G.nodes():
        graph_index[key] = i
        i += 1
    return graph_index






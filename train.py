# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import networkx as nx
from sklearn.metrics import classification_report, roc_auc_score
from halpnet import HalpNet
from utils import *
import random
import argparse

random.seed(1001)


class Config(object):
    """Model setting and data setting"""
    def __init__(self, args, split_num):
        self.lr = 0.001
        self.emb_dim = 64
        self.batch_size = 512
        self.data_name = 'ceg'
        self.hid_units = 64
        self.num_nodes = 0
        self.l2_coef = 0.0005
        self.num_epochs = 500
        self.patience = 50
        self.num_hops = 1

        if args.data:
            self.data_name = args.data
        if args.batch_size:
            self.batch_size = args.batch_size
        if args.hidden_size:
            self.emb_dim = args.hidden_size
            self.hid_units = args.hidden_size
        if args.lr:
            self.lr = args.lr
        if args.num_hops:
            self.num_hops = args.nh

        self.data_path = 'data/'
        self.dataset = self.data_name + '/' + self.data_name.upper()

        self.whole_graph_file = self.data_path + self.dataset + '.net'
        # training file
        self.train_graph_file = self.data_path + self.dataset + '_train_' + str(split_num) + '.net'
        self.train_neg_file = self.data_path + self.dataset + '_train_neg_' + str(split_num) + '.net'
        # validation file
        self.val_pos_file = self.data_path + self.dataset + '_val_pos_' + str(split_num) + '.net'
        self.val_neg_file = self.data_path + self.dataset + '_val_neg_' + str(split_num) + '.net'
        # test file
        self.test_pos_file = self.data_path + self.dataset + '_test_pos_' + str(split_num) + '.net'
        self.test_neg_file = self.data_path + self.dataset + '_test_neg_' + str(split_num) + '.net'


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lr", type=float, help="learning rate")
    parser.add_argument("-x", "--hidden_size", type=int, help="hidden dimension")
    parser.add_argument("-d", "--data", help="data name")
    parser.add_argument("-g", "--gpu", help="gpu id")
    parser.add_argument("-b", "--batch_size", type=int, help="batch size")
    parser.add_argument("-n", "--num_hops", type=int, help="number of hops")
    return parser.parse_args()


if __name__ == '__main__':
    best_performances = []
    args = args_parser()

    for test_iter in range(5):
        comp_g = tf.Graph()
        with comp_g.as_default():
            config = Config(args, test_iter)
            # Read whole graph
            whole_graph = load_graph(config.whole_graph_file)
            # Create index
            graph_index = creat_index(whole_graph)
            config.num_nodes = nx.number_of_nodes(whole_graph)
            # Whole adj matrix
            whole_adj = nx.adjacency_matrix(whole_graph)
            # get train graph information
            train_adj, train_edges_pos, nns = process_train_graph(whole_adj, graph_index,
                                                                  config.val_pos_file, config.test_pos_file, False)
            # get train edges
            train_sources_pos, train_targets_pos = train_edges_pos[0], train_edges_pos[1]
            tr_size = len(train_sources_pos)
            train_edges_pos = np.concatenate((np.expand_dims(train_sources_pos, -1), np.expand_dims(train_targets_pos, -1) ), -1)

            val_set, test_set = get_val_test(config.val_pos_file, config.val_neg_file, config.test_pos_file, config.test_neg_file, graph_index, False)
            val_edges = np.concatenate((np.expand_dims(val_set[0], -1), np.expand_dims(val_set[1], -1)), -1)
            test_edges = np.concatenate((np.expand_dims(test_set[0], -1), np.expand_dims(test_set[1], -1)), -1)

            batch_size = config.batch_size
            if batch_size == -1:
                batch_size = train_edges.shape[0]

            # create val and test batch data
            val_user_batch, val_mask_batch = prepare_batch(train_adj, val_edges,
                                                           config.num_nodes, config.num_hops, batch_size)
            test_user_batch, test_mask_batch = prepare_batch(train_adj, test_edges,
                                                             config.num_nodes, config.num_hops, batch_size)

            sparse_adj = adj_nhood(train_adj, config.num_hops)
            sparse_adj = preprocess_adj_bias(sparse_adj)
            model = HalpNet(config, sparse_adj)

            saver = tf.train.Saver(max_to_keep=1)
            vauc_mx = 0.0
            tfconfig = tf.ConfigProto()
            tfconfig.gpu_options.allow_growth = True

            with tf.Session(config=tfconfig) as sess:
                tf.set_random_seed(1402)
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                train_loss = 0
                train_auc = 0
                val_auc = 0
                test_auc = 0

                for epoch in range(config.num_epochs):
                    np.random.shuffle(train_edges_pos)
                    train_sources_neg, train_targets_neg = negative_sample_from_nns(nns, config.num_nodes, tr_size)
                    train_edges_neg = np.concatenate((np.expand_dims(train_sources_neg, -1),
                                                      np.expand_dims(train_targets_neg, -1)), -1)
                    train_edges = np.concatenate((train_edges_pos, train_edges_neg), 0)
                    user_batch, mask_batch = prepare_batch(train_adj, train_edges,
                                                           config.num_nodes, config.num_hops, batch_size, True)
                    train_time = 0
                    tr_step = 0

                    len(user_batch)
                    for i in range(len(user_batch)):
                        start_time = time.time()
                        users = user_batch[i]
                        masks = mask_batch[i]

                        _, loss_value_tr = sess.run([model.train_op, model.loss],
                                                    feed_dict={
                                                        model.users: users,
                                                        model.mask: masks,
                                                        model.is_train: True,
                                                        model.attn_drop: 0.2,
                                                        model.ffd_drop: 0.2})
                        train_loss += loss_value_tr
                        tr_step += 1
                        train_time += time.time() - start_time

                    print('Epoch %d Training Time is %.3f' % (epoch, train_time))

                    # Validation AUC
                    pred_vl = []
                    for i in range(len(val_user_batch)):
                        users = val_user_batch[i]
                        masks = val_mask_batch[i]
                        pred = sess.run(model.lp,
                                        feed_dict={
                                            model.users: users,
                                            model.mask: masks,
                                            model.is_train: False,
                                            model.attn_drop: 0.0,
                                            model.ffd_drop: 0.0})

                        for i in range(len(pred)):
                            pred_vl.append(pred[i])

                    val_auc = roc_auc_score(val_set[-1], pred_vl)

                    # Testing AUC
                    pred_te = []
                    for i in range(len(test_user_batch)):
                        users = test_user_batch[i]
                        masks = test_mask_batch[i]
                        pred = sess.run(model.lp,
                                        feed_dict={
                                            model.users: users,
                                            model.mask: masks,
                                            model.is_train: False,
                                            model.attn_drop: 0.0,
                                            model.ffd_drop: 0.0})
                        for i in range(len(pred)):
                            pred_te.append(pred[i])

                    test_auc = roc_auc_score(test_set[-1], pred_te)

                    print('%s-Epoch:%d | Train:loss=%.5f | Val:auc=%.5f | Test: auc=%.5f' %
                            (config.dataset+str(test_iter), epoch, train_loss/tr_step, val_auc, test_auc))

                    if val_auc >= vauc_mx:
                        tauc_best_model = test_auc
                        vauc_mx = np.max((val_auc, vauc_mx))
                        curr_step = 0
                    else:
                        curr_step += 1

                    if curr_step == config.patience:
                        print('Early stop! Max Validation AUC: ', vauc_mx)
                        print('Early stop model Test AUC: ', tauc_best_model)
                        break

                    train_loss = 0
                    train_auc = 0
                    val_auc = 0
                    test_auc = 0

        best_performances.append(tauc_best_model)

    for i in range(len(best_performances)):
        print('AUC on ' + config.dataset + str(i) + ':', best_performances[i])

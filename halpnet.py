# -*- coding: utf-8 -*-

import tensorflow as tf
from attention import *


class HalpNet(object):
    def __init__(self, config, adj_mat):
        self.emb_dim = config.emb_dim
        self.hid_units = config.hid_units
        self.num_nodes = config.num_nodes
        self.activation = tf.nn.elu
        self.l2_coef = config.l2_coef
        self.lr = config.lr
        self.global_adj = tf.SparseTensor(adj_mat[0], adj_mat[1], adj_mat[2])

        with tf.device("/cpu:0"):
            self.embedding = tf.get_variable(
                "embedding", [self.num_nodes + 1,
                              self.emb_dim], dtype=tf.float32)  # self.num_nodes + 1 for mask node

        self.users = tf.placeholder(tf.int32, [None, None])  # [2b, n]
        self.mask = tf.placeholder(tf.float32, [None, None])  # [2b, n]

        self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        self.is_train = tf.placeholder(dtype=tf.bool, shape=())

        self.emb = tf.nn.embedding_lookup(struc_preserve_attention(self.embedding, self.global_adj, self.hid_units,
                                                                   self.activation, self.attn_drop, self.ffd_drop),
                                          self.users)

        self.h2 = struc_aggregate_attention(self.emb, self.hid_units, self.mask, self.activation,
                                            coef_drop=self.attn_drop, in_drop = self.ffd_drop)  # (2b, d)
        self.lp = tf.layers.dense(self.h2, 1, tf.identity, use_bias=False) # (2b)

        lp_reshape = tf.reshape(self.lp, [2, -1])
        self.lp_pos = lp_reshape[0]
        self.lp_neg = lp_reshape[1]

        self.rank_loss = tf.reduce_sum(tf.square(1-(self.lp_pos - self.lp_neg)))

        # weight decay
        vars = tf.trainable_variables()
        self.L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                                 in ['bias', 'gamma', 'b', 'g', 'beta']]) * self.l2_coef

        # optimizer
        self.loss = self.rank_loss + self.L2_loss

        # optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)

        # training op
        self.train_op = self.opt.minimize(self.loss)
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

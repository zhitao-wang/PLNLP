# -*- coding: utf-8 -*-

import tensorflow as tf


def struc_preserve_attention(emb, adj_mat, out_sz, activation,
                             coef_drop=0.0, in_drop=0.0):
    # emb:(n, D), adj_mat:(n, n)
    scope = "struc_preserve_attention"
    with tf.variable_scope(scope):
        if in_drop != 0.0:
            emb = tf.nn.dropout(emb, 1.0 - in_drop)
        fts = tf.layers.dense(emb, out_sz, activation)  # (n, d)

        f_c = tf.layers.dense(emb, out_sz, activation)  # (n, d)
        f_n = tf.layers.dense(emb, out_sz, activation)  # (n, d)

        logits = tf.matmul(f_c, f_n, transpose_b=True)
        logits = adj_mat * logits  # (n, n) sparse
        coefs = tf.sparse.softmax(logits)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            fts = tf.nn.dropout(fts, 1.0 - in_drop)

        vals = tf.sparse.matmul(coefs, fts)  # (n, n) * (n, d) = (n, d)
        vals = tf.contrib.layers.bias_add(vals)

        return activation(vals)  # activation


def struc_aggregate_attention(emb, out_sz, mask, activation,
                              coef_drop=0.0, in_drop=0.0):
    scope = "struc_aggregate_attention"
    with tf.variable_scope(scope):
        if in_drop != 0.0:
            emb = tf.nn.dropout(emb, 1.0 - in_drop)

        fts = tf.layers.dense(emb, out_sz, tf.identity, name='feat_W', use_bias=False)   # fts size: (b, n, d)
        f1 = tf.expand_dims(tf.layers.dense(emb[:, 0, :] * emb[:, 1, :], out_sz, tf.identity,
                                            name='attn_W1', use_bias=False), 1)  # f1 size: (b, 1, d)
        f2 = tf.layers.dense(emb[:, 2:, :], out_sz, tf.identity, name='attn_W2', use_bias=False)  # f2 size: (b, n, d)

        logits = tf.reduce_sum(f1 * f2 , -1)  # logits size: (b, n)
        coefs = tf.nn.softmax(logits + mask[:, 2:])   # coefs size: (b, n)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)

        if in_drop != 0.0:
            fts = tf.nn.dropout(fts, 1.0 - in_drop)

        h = tf.reduce_sum(tf.expand_dims(coefs, -1) * fts[:, 2:, :], -2)   # (b, n, 1) * (b, n, d) = (b, d)
        h = tf.concat([fts[:, 0, :] * fts[:, 1, :], h], -1)

        return activation(h)   # activation

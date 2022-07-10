
import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, num_nodes, num_feat, feature_lens, n_sampled=100, embedding_dim=128, lr=0.001):
        self.n_sampled = n_sampled
        self.num_feat = num_feat
        self.feature_lens = feature_lens
        self.embedding_dim = embedding_dim
        self.num_nodes = num_nodes
        self.lr = lr
        self.softmax_w = tf.Variable(tf.random.truncated_normal((num_nodes, embedding_dim), stddev=0.1), name='softmax_w')
        self.softmax_b = tf.Variable(tf.zeros(num_nodes), name='softmax_b')
        self.inputs = self.input_init()
        self.embedding = self.embedding_init()
        self.alpha_embedding = tf.Variable(tf.random.uniform((num_nodes, num_feat), -1, 1))
        self.merge_emb = self.attention_merge()
        self.cost = self.make_skipgram_loss()
        self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.cost)

    def embedding_init(self):
        cat_embedding_vars = []
        for i in range(self.num_feat):
            embedding_var = tf.Variable(tf.random.uniform((self.feature_lens[i], self.embedding_dim), -1, 1), 
                                        name=f'embedding{str(i)}', trainable=True)
            # embedding_var = tf.Variable(tf.random.uniform((self.feature_lens[i], self.embedding_dim), -1, 1), 
            #                             name=f'embedding{str(i)}', trainable=True)

            cat_embedding_vars.append(embedding_var)
        return cat_embedding_vars

    def attention_merge(self):
        """
        get the weighting sum of base embedding and side information embedding
        """
        # sourcery skip: inline-immediately-returned-variable
        embed_list = []
        for i in range(self.num_feat):
            cat_embed = tf.nn.embedding_lookup(self.embedding[i], self.inputs[i])
            embed_list.append(cat_embed)
        stack_embed = tf.stack(embed_list, axis=-1)
        # attention merge
        alpha_embed = tf.nn.embedding_lookup(self.alpha_embedding, self.inputs[0])
        alpha_embed_expand = tf.expand_dims(alpha_embed, 1)
        alpha_i_sum = tf.reduce_sum(tf.exp(alpha_embed_expand), axis=-1)
        merge_emb = tf.reduce_sum(stack_embed * tf.exp(alpha_embed_expand), axis=-1) / alpha_i_sum
        return merge_emb

    def input_init(self):
        input_list = []
        for i in range(self.num_feat):
            input_col = tf.compat.v1.placeholder(tf.int32, [None], name=f'inputs_{str(i)}')
            input_list.append(input_col)
        input_list.append(tf.compat.v1.placeholder(tf.int32, shape=[None, 1], name='label'))
        return input_list

    def make_skipgram_loss(self):
        # sourcery skip: inline-immediately-returned-variable
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=self.softmax_w,
            biases=self.softmax_b,
            labels=self.inputs[-1],
            inputs=self.merge_emb,
            num_sampled=self.n_sampled,
            num_classes=self.num_nodes,
            num_true=1,
            sampled_values=tf.random.uniform_candidate_sampler(
                true_classes=tf.cast(self.inputs[-1], tf.int64), 
                num_true=1, 
                num_sampled=self.n_sampled, 
                unique=True, 
                range_max=self.num_nodes
            )
        ))
        return loss
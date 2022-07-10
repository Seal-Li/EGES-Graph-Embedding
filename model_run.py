
from operator import index
import numpy as np
import tensorflow as tf
import time
from const import parse_args
from walk import RWGraph
from model import Model
from utils import *

def get_walks(path, num_walks=10, walk_length=20, schema=None):
    # sourcery skip: inline-immediately-returned-variable
    edge = load_edge_data(path)
    G = get_G_from_edges(edge)
    node_type = load_node_type(path)

    walker = RWGraph(G, node_type, num_workers=8)
    print("walking...")
    walks = walker.simulate_walks(num_walks, walk_length, schema=schema)
    # print(walks)

    return walks


def get_all_pairs(walks, window_size, path):
    # sourcery skip: for-append-to-extend, remove-redundant-continue
    all_pairs = []
    for k in range(len(walks)):
        for i in range(len(walks[k])):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 0 or j >= len(walks[k]):
                    continue
                else:
                    all_pairs.append([walks[k][i], walks[k][j]])
    all_pairs = np.array(all_pairs, dtype=np.int32)
    np.savetxt(f'{path}/all_pairs.txt', X=all_pairs, fmt="%d", delimiter=" ")
    return None

def generate(path, num_walks, walk_length, window_size, schema=None):
    # data preprocess
    walks = get_walks(path, num_walks, walk_length, schema)
    get_all_pairs(walks, window_size, path)
    # read train_data
    print('read features...')
    start_time = time.time()
    side_info = np.loadtxt(f'{path}/side_info.txt', dtype=np.int32, delimiter=' ')
    all_pairs = np.loadtxt(f'{path}/all_pairs.txt', dtype=np.int32, delimiter=' ')
    feature_lens = []
    for i in range(side_info.shape[1]):
        tmp_len = len(set(side_info[:, i]))
        feature_lens.append(tmp_len)
    end_time = time.time()
    print('time consumed for read features: %.2f' % (end_time - start_time))
    # print(side_info, type(side_info))
    return side_info, feature_lens, all_pairs


def train_model(side_info, feature_lens, num_feat, n_sampled, embedding_dim, lr):
    # sourcery skip: convert-to-enumerate, for-index-underscore, move-assign-in-block, use-fstring-for-formatting
    myModel = Model(len(side_info), num_feat, feature_lens, n_sampled, embedding_dim, lr)
    # init model
    print('init...')
    start_time = time.time()
    init = tf.compat.v1.global_variables_initializer()
    config_tf = tf.compat.v1.ConfigProto()
    config_tf.gpu_options.allow_growth = True
    sess =  tf.compat.v1.Session(config=config_tf)
    sess.run(init)
    end_time = time.time()
    print('time consumed for initialize the model: %.2f' % (end_time - start_time))

    print_every_k_iterations = 100
    loss = 0
    iteration = 0
    start = time.time()

    max_iter = len(all_pairs)//args.batch_size*args.epochs
    for iter in range(max_iter):
        iteration += 1
        batch_features, batch_labels = next(graph_context_batch_iter(all_pairs, args.batch_size, side_info,
                                                                     args.num_feat))
        feed_dict = {input_col: batch_features[:, i] for i, input_col in enumerate(myModel.inputs[:-1])}
        feed_dict[myModel.inputs[-1]] = batch_labels
        _, train_loss = sess.run([myModel.train_op, myModel.cost], feed_dict=feed_dict)

        loss += train_loss

        if iteration % print_every_k_iterations == 0:
            end = time.time()
            e = iteration*args.batch_size//len(all_pairs)
            print("Epoch {}/{}".format(e, args.epochs),
                  "Iteration: {}".format(iteration),
                  "Avg. Training loss: {:.4f}".format(loss / print_every_k_iterations),
                  "{:.4f} sec/batch".format((end - start) / print_every_k_iterations))
            loss = 0
            start = time.time()

    print('optimization finished...')
    saver = tf.compat.v1.train.Saver()
    saver.save(sess, "checkpoints/myModel")
    return myModel, sess


if __name__ == '__main__':
    args = parse_args()
    
    side_info, feature_lens, all_pairs = generate(args.path, args.num_walks, args.walk_length, args.window_size, args.schema)
    myModel, sess = train_model(side_info, feature_lens, args.num_feat, args.n_sampled, args.embedding_dim, args.lr)

    feed_dict_test = {input_col: list(side_info[:, i]) for i, input_col in enumerate(myModel.inputs[:-1])}
    feed_dict_test[myModel.inputs[-1]] = np.zeros((len(side_info), 1), dtype=np.int32)
    embedding_result = sess.run(myModel.merge_emb, feed_dict=feed_dict_test)

    print('saving embedding result...')
    write_embedding(embedding_result, args.outputEmbedFile)
    # print(embedding_result.shape)

    # print('visualization...')
    # plot_embeddings(embedding_result[:5000, :], side_info[:5000, :])
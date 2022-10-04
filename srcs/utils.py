import dgl
import random
import argparse
import torch as th
import networkx as nx
from collections import defaultdict


def init_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--root_path', type=str, default='data folder')
    argparser.add_argument('--save_path', type=str, default='embedding folder')
    argparser.add_argument('--walk_length', type=int, default=50)
    argparser.add_argument('--num_walks', type=int, default=10)
    argparser.add_argument('--schema', type=str, default=None, 
                           help="the path of random walk, set as '1-2-1,1-2-3-2-1',multi path should be separate with comma(,)")
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--dim', type=int, default=100)
    argparser.add_argument('--epochs', type=int, default=10)
    argparser.add_argument('--window_size', type=int, default=2)
    argparser.add_argument('--num_negative', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.1)
    argparser.add_argument('--shrinkage', type=float, default=0.5,
                           help="the shrinkage factor of learning rate")
    argparser.add_argument('--shrink_step', type=int, default=2,
                           help="adjust learning rate every # step")
    argparser.add_argument('--num_features', type=int, default=50,
                           help="the number of node feature")
    argparser.add_argument('--log_every', type=int, default=100,
                           help="print log for every # times batch")
    return argparser.parse_args()


def construct_graph(edges, nodes):
    graph = defaultdict(set)
    for edge in edges:
        u, v = str(edge[0]), str(edge[1])
        graph[u].add(v)
        graph[v].add(u)
    node_encodedr, node_decoder = node_transfer(nodes)
    graph = convert_to_dgl_graph(graph)
    return graph, node_encodedr, node_decoder


def convert_to_dgl_graph(graph):
    g = nx.DiGraph()
    for head, tails in graph.items():
        for tail in tails:
            src, dst = int(head), int(tail)
            g.add_edge(src, dst)
    return dgl.from_networkx(g)


def node_transfer(node_raw_ids):
    # raw_id -> new_id and new_id -> raw_id
    node_encoder, node_decoder = {}, []
    node_id = -1
    for node_raw_id in node_raw_ids:
        node_id = encode_id(node_encoder, 
                        node_decoder, 
                        node_raw_id, 
                        node_id)
    return node_encoder, node_decoder


def encode_id(encoder, decoder, raw_id, encoded_id):
    if raw_id in encoder:
        return encoded_id
    encoded_id += 1
    encoder[raw_id] = encoded_id
    decoder.append(raw_id)
    return encoded_id


def get_valid_node_set(path):
    f_name = f'{path}/edge.txt'
    node_ids = []
    edge_data = []
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[0], words[1]
            edge_data.append((x, y))
            node_ids.extend([x, y])
    node_ids = list(set(node_ids))
    return node_ids, edge_data


def encode_side_info(path, feature_num, node_encoder):
    f_name = f'{path}/side_info.txt'
    side_info_encoder = {}
    side_info_decoder = {}
    for i in range(feature_num):
        side_info_encoder[f"feature_{str(i)}"] = {}
        side_info_decoder[f"feature_{str(i)}"] = []
    side_info = {}
    features_id = [-1 for _ in range(feature_num)]
    with open(f_name, "r") as f:
        for line in f:
            fields = line[:-1].split(" ")
            node_raw_id = str(int(eval(fields[0])))
            side_raw_id = fields[1:]
            if node_raw_id in node_encoder:
                node_id = node_encoder[node_raw_id]
                side_info[node_id] = [node_id]
                for i in range(feature_num):
                    features_id[i] = encode_id(
                        side_info_encoder[f"feature_{str(i)}"], 
                        side_info_decoder[f"feature_{str(i)}"], 
                        side_raw_id[i],
                        features_id[i])
                    side_info[node_id] += [side_info_encoder[f"feature_{str(i)}"][side_raw_id[i]]]
    return side_info_encoder, side_info_decoder, side_info


def load_node_type(path, node_encoder):
    f_name = f'{path}/node_type.txt'
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[node_encoder[items[0]]] = items[1]
    return node_type


class TestEdge:
    def __init__(self, src, dst, label):
        self.src = src
        self.dst = dst
        self.label = label


def split_train_test_graph(graph):
    test_edges = []
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    sampled_edge_ids = random.sample(range(graph.num_edges()), int(graph.num_edges() / 3))
    srcs, dsts = graph.find_edges(sampled_edge_ids)
    srcs_test, dsts_test = neg_sampler(graph, th.tensor(sampled_edge_ids))
    for src, dst, src_test, dst_test in zip(srcs, dsts, srcs_test, dsts_test):
        test_edges.extend((TestEdge(src, dst, 1), TestEdge(src_test, dst_test, 0)))
    graph.remove_edges(sampled_edge_ids)
    test_graph = test_edges
    return graph, test_graph

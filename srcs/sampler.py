import numpy as np
import torch as th
from walk import RWGraph
from collections import defaultdict


class Sampler:
    def __init__(self, graph, walk_length, num_walks, window_size, num_negative, node_type, schema):
        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negative = num_negative
        self.node_weights = self.compute_node_sample_weight()
        self.node_type = node_type
        self.schema = schema
        self.graph_dict = self.generate_graph_dict()


    def sample(self, batch, side_info):
        """
            Given a batch of target nodes, sample postive
            pairs and negative pairs from the graph
        """
        batch = np.repeat(batch, 1)
        pos_pairs = self.generate_pos_pairs(batch)
        if len(pos_pairs) == 0:
            return None, None, None
        neg_pairs = self.generate_neg_pairs(pos_pairs)

        # get side info with id
        srcs, dsts, labels = [], [], []
        for pair in pos_pairs + neg_pairs:
            src, dst, label = pair
            src_info = side_info[int(src)]
            dst_info = side_info[int(dst)]

            srcs.append(src_info)
            dsts.append(dst_info)
            labels.append(label)

        return th.tensor(srcs), th.tensor(dsts), th.tensor(labels)


    def generate_graph_dict(self):
        edges = [self.graph.edges()[0].tolist(), self.graph.edges()[1].tolist()]
        graph_dict = defaultdict(set)
        for u, v in zip(edges[0], edges[1]):
            graph_dict[u].add(v)
            graph_dict[v].add(u)
        return graph_dict


    def generate_walks(self, nodes):
        walker = RWGraph(self.graph_dict, self.node_type)
        walks = walker.simulate_walks(nodes, self.num_walks, self.walk_length, schema=self.schema)
        return walks
    
    
    def generate_pos_pairs(self, nodes):
        """
            For seq [1, 2, 3, 4] and node NO.2, 
            the window_size=1 will generate:
                (1, 2) and (2, 3)
        """
        traces = self.generate_walks(nodes)
        # skip-gram
        pairs = []
        for trace in traces:
            for i in range(len(trace)):
                center = trace[i]
                left = max(0, i - self.window_size)
                right = min(len(trace), i + self.window_size + 1)
                pairs.extend([[center, x, 1] for x in trace[left:i]])
                pairs.extend([[center, x, 1] for x in trace[i+1:right]])
        return pairs

    def compute_node_sample_weight(self):
        """
            Using node degree as sample weight
        """
        return self.graph.in_degrees().float()


    def generate_neg_pairs(self, pos_pairs):
        """
            Sample based on node freq in traces, frequently shown
            nodes will have larger chance to be sampled as 
            negative node.
        """
        # sample `self.num_negative` neg dst node 
        # for each pos node pair's src node.
        negs = th.multinomial(self.node_weights, int(len(pos_pairs) * self.num_negative), replacement=True).tolist()
        tar = np.repeat([pair[0] for pair in pos_pairs], self.num_negative)
        assert(len(tar) == len(negs))
        neg_pairs = [[x, y, 0] for x, y in zip(tar, negs)]
        return neg_pairs

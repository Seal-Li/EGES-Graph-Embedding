import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

def plot_embeddings(embebed_mat, side_info_mat):
    model = TSNE(n_components=2)
    node_pos = model.fit_transform(embebed_mat)
    brand_color_idx, shop_color_idx, cate_color_idx = {}, {}, {}
    for i in range(len(node_pos)):
        brand_color_idx.setdefault(side_info_mat[i, 1], [])
        brand_color_idx[side_info_mat[i, 1]].append(i)
        shop_color_idx.setdefault(side_info_mat[i, 2], [])
        shop_color_idx[side_info_mat[i, 2]].append(i)
        cate_color_idx.setdefault(side_info_mat[i, 3], [])
        cate_color_idx[side_info_mat[i, 3]].append(i)

    plt.figure()
    for c, idx in brand_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.title('feature1 distribution')
    plt.savefig('./feature1_dist.png')

    plt.figure()
    for c, idx in shop_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.title('feature2 distribution')
    plt.savefig('./feature2_dist.png')

    plt.figure()
    for c, idx in cate_color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)
    plt.title('feature3 distribution')
    plt.savefig('./feature3_dist.png')



def write_embedding(embedding_result, outputFileName):
    # sourcery skip: ensure-file-closed
    f = open(outputFileName, 'w')
    for i in range(len(embedding_result)):
        s = " ".join(str(f) for f in embedding_result[i].tolist())
        f.write(s + "\n")
    f.close()


def graph_context_batch_iter(all_pairs, batch_size, side_info, num_features):
    while True:
        start_idx = np.random.randint(0, len(all_pairs) - batch_size)
        batch_idx = np.array(range(start_idx, start_idx + batch_size))
        batch_idx = np.random.permutation(batch_idx)
        batch = np.zeros((batch_size, num_features), dtype=np.int32)
        labels = np.zeros((batch_size, 1), dtype=np.int32)
        batch[:] = side_info[all_pairs[batch_idx, 0]]
        labels[:, 0] = all_pairs[batch_idx, 1]
        yield batch, labels


def preprocess_nxgraph(graph):  # sourcery skip: convert-to-enumerate
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):  # sourcery skip: convert-to-enumerate
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    if num % workers == 0:
        return [num//workers]*workers
    else:
        return [num//workers]*workers + [num % workers]


def load_edge_data(path):
    f_name = f'{path}/edge.txt'
    print('We are loading data from:', f_name)
    edge_data = []
    all_nodes = []
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[0], words[1]
            edge_data.append((x, y))
            all_nodes.extend([x, y])
    all_nodes = list(set(all_nodes))
    print(f'Total training nodes: {len(all_nodes)}')
    return edge_data

def load_node_type(path):
    f_name = f'{path}/node_type.txt'
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type


def get_G_from_edges(edges):
    edge_dict = defaultdict(set)
    for edge in edges:
        u, v = str(edge[0]), str(edge[1])
        edge_dict[u].add(v)
        edge_dict[v].add(u)
    return edge_dict

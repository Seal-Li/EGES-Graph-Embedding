import random
import multiprocessing

from tqdm import tqdm

def walk(walk_length, start, schema, G, node_type):  # sourcery skip: for-append-to-extend, list-comprehension, use-named-expression
    
    # Simulate a random walk starting from start node.
    rand = random.Random()

    if schema:
        schema_items = schema.split('-')
        assert schema_items[0] == schema_items[-1]
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        candidates = []
        for node in G[cur]:
            if schema == '' or node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                candidates.append(node)
        if candidates:
            walk.append(rand.choice(candidates))
        else:
            break
    return [str(node) for node in walk]

def initializer(init_G, init_node_type):
    global G
    G = init_G
    global node_type
    node_type = init_node_type

class RWGraph():
    def __init__(self, nx_G, node_type_arr=None, num_workers=16):
        self.G = nx_G
        self.node_type = node_type_arr
        self.num_workers = num_workers

    def node_list(self, nodes, num_walks):
        for _ in range(num_walks):
            for node in nodes:
                yield node

    def simulate_walks(self, nodes, num_walks, walk_length, schema=None):
        all_walks = []
        random.shuffle(nodes)

        if schema is None:
            for node in nodes:
                walks = walk(walk_length, node, schema='', G=self.G, node_type=self.node_type)
                all_walks.append(walks)
        else:
            schema_list = schema.split(',')
            for schema_iter in schema_list:
                for node in nodes:
                    if schema_iter.split('-')[0]==self.node_type[node]:
                        walks = walk(walk_length, node, schema=schema_iter, G=self.G, node_type=self.node_type)
                        all_walks.append(walks)
        return all_walks

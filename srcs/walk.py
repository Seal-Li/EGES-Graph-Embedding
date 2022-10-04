import random
import itertools


def walk(walk_length, start, schema, G, node_type):
    # Simulate a random walk starting from start node.
    rand = random.Random()
    if schema:
        schema_items = schema.split('-')
        walk_length = walk_length*len(schema_items)
        assert schema_items[0] == schema_items[-1]
    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        candidates = [
            node
            for node in G[cur]
            if schema == '' or node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]
        ]

        if candidates:
            walk.append(rand.choice(candidates))
        else:
            break
    return [str(node) for node in walk]


class RWGraph():
    def __init__(self, nx_G, node_type_arr=None):
        self.G = nx_G
        self.node_type = node_type_arr

    def simulate_walks(self, nodes, num_walks, walk_length, schema=None):
        nodes = [
            node 
            for node in nodes
            for _ in range(num_walks)
            ]
        random.shuffle(nodes)

        if schema is None:
            all_walks = [
                walk(walk_length, node, schema='', G=self.G, node_type=self.node_type)
                for node in nodes
            ]
        else:
            schema_list = schema.split(',')
            all_walks = [
                walk(walk_length, values[0], schema=values[1], G=self.G, node_type=self.node_type)
                for values in itertools.product(nodes, schema_list)
                if values[1].split('-')[0]==self.node_type[values[0]]
            ]
        return all_walks

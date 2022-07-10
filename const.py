import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--n_sampled", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--path", type=str, default='myModel/data')
    parser.add_argument("--num_feat", type=int, default=11,
                        help='the number of node feature')
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--outputEmbedFile", type=str, default='myModel/data/myModel.embed')
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--walk_length", type=int, default=20)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--schema", type=str, default='1-2-1,1-2-3-1',
                        help='metapath for random walk')
    return parser.parse_args()

import torch as th
import numpy as np


class EGES(th.nn.Module):
    def __init__(self, dim, num_nodes, encoder_num_features):
        super(EGES, self).__init__()
        self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        self.encoder_num_features = encoder_num_features
        self.dim = dim
        # embeddings for nodes
        self.embeds = [th.nn.Embedding(num_nodes, dim).to(self.device)] + [th.nn.Embedding(encoder_num, dim).to(self.device) for encoder_num in encoder_num_features]
        # weights for each node's side information
        self.side_info_weights = th.nn.Embedding(num_nodes, len(encoder_num_features))


    def forward(self, srcs, dsts):
        srcs = self.query_node_embed(srcs)
        dsts = self.query_node_embed(dsts)
        return srcs, dsts
    
    def query_node_embed(self, nodes):
        """
            @nodes: tensor of shape (batch_size, num_side_info)
        """
        nodes = nodes.to(self.device)
        batch_size = nodes.shape[0]
        # query side info weights, (batch_size, num_features)
        side_info_weights = th.exp(self.side_info_weights(nodes[:, 0]))
        # merge all embeddings
        side_info_weighted_embeds_sum = []
        side_info_weights_sum = []
        for i in range(len(self.encoder_num_features)):
            # weights for i-th side info, (batch_size, ) -> (batch_size, 1)
            i_th_side_info_weights = side_info_weights[:, i].view((batch_size, 1))
            # batch of i-th side info embedding * its weight, (batch_size, dim)
            side_info_weighted_embeds_sum.append(i_th_side_info_weights * (self.embeds[i])(nodes[:, i]))
            side_info_weights_sum.append(i_th_side_info_weights)
        # stack: (batch_size, num_features, dim), sum: (batch_size, dim)
        side_info_weighted_embeds_sum = th.sum(th.stack(side_info_weighted_embeds_sum, axis=1), axis=1)
        # stack: (batch_size, num_features), sum: (batch_size, )
        side_info_weights_sum = th.sum(th.stack(side_info_weights_sum, axis=1), axis=1)
        # (batch_size, dim)
        H = side_info_weighted_embeds_sum / side_info_weights_sum
        # print(f"side info wights is {side_info_weights}")
        return H

    def loss(self, srcs, dsts, labels):
        dots = th.sigmoid(th.sum(srcs * dsts, axis=1))
        dots = th.clamp(dots, min=1e-7, max=1 - 1e-7)
        v = th.mean(- (labels * th.log(dots) + (1 - labels) * th.log(1 - dots)))
        return v
        # return v

    def output_embeds(self, side_info, node_decoder):
        """
        output the embedding of nodes,
        the first column is node id.
        """
        node_id_src, side_info_ls = [], []
        for key, value in side_info.items():
            node_id_src.append(int(node_decoder[key]))
            side_info_ls.append(value)
        node_id_src = th.tensor(node_id_src)
        side_info_ls = th.tensor(side_info_ls)
        embeds = self.query_node_embed(side_info_ls)
        print(f"embeds shape is {embeds.cpu().detach().numpy().shape}")
        # embeds = th.cat(th.tensor(node_decoder).to(self.device), embeds, 1)
        embeds = np.c_[node_decoder, embeds.cpu().detach().numpy()]
        return embeds

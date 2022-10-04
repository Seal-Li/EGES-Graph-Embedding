import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
import utils
from model import EGES
from sampler import Sampler


def train(args, train_g, side_info, num_nodes, encoder_num_features, node_type):
    sampler = Sampler(
        train_g, 
        args.walk_length, 
        args.num_walks, 
        args.window_size, 
        args.num_negative,
        node_type,
        args.schema
    )

    dataloader = DataLoader(
        th.arange(train_g.num_nodes()),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: sampler.sample(x, side_info)
    )
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model = EGES(args.dim, num_nodes, encoder_num_features)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, args.shrinkage)
    
    for epoch in range(args.epochs):
        epoch_total_loss = 0
        for step, (srcs, dsts, labels) in enumerate(dataloader):
            if (srcs is None) or (dsts is None) or (labels is None):
                continue
            # the batch size of output pairs is unfixed
            srcs, dsts, labels = srcs.to(device), dsts.to(device), labels.to(device)
            srcs_embeds, dsts_embeds = model(srcs, dsts)
            loss = model.loss(srcs_embeds, dsts_embeds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_total_loss += loss.item()
            if step % args.log_every == 0:
                print('Epoch {:05d} | Step {:05d} | Step Loss {:.4f} | Epoch Avg Loss: {:.4f}'\
                    .format(epoch, step, loss.item(), epoch_total_loss / (step + 1)))
        eval(model, test_g, side_info, encoder_num_features)
        if (epoch + 1) % args.shrink_step == 0:
            scheduler.step()
    return model


def eval(model, test_graph, side_info, encoder_num_features):
    preds, labels = [], []
    srcs = th.tensor([side_info[edge.src.item()] for edge in test_graph])\
        .view(len(test_graph), len(encoder_num_features)+1)
    dsts = th.tensor([side_info[edge.dst.item()] for edge in test_graph])\
        .view(len(test_graph), len(encoder_num_features)+1)
    labels = [edge.label for edge in test_graph]
    srcs = model.query_node_embed(srcs)
    dsts = model.query_node_embed(dsts)
    preds = th.sigmoid(th.sum(srcs * dsts, dim=1))\
        .cpu().detach().numpy().tolist()
    fpr, tpr, _ = metrics.roc_curve(labels, preds, pos_label=1)
    print("Evaluate link prediction AUC: {:.4f}".format(metrics.auc(fpr, tpr)))
    return None


if __name__ == "__main__":
    print('Now start to run!')
    args = utils.init_args()
    node_raw_ids, edge_data = utils.get_valid_node_set(args.root_path)
    g, node_encoder, node_decoder = utils.construct_graph(edge_data, node_raw_ids)
    node_type_encoder = utils.load_node_type(args.root_path, node_encoder)
    side_info_encoder, side_info_decoder, side_info = utils.encode_side_info(args.root_path, args.num_features, node_encoder) 
    num_nodes = len(node_encoder)
    encoder_num_features = [
        len(side_info_encoder[f"feature_{str(i)}"]) 
        for i in range(args.num_features)
        ]
    print(f"num_nodes: {num_nodes}, num encoder side info: {encoder_num_features}")
    print('data encoder finished!')
    train_g, test_g = utils.split_train_test_graph(g)
    print("dataset split finished!")
    print('start training model!')
    model = train(args, train_g, side_info, num_nodes, encoder_num_features, node_type_encoder)
    embeds = model.embeds
    model.save_emb(args.save_path)
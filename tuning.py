import dgl
import torch
import random
import os
import numpy as np
import networkx as nx
from networkx.algorithms.approximation.maxcut import one_exchange
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from collections import defaultdict
from dgl.nn.pytorch import GraphConv
from itertools import chain
from time import time
from utils import qubo_dict_to_torch, loss_func, generate_graph
import wandb

# DGL backend
os.environ["DGLBACKEND"] = "pytorch"

# fix seed to ensure consistent results
seed_value = 1
random.seed(seed_value)  # seed python RNG
np.random.seed(seed_value)  # seed global NumPy RNG
torch.manual_seed(seed_value)  # seed torch RNG

# Set GPU/CPU
TORCH_DEVICE = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float32
print(f"Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}")


def run_maxcut_solver(nx_graph, seed=seed_value):
    """
    Helper function to run traditional greedy solver for MaxCUT.

    Input:
        nx_graph: networkx Graph object
    Output:
        ind_set_bitstring_nx: bitstring solution as list
        ind_set_nx_size: size of independent set (int)
        number_violations: number of violations of ind.set condition
    """
    random_state = np.random.RandomState(seed)
    # traditional solver
    t_start = time()
    cut_value, partition = one_exchange(nx_graph, seed=random_state)
    t_solve = time() - t_start
    set_size_1 = len(partition[0])
    set_size_2 = len(partition[1])

    nx_bitstring = [
        1 if (node in partition[0]) else 0 for node in sorted(list(nx_graph.nodes))
    ]

    return nx_bitstring, set_size_1, set_size_2, t_solve, partition, cut_value


def gen_q_dict_maxcut(nx_G):
    """Helper function to generate the QUBO matrix for MaxCUT bases on the adjacency matrix of the graph, following equation 7 of the paper

    Args:
        nx_G (networkx.Graph): A graph created with the networkx library

    Returns:
        dict: a dictionary where each item is a tuple representing an edge in the graph with a value given by the MaxCUT problem
    """
    Adj = nx.to_numpy_matrix(nx_G)
    # print(A)
    S = Adj.sum(axis=0)
    # print(S)
    Q_dic = defaultdict(int)
    for (u, v) in nx_G.edges:
        Q_dic[(u, v)] = 2
    for u in nx_G.nodes:
        Q_dic[(u, u)] = -S[0, u]

    return Q_dic


class Graph_conv_net(nn.Module):
    def __init__(
        self,
        input_features,
        hidden_size,
        number_classes,
        num_hid_layers,
        dropout,
        device1,
    ):
        super().__init__()
        self.gcn_layers = []
        # n-layer GCN
        self.gcn_layers.append(GraphConv(input_features, hidden_size).to(device1))
        for _ in range(num_hid_layers - 1):
            self.gcn_layers.append(GraphConv(hidden_size, hidden_size).to(device1))
        self.gcn_layers.append(GraphConv(hidden_size, number_classes).to(device1))
        self.dropout_frac = dropout
        self.num_hid_layers = num_hid_layers

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gcn_layers):
            h = layer(g, h)
            if i == (self.num_hid_layers):  # last layer
                h = torch.sigmoid(h)
            else:
                h = torch.relu(h)
                h = F.dropout(h, p=self.dropout_frac)
        return h


def get_gnn(n_nodes, gnn_hypers, torch_device, torch_dtype):
    dim_embedding = int(gnn_hypers["dim_embedding"])
    hidden_dim = int(gnn_hypers["hidden_dim"])
    dropout = gnn_hypers["dropout"]
    number_classes = int(gnn_hypers["number_classes"])
    num_hid_layers = int(gnn_hypers["num_hid_layers"])
    optimizer_name = gnn_hypers["optimizer"]

    net = Graph_conv_net(
        dim_embedding, hidden_dim, number_classes, num_hid_layers, dropout, torch_device
    )
    net = net.type(torch_dtype).to(torch_device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(torch_dtype).to(torch_device)

    params = chain(net.parameters(), embed.parameters())
    if optimizer_name == "adam":
        optimizer = Adam(params, lr=gnn_hypers["lr"])
    if optimizer_name == "adamw":
        optimizer = AdamW(params, lr=gnn_hypers["lr"])
    return net, embed, optimizer


def run_gnn_training(
    q_torch,
    dgl_graph,
    net,
    embed,
    optimizer,
    number_epochs,
    tol,
    patience,
    prob_threshold,
):
    inputs = embed.weight
    prev_loss = 1.0
    count = 0

    losses = []
    epochs = []

    best_bitstring = (
        torch.zeros((dgl_graph.number_of_nodes(),))
        .type(q_torch.dtype)
        .to(q_torch.device)
    )
    best_loss = loss_func(best_bitstring, q_torch)

    print("best_bitstring_shape", best_bitstring.shape)
    print("best_loss_shape", best_loss.shape)

    t_gnn_start = time()

    for epoch in range(number_epochs):
        probs = net(dgl_graph, inputs)[:, 0]
        loss = loss_func(probs, q_torch)
        loss_ = loss.detach().item()

        bitstring = (probs.detach() >= prob_threshold) * 1
        if loss < best_loss:
            best_loss = loss
            best_bitstring = bitstring

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss_}")
            losses.append(loss_)
            epochs.append(epoch)

        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f"Stopping early on epoch {epoch} (patience: {patience})")
            break

        prev_loss = loss_

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    t_gnn = time() - t_gnn_start
    print(f"GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}")
    print(f"GNN final continuous loss: {loss_}")
    print(f"GNN best continuous loss: {best_loss}")

    final_bitstring = (probs.detach() >= prob_threshold) * 1

    wandb.log({"loss": best_loss})
    return net, epoch, final_bitstring, best_bitstring, losses, epochs


hyperparameter_defaults = dict(
    dim_embedding=10,
    hidden_dim=5,
    num_hid_layers=1,
    dropout=0.0,
    number_classes=1,
    prob_threshold=0.5,
    number_epochs=12000,
    tolerance=1e-4,
    patience=1000,
    optimizer="adam",
    lr=0.001,
)

wandb.init(config=hyperparameter_defaults, project="gcnn-maxcut-g14")
config = wandb.config


def main():
    # run on G14 graph
    g14 = np.loadtxt("g14.txt", dtype=int, skiprows=1, usecols=[0, 1])
    g14_edgelist = list(map(tuple, g14))
    nx_temp = nx.from_edgelist(g14_edgelist, create_using=nx.Graph)
    nx_temp = nx.relabel.convert_node_labels_to_integers(
        nx_temp
    )  # start labeling nodes from zero
    # create ordered graph
    g14_graph = nx.Graph()
    g14_graph.add_nodes_from(sorted(nx_temp.nodes()))
    g14_graph.add_edges_from(nx_temp.edges)

    # Construct Q matrix for graph
    q_mat = gen_q_dict_maxcut(g14_graph)
    q_torch = qubo_dict_to_torch(
        g14_graph, q_mat, torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE
    )

    # use dgl structure for the training
    graph_dgl = dgl.from_networkx(nx_graph=g14_graph)
    graph_dgl = graph_dgl.to(TORCH_DEVICE)

    # Constructs a random d-regular or p-probabilistic graph
    # nx_graph = generate_graph(
    #     n=100, d=3, p=None, graph_type="reg", random_seed=seed_value
    # )
    # # get DGL graph from networkx graph, load onto device
    # graph_dgl = dgl.from_networkx(nx_graph=nx_graph)

    # q_mat = gen_q_dict_maxcut(nx_graph)
    # q_torch = qubo_dict_to_torch(
    #     nx_graph, q_mat, torch_dtype=TORCH_DTYPE, torch_device=TORCH_DEVICE
    # )

    # set up gcnn network
    net, embed, optimizer = get_gnn(
        graph_dgl.number_of_nodes(), config, TORCH_DEVICE, TORCH_DTYPE
    )

    # run training

    print("Running GNN...")
    gnn_start = time()

    _, epoch, final_bitstring, best_bitstring, losses, epochs = run_gnn_training(
        q_torch,
        graph_dgl,
        net,
        embed,
        optimizer,
        config["number_epochs"],
        config["tolerance"],
        config["patience"],
        config["prob_threshold"],
    )

    gnn_time = time() - gnn_start
    wandb.log({"run_time": gnn_time})

    # compute cut value from bitstrings
    print(f"Best solution: {best_bitstring}")
    hamiltonian_maxcut = -loss_func(best_bitstring, q_torch.type(torch.LongTensor))
    print(f"Max cut: {hamiltonian_maxcut.detach().numpy()}")
    wandb.log({"max_cut": hamiltonian_maxcut})
    wandb.log({"best_bitstring": best_bitstring})

    # compute cut value from regular solver
    (
        nx_bitstring,
        set_size_1,
        set_size_2,
        t_solve,
        partition,
        value_cut,
    ) = run_maxcut_solver(g14_graph)
    print(f"Greedy cut: {value_cut}")
    wandb.run.summary["greedy_cut"] = value_cut
    wandb.run.summary["greedy_bitstring"] = nx_bitstring


if __name__ == "__main__":
    main()

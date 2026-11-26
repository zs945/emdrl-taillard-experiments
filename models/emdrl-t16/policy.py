"""CPU forward pass of the published T=16 policy for deterministic evaluation."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv


class ConflictEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = GATConv(10, 8, heads=4, concat=True, dropout=0.0)

    def forward(self, graph):
        return F.elu(self.gat(graph.x, graph.edge_index))


class HardResetIF(nn.Module):
    def forward(self, current, previous):
        charged = previous + current
        spikes = (charged >= 1.0).to(current.dtype)
        return spikes, charged * (1.0 - spikes)


class MembraneReadout(nn.Module):
    def forward(self, current, previous):
        return previous + (current - previous) / 2.0


def branch(outputs):
    return nn.Sequential(nn.Linear(32, 64), nn.Identity(), nn.Identity(),
                         HardResetIF(), nn.Linear(64, outputs), MembraneReadout())


class EMDRL(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = ConflictEncoder()
        self.actor = nn.Module()
        self.actor.actor = branch(2)
        self.actor.critic = branch(1)
        self.actor.log_std = nn.Parameter(torch.zeros(1, 2), requires_grad=False)
        self.residual = nn.Linear(32, 1)

    @staticmethod
    def integrate(layers, embedding):
        current = layers[0](embedding)
        hidden = torch.zeros_like(current)
        readout = 0.0
        for _ in range(16):
            spikes, hidden = layers[3](current, hidden)
            readout = layers[5](layers[4](spikes), readout)
        return readout

    def forward(self, state):
        embedding = self.gat(Batch.from_data_list([state["graph"]]))
        mu = self.integrate(self.actor.actor, embedding)
        values = self.integrate(self.actor.critic, embedding)
        return mu[:, 1] - mu[:, 0] + self.residual(embedding).squeeze(-1), values

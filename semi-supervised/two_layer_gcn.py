import math
import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F 
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add

class TwoLayer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.conv1 = NaiveGCNConv(input_dim, hidden_dim) # GCNConv(input_dim, hidden_dim)
        self.conv2 = NaiveGCNConv(hidden_dim, output_dim) # GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Try
class NaiveGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(NaiveGCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(Tensor(in_channels, out_channels))
        stdv = math.sqrt(1 / self.weight.size(-1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index):
        N = int(edge_index.max()) + 1
        row, col = edge_index[0], edge_index[1]
        
        mask = (row != col)
        loop_index = torch.arange(0, N, dtype=row.dtype, device=row.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)
        edge_weight = torch.ones((edge_index.size(-1), ),  dtype=x.dtype, device=edge_index.device)

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=N)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        x = torch.matmul(x, self.weight)
        out = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_weight=edge_weight
        )
        
        return out

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


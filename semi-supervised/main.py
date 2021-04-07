import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from two_layer_gcn import TwoLayer
from tqdm import tqdm

import argparse
# import numpy as np

'''
print(dataset, len(dataset))
print(dataset.num_classes)
print(dataset.num_node_features)

data = dataset[0]
print(data)
print(data.is_undirected())
print(data.train_mask.sum().item())
print(data.val_mask.sum().item())
print(data.test_mask.sum().item())
'''

class NodeClassifier:
    def __init__(self, dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = dataset[0].to(self.device)
        self.model = TwoLayer(dataset.num_node_features, 16, dataset.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        self.criterion = torch.nn.NLLLoss()
    
    def train(self, max_epoch=200):
        train_mask = self.data.train_mask
        # val_mask = self.data.val_mask
        self.model.train()
        for epoch in tqdm(range(max_epoch)):
            self.optimizer.zero_grad()
            out = self.model(self.data)
            loss = self.criterion(out[train_mask], self.data.y[train_mask])
            loss.backward()
            self.optimizer.step()
    
    def test(self):
        self.model.eval()
        out = self.model(self.data)
        _, pred = out.max(dim=1)
        test_mask = self.data.test_mask
        correct = (pred[test_mask] == self.data.y[test_mask]).sum().item()
        acc = correct / self.data.test_mask.sum()
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    args = parser.parse_args()

    dataset = Planetoid(root='./{}'.format(args.dataset), name='{}'.format(args.dataset))
    
    classifier = NodeClassifier(dataset)
    classifier.train()
    print("Accuracy: {:.4f}".format(classifier.test()))

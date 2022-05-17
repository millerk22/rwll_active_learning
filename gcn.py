import os.path as osp
import sys
import argparse
from scipy import sparse

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.data import Data
import graphlearning as gl
import numpy as np

"""We now define the graph convolutional neural network."""

class Dataset(object):
    def __init__(self, name, num_features, num_classes):
      self.name = name
      self.num_features = num_features
      self.num_classes = num_classes

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convinitial = GCNConv(dataset.num_features, num_hidden_nodes, cached=True)
        self.convfinal = GCNConv(num_hidden_nodes, dataset.num_classes, cached=True)
        self.conv = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.conv.append(GCNConv(num_hidden_nodes, num_hidden_nodes, cached=True))

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.convinitial(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        for i,l in enumerate(self.conv):
            x = F.relu(l(x, edge_index, edge_weight))
            x = F.dropout(x, training=self.training)
        x = self.convfinal(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
    #loss +=
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

"""The code below trains the network, and has parameters for the number of hidden notes and number of layers"""

num_hidden_nodes = 50
num_hidden_layers = 4

#MNIST/FashionMNIST graph
labels = gl.datasets.load('mnist', labels_only=True)
W = gl.weightmatrix.knn('mnist',10,metric='vae')
i,j,v = sparse.find(W)
edge_weight=v
edge_index = np.vstack((i,j))

#Convert to torch
x,y = gl.datasets.load('mnist',metric='vae')
x = torch.from_numpy(x).float()
y = torch.from_numpy(labels).long()
edge_index = torch.from_numpy(edge_index).long()
dataset = Dataset('MNIST',x.shape[1],len(np.unique(y.numpy())))

#GPU stuff
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Randomly choose training datapoints
num_train_per_class = 50
I = gl.trainsets.generate(labels, num_train_per_class)

#Masks
train_mask = np.zeros((W.shape[0],),dtype=bool)
train_mask[I] = True

val_size = 500
val_mask = np.zeros((W.shape[0],),dtype=bool)
ind = np.arange(W.shape[0])[~train_mask]
val_ind = np.random.choice(ind,size=val_size,replace=False)
val_mask[val_ind] = True

test_mask = np.ones((W.shape[0],),dtype=bool)
test_mask[I] = False
test_mask[val_ind] = False

train_mask = torch.from_numpy(train_mask).bool()
test_mask = torch.from_numpy(test_mask).bool()
val_mask = torch.from_numpy(val_mask).bool()

#Build dataset
data = Data(x=x,y=y,train_mask=train_mask,test_mask=test_mask,val_mask=val_mask,edge_index=edge_index,edge_weight=edge_weight)

#Declare new model
model = Net()
model, data = model.to(device), data.to(device)
param_list = []
param_list.append(dict(params=model.convinitial.parameters(), weight_decay=5e-4))
for i,l in enumerate(model.conv):
    param_list.append(dict(params=l.parameters(), weight_decay=0))
param_list.append(dict(params=model.convfinal.parameters(), weight_decay=0))
optimizer = torch.optim.Adam(param_list,lr=0.01)

best_val_acc = test_acc = 0
for epoch in range(1, 100):
    train()
    train_acc, val_acc, test_acc = test()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, val_acc, test_acc))
print('GCN: %d,%.2f'%(len(I),100*test_acc))

#Run Poisson learning
#poisson_labels = gl.graph_ssl(W,I,labels[I],algorithm='poisson')
#print('Poisson Learning: %.2f'%(gl.accuracy(labels,poisson_labels,len(I))))

#Run Laplace learning
#laplace_labels = gl.graph_ssl(W,I,labels[I],algorithm='laplace')
#print('Laplace Learning: %.2f'%(gl.accuracy(labels,laplace_labels,len(I))))

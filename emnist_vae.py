import graphlearning as gl
import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from glob import glob

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

class VAE(nn.Module):
    def __init__(self, layer_widths):
        super(VAE, self).__init__()

        self.lw = layer_widths
        self.fc1 = nn.Linear(self.lw[0], self.lw[1])
        self.fc21 = nn.Linear(self.lw[1], self.lw[2])
        self.fc22 = nn.Linear(self.lw[1], self.lw[2])
        self.fc3 = nn.Linear(self.lw[2], self.lw[1])
        self.fc4 = nn.Linear(self.lw[1], self.lw[0])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.lw[0]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, data.shape[1]), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(data_loader.dataset)))

# default parameters
layer_widths=[400,20]
no_cuda=False
batch_size=128
epochs=1000
learning_rate=1e-3



data, labels = gl.datasets.load("emnist", metric="raw")
print("Training VAE....")


layer_widths = [data.shape[1]] + layer_widths
log_interval = 10    #how many batches to wait before logging training status

#GPU settings
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

#Convert to torch dataloaders
data = data - data.min()
data = data/data.max()
data = torch.from_numpy(data).float()
target = np.zeros((data.shape[0],)).astype(int)
target = torch.from_numpy(target).long()
dataset = MyDataset(data, target)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

#Put model on GPU and set up optimizer
model = VAE(layer_widths).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# #Training epochs
# for epoch in range(1, epochs + 1):
#     train(epoch)


# if not os.path.exists(os.path.join("torch_models", f"{epochs}")):
#     os.makedirs(os.path.join("torch_models", f"{epochs}"))
# print(f"Save pytorch model torch_models/vae_{epochs}.pt")
# torch.save(model.state_dict(), os.path.join("torch_models", f"{epochs}", f"vae_{epochs}.pt"))

model.load_state_dict(torch.load(os.path.join("torch_models", f"{epochs}", f"vae_{epochs}.pt")))
model.eval()

print("Pushing through representations...")
N = dataset.data.shape[0]
data_loader = DataLoader(dataset, batch_size=N, shuffle=False, **kwargs)

#Encode the dataset and save to npz file
with torch.no_grad():
    mu, logvar = model.encode(data.to(device).view(-1, layer_widths[0]))
    print(data.shape)
    print(mu.shape)

    # for batch_idx, (batch_data, batch_labels) in enumerate(data_loader):
    #     print(batch_data.shape)
    #     batch_data_vae = mu.cpu().numpy()
    #     print(f"Done with batch {batch_idx}")


# fnames = sorted(glob(f"torch_models/{epochs}/emnist_vae_*.npz"))
# for i, fname in enumerate(fnames):
#     batch_data = np.load(fname)
#     if i == 0:
#         X, y = batch_data['data'], batch_data['labels']
#     else:
#         X, y = np.concatenate((X, batch_data['data']), axis=0), np.concatenate((y, batch_data['labels']))
#
# np.savez(f"torch_models/{epochs}/emnist_vae.npz", data=X, labels=y)
# print("Done!")

print("saving")
np.savez(f"torch_models/{epochs}/emnist_vae.npz", data=mu.cpu().numpy(), labels=target.cpu().numpy())
# gl.datasets.save(data_vae, target, "emnist", metric="vae", overwrite=True)
#
# print("Constructing similarity graphs")
# W_raw = gl.weightmatrix.knn(data, 20)
# W_vae = gl.weightmatrix.knn(data_vae, 20)
#
# G_raw = gl.graph(W_raw)
# print(f"raw graph connected = {G_raw.isconnected()}")
# G_vae = gl.graph(W_vae)
# print(f"vae graph connected = {G_vae.isconnected()}")
#
# num_train_per_class = 1
# train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
# train_labels = labels[train_ind]
#
# pred_labels_raw = gl.ssl.poisson(W_raw).fit_predict(train_ind,train_labels)
# pred_labels_vae = gl.ssl.poisson(W_vae).fit_predict(train_ind,train_labels)
#
# accuracy_raw = gl.ssl.ssl_accuracy(labels,pred_labels_raw,len(train_ind))
# accuracy_vae = gl.ssl.ssl_accuracy(labels,pred_labels_vae,len(train_ind))
#
# print('Raw Accuracy: %.2f%%'%accuracy_raw)
# print('VAE Accuracy: %.2f%%'%accuracy_vae)

import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from dataset import MyDataset

train_data = MyDataset('./data/FashionMNIST/img/train/', 'train_','./data/FashionMNIST/label/train.csv', ToTensor())
test_data = MyDataset('./data/FashionMNIST/img/test/', 'test_','./data/FashionMNIST/label/test.csv', ToTensor())

batch_size = 16

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Sequential(nn.Linear(28*28, 2048), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(1024, 10), nn.ReLU(inplace=True))

    def forward(self, x):
        r = self.flatten(x)
        r = self.layer1(r)
        r = self.layer2(r)
        r = self.layer3(r)
        return r

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, curr = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{curr:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n==============================")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done.")


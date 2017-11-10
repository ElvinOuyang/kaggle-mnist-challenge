import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix

# download data from MNIST and create mini-batch data loader
torch.manual_seed(1122)

trainset = torchvision.datasets.MNIST(root='./mnist', train=True,
                                      download=True,
                                      transform=transforms.ToTensor())

trainloader = torch.utils.data.DataLoader(trainset, batch_size=250,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./mnist', train=False,
                                     download=True,
                                     transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(testset, batch_size=250,
                                         shuffle=True)


# define and initialize a multilayer-perceptron, a criterion, and an optimizer
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(1 * 28 * 28, 20)
        self.t1 = nn.Tanh()
        self.l2 = nn.Linear(20, 10)
        self.t2 = nn.LogSoftmax()
    def forward(self, x):
        x = x.view(-1, 1*28*28)
        x = self.t1(self.l1(x))
        x = self.t2(self.l2(x))
        return x

mlp = MLP()
criterion = nn.NLLLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.1, momentum=0.9)


# define a training epoch function
def trainEpoch(dataloader, epoch):
    print("Training Epoch %i" % (epoch + 1))
    mlp.train()
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def validateModel(dataloader, epoch):
    mlp.eval()
    test_loss = 0
    correct = 0
    pred = np.array([])
    targ = np.array([])
    for inputs, targets in dataloader:
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = mlp(inputs)
        test_loss += F.nll_loss(outputs, targets, size_average=False).data[0]
        pred = np.append(pred, outputs.topk(1)[1].data.view(1, -1).numpy())
        targ = np.append(targ, targets.data.numpy())
        prd = outputs.topk(1)[1].data
        correct += prd.eq(targets.data.view_as(prd)).sum()
    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)
    cm = confusion_matrix(targ, pred)
    print('[Epoch %i] Accuracy: %.2f, Average Loss: %.2f' %
          (epoch, test_acc, test_loss))
    print(cm)
    return test_loss, test_acc, cm


def testModel(dataloader):
    mlp.eval()
    pred = np.array([])
    for inputs, _ in dataloader:
        inputs = Variable(inputs)
        outputs = mlp(inputs)
        pred = np.append(pred, outputs.topk(1)[1].data.view(1, -1).numpy())
    return pred

# run the training epoch 30 times and test the result

epoch_loss = []
epoch_acc = []
for epoch in range(30):
    trainEpoch(trainloader, epoch)
    loss, acc, _ = validateModel(testloader, epoch)
    epoch_loss.append(loss)
    epoch_acc.append(acc)

"""DEBUGGING CODES
# download one batch of data for input examination
for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    if i > 0:
        break
"""


# test change of kaggle data loader
class MNISTDataset(TensorDataset):
    """Framework for Kaggle MNIST"""
    def __init__(self, data_tensor, target_tensor, transform=None):
        super(MNISTDataset, self).__init__(data_tensor, target_tensor)
        self.transform = transform
    def __getitem__(self, idx):
        data = self.data_tensor[idx]
        target = self.target_tensor[idx]
        if self.transform is not None:
            data = self.transform(data)
            return data, target

x_train = x_train / 1000
y_train /= 1000

trainset = MNISTDataset(torch.Tensor(x_train.tolist()).view(-1, 1, 28, 28),
                        torch.Tensor(y_train.tolist()),
                        transform=transforms.Normalize([0.5], [0.5]))

trainloader = DataLoader(trainset, batch_size=100, shuffle=True)

valset = TensorDataset(torch.Tensor(x_val.tolist()).view(-1, 1, 28, 28),
                       torch.Tensor(y_val.tolist()).long())

valloader = DataLoader(valset, batch_size=250, shuffle=True)

testset = TensorDataset(torch.Tensor(x_test.tolist()).view(-1, 1, 28, 28),
                        torch.Tensor(y_test.tolist()).long())

testloader = DataLoader(testset, batch_size=100, shuffle=False)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.cm as CM

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SequentialSampler

train_file = 'train.csv'
test_file = 'test.csv'
output_file = 'submission.csv'

# set random seed
torch.manual_seed(1122)

# load data
raw_data = np.loadtxt(train_file, skiprows=1, dtype='int', delimiter=',')
test_data = np.loadtxt(test_file, skiprows=1, dtype='int', delimiter=',')

# create np matrices
x_train, x_val, y_train, y_val = train_test_split(
    raw_data[:, 1:], raw_data[:, 0], test_size=0.001)
x_test, y_test = test_data[:, :], np.zeros((test_data.shape[0], 1))

# normalize training data
x_train = normalize(x_train, norm='max', axis=1)
x_val = normalize(x_val, norm='max', axis=1)
x_test = normalize(x_test, norm='max', axis=1)

# create pytorch compatible dataset that has API for automated loaders
trainset = TensorDataset(torch.Tensor(x_train.tolist()).view(-1, 1, 28, 28),
                         torch.Tensor(y_train.tolist()).long())

valset = TensorDataset(torch.Tensor(x_val.tolist()).view(-1, 1, 28, 28),
                       torch.Tensor(y_val.tolist()).long())

testset = TensorDataset(torch.Tensor(x_test.tolist()).view(-1, 1, 28, 28),
                        torch.Tensor(y_test.tolist()).long())

# create pytorch mini-batch loader DataLoader for the dataset
trainloader = DataLoader(trainset, batch_size=100, shuffle=True)

valloader = DataLoader(valset, batch_size=250, shuffle=True)

# for test set, we want to maintain the sequence of the data
testsampler = SequentialSampler(testset)
testloader = DataLoader(testset, batch_size=100, shuffle=False,
                        sampler=testsampler)


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

# run the training epoch 100 times and test the result

epoch_loss = []
epoch_acc = []
for epoch in range(100):
    trainEpoch(trainloader, epoch)
    loss, acc, _ = validateModel(valloader, epoch)
    epoch_loss.append(loss)
    epoch_acc.append(acc)

pred = testModel(testloader)

pred = pred.astype(int)
imageid = np.arange(len(pred)) + 1
submission = pd.DataFrame()
submission['ImageId'] = imageid
submission['Label'] = pred
submission.to_csv('submission.csv', index=False)


# check the actual image for confirmation
def display(img):
    one_image = img.reshape(28, 28)
    plt.axis('off')
    plt.imshow(one_image, cmap=CM.binary)

for idx in torch.randperm(len(pred))[:10]:
    display(test_data[idx])
    plt.show()
    print(pred[idx])

"""DEBUGGING CODES
# download one batch of data for input examination
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs, labels = Variable(inputs), Variable(labels)
    if i > 0:
        break
"""

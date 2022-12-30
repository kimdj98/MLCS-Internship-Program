import torch
from torch import nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from torchinfo import summary

from cnn_network import model

# ==================================
# configs
# ==================================
train_flag = False # set to True to train the model
PATH = './cifar_net.pth'

# ==================================
# train on gpu
# ==================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Train on gpu:", device)
print(torch.cuda.get_device_name())

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# ==================================
# download cifar10 dataset
# ==================================
batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ==================================
# set criterion and optimizer
# ==================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.Adam([{'params': last_params, 'lr': 1e-3},
#                         {'params': intermediate_params, 'lr': 1e-3}])

model = model.to(device) # load model to gpu
if train_flag:
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # change data: inputs and labels tenosrs to cuda
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 500 == 499:    # print every 500 mini-batches (batch size: 4 from dataloader)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
    # ==================================
    # save our trained data 
    # ==================================
    torch.save(model.state_dict(), PATH)
    print('Finished Training')

model.load_state_dict(torch.load(PATH))
# ==================================
# evaluate our model
# ==================================
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
# Accuracy of the network on the 10000 test images: 80.0 %
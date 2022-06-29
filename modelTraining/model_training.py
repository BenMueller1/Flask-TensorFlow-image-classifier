import PIL
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# lets try training it on Food101
LEARNING_RATE = 0.001
BATCH_SIZE = 4    # minibatch size
NUM_EPOCHS = 2    # number of times we pass over the entire training dataset

def main():
    trainset, trainloader, testset, testloader, classes = fetch_data()
    CNNetwork = CNN()
    CNNetwork = train(CNNetwork, trainloader)
    save_trained_model(CNNetwork)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # in-channels is the z index
            # out_channels is the number of filters (same as the z-index, picture it in head for it to make sense)
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),  
            F.relu(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            F.relu(),
            # nn.MaxPool2d(2, 2), don't think this is needed as images are only 32x32 to begin with (and now 8x8)
            torch.flatten(start_dim = 1), # flatten all dimensions except the batch dimension, is syntax right?

            nn.Linear(16*5*5, 120),
            F.relu(),
            nn.Linear(120, 84),
            F.relu(),
            nn.Linear(84, 10),
            # if I'm getting wierd results, use breakpoints to see what the sigmoid is doing
            F.sigmoid()  # turn into probability distribution? This way we can tell users the certainty
        )
        

    def forward(self, x):
        return self.model(x)




def train(CNNetwork, trainloader):
    loss_fn = nn.CrossEntropyLoss()
    # is there a better optimizer to use?
    optimizer = optim.Adam(CNNetwork.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):  # each iteration is a mini batch
            inputs, labels = data
            optimizer.zero_grad()

            outputs = CNNetwork(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            loss_fn.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print('done training')
    return CNNetwork
    


def fetch_data():
    transform = transforms.Compose([  # first change into a tensor, then normalize values to range [-1,1]
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = BATCH_SIZE

    trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    # a loader represents a python iterable over a dataset (each iteration step is a mini_batch of size batch_size)
    # batch_size = number of samples processed before model is updated

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainset, trainloader, testset, testloader, classes


def save_trained_model(model, name):
    PATH = f'./trainedModels/{name}.pth'
    torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    main()
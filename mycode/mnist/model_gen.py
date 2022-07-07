import torch
import torch.nn as nn
import torch.nn.functional as f



def train(net, trainloader, optimizer, loss_fn, batch_size=32, epochs=3):
    net.train() # indicate to layers they're being trained on
    n = len(trainloader.dataset)

    # iterate for n epochs
    for epoch in range(epochs):

        trained = 0

        # iterate and train on each batch
        for i, data in enumerate(trainloader, 0):

            # get data, predict data, find loss, update & take a step to optimize on next iteration
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            trained += batch_size

    # display info every half of epoch
            if not (i + 1) % (len(trainloader) // 2):
                print(f"epoch: {epoch + 1} trained: {trained if trained < n else n}/{n} loss: {loss.item()}")
        print(f"epoch complete trained: {n}/{n} loss: {loss.item()}")
    print("training complete")


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = f.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        out = f.log_softmax(x, dim=1)
        return out
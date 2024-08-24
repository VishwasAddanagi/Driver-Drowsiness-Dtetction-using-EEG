
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

device = torch.device('mps' if torch.backends.mps.is_available()
                      and torch.backends.mps.is_built() else 'cpu')
input_size = 784
hidden_size = 500
num_class = 10
num_epochs = 5
batch_size = 100
"""
1000 images --> 10 batches
"""
lr = 0.001

# Download MNIST Dataset
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train='true',
                                           transform=transforms.ToTensor(),
                                           download='True')
test_dataset = torchvision.datasets.MNIST(root='data',
                                          train='False',
                                          transform=transforms.ToTensor(),
                                          download='True')

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Network

######################### Your Code starts here #######################
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


Model = NeuralNet(input_size, hidden_size, num_class).to(device)

criteria = nn.CrossEntropyLoss()
# nn.MSELoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=lr)

#################### Your code ends here##############################

total_step = len(train_loader)
lossval = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28)
        images = images.to(device)
        labels = labels.to(device)
        ################### Your code starts here####################
        out = Model(images)
        loss = criteria(out, labels)

        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch = ', epoch, ' Itr = ', i, ' Loss = ', loss.item())
            lossval.append(loss.item())
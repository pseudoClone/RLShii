import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#HYPs
batch_size = 64
input_size = 784 # image standard
n_classes = 10
learning_rate = 1e-3
n_epochs = 10

class NN(nn.Module):
        def __init__(self, input_size, n_classes):
                super(NN, self).__init__()
                self.fc1 = nn.Linear(in_features=input_size, out_features=50)
                self.fc2 = nn.Linear(50, out_features=n_classes)

        def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

train_data = datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, n_classes=n_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
        for batch_idx, (data, targets) in enumerate(train_loader):
                data = data.to(device=device)
                targets = targets.to(device=device)

                data = data.reshape(data.shape[0], -1)

                scores = model(data) # FWD pass

                loss = criterion(scores, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch}: {batch_idx}")
        print("..Done Epoch {epoch}")

def check_accuracy(loader, model):
        if loader.dataset.train:
                print("Checking accuracy in train data")
        else:
                print("Checking accuracy on test data")
        
        n_correct = 0
        n_samples = 0
        model.eval()

        with torch.no_grad():
                for x, y in loader:
                        x = x.to(device=device)
                        y = y.to(device=device)
                        x = x.reshape(x.shape[0], -1) # x = torch.flatten(x, start_dim = 1)

                        scores = model(x)
                        _, predictions = scores.max(1)
                        n_correct += (predictions == y).sum()
                        n_samples += predictions.size(0)
                print(f"{n_correct} / {n_samples}")
        model.train()

check_accuracy(train_loader, model=model)
check_accuracy(test_loader, model=model)

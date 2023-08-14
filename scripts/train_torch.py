import ssl
ssl._create_default_https_context = ssl._create_unverified_context # Fixed problem with downloading CIFAR10 dataset

import torch, einx, os
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 256

cifar10_path = os.path.join(os.path.dirname(__file__), "cifar10")
trainset = torchvision.datasets.CIFAR10(root=cifar10_path, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=cifar10_path, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for c in [1024, 512, 256]:
            blocks.append(einx.torch.Linear("b [...|c]", c=c))
            blocks.append(einx.torch.Norm("b [c]"))
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(p=0.1))
        blocks.append(einx.torch.Linear("b [...|c]", c=10))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

net = Net()

# Run once to initialize parameter shapes (see: https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html)
inputs, labels = next(iter(trainloader))
net(inputs)

# Just-in-time compile
# TODO: currently fails due to https://github.com/pytorch/pytorch/issues/104946
# net = torch.compile(net)

optimizer = optim.Adam(net.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

print("Starting training")
for epoch in range(100):
    # Train
    net.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        accuracy = (outputs.argmax(dim=1) == labels).float().mean()

    # Test
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test accuracy after {epoch + 1:5d} epoch: ", float(correct) / total)
import ssl

ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # Fixed problem with downloading CIFAR10 dataset

import torch
import einx
import os
import torchvision
import time
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import einx.nn.torch as einn

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 256

cifar10_path = os.path.join(os.path.dirname(__file__), "cifar10")
trainset = torchvision.datasets.CIFAR10(
    root=cifar10_path, train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root=cifar10_path, train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for c in [1024, 512, 256]:
            blocks.append(einn.Linear("b [...|c]", c=c))
            blocks.append(einn.Norm("[b] c", decay_rate=0.99))
            blocks.append(nn.GELU())
            blocks.append(einn.Dropout("[...]", drop_rate=0.2))
        blocks.append(einn.Linear("b [...|c]", c=10))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


net = Net()

# Call on dummy batch to initialize parameters (before torch.compile!)
inputs, _ = next(iter(trainloader))
net(inputs)

net = net.cuda()
net = torch.compile(net)

optimizer = optim.Adam(net.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()


@torch.compile
def test_step(inputs, labels):
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    return predicted == labels


print("Starting training")
for epoch in range(100):
    t0 = time.time()

    # Train
    net.train()
    for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Test
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            accurate = test_step(inputs, labels)
            total += accurate.size(0)
            correct += int(torch.count_nonzero(accurate))

    print(
        f"Test accuracy after {epoch + 1:5d} epochs {float(correct) / total} "
        f"({time.time() - t0:.2f}sec)"
    )

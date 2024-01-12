import ssl
ssl._create_default_https_context = ssl._create_unverified_context # Fixed problem with downloading CIFAR10 dataset

import torch, einx, os, torchvision, time, jax, optax
import torchvision.transforms as transforms
import einx.nn.equinox as einn
import equinox as eqx
from functools import partial
import jax.numpy as jnp
from typing import List

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

batch_size = 256
rng = jax.random.PRNGKey(42)
def next_rng():
    global rng
    rng, x = jax.random.split(rng)
    return x

cifar10_path = os.path.join(os.path.dirname(__file__), "cifar10")
trainset = torchvision.datasets.CIFAR10(root=cifar10_path, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=cifar10_path, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)





class Block(eqx.Module):
    linear: einn.Linear
    norm: einn.Norm
    dropout: einn.Dropout

    def __init__(self, c):
        self.linear = einn.Linear("b [...|c]", c=c)
        self.norm = einn.Norm("b [c]")
        self.dropout = einn.Dropout("[...]", drop_rate=0.2)

    def __call__(self, x, rng):
        x = self.linear(x, rng=rng)
        x = self.norm(x, rng=rng)
        x = jax.nn.gelu(x)
        x = self.dropout(x, rng=rng)
        return x

class Net(eqx.Module):
    blocks: List[Block]
    classifier: einn.Linear

    def __init__(self):
        self.blocks = [Block(c) for c in [1024, 512, 256]]
        self.classifier = einn.Linear("b [...|c]", c=10)

    def __call__(self, x, rng):
        for block in self.blocks:
            x = block(x, rng=rng)
        return self.classifier(x, rng=rng)

train_net = Net()
inputs, _ = next(iter(trainloader))
train_net(jnp.asarray(inputs), rng=next_rng()) # Run on dummy batch

optimizer = optax.adam(3e-4)
opt_state = optimizer.init(eqx.filter(train_net, eqx.is_array))

@partial(eqx.filter_jit, donate="all")
def update_step(opt_state, net, images, labels, rng):
    def loss_fn(net):
        logits = net(images, rng=rng)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(net)

    updates, new_opt_state = optimizer.update(grads, opt_state, net)
    new_net = eqx.apply_updates(net, updates)

    return new_opt_state, new_net

@partial(eqx.filter_jit, donate="all")
def test_step(net, images, labels):
    logits = net(images, rng=rng)
    accurate = jnp.argmax(logits, axis=1) == jnp.asarray(labels)
    return accurate



print("Starting training")
for epoch in range(100):
    t0 = time.time()

    # Train
    for i, data in enumerate(trainloader):
        inputs, labels = data
        opt_state, train_net = update_step(opt_state, train_net, jnp.asarray(inputs), jnp.asarray(labels), next_rng())

    # Test
    correct = 0
    total = 0
    infer_net = eqx.nn.inference_mode(train_net)
    for data in testloader:
        images, labels = data
        accurate = test_step(infer_net, jnp.asarray(images), jnp.asarray(labels))
        total += accurate.shape[0]
        correct += jnp.sum(accurate)
    
    print(f"Test accuracy after {epoch + 1:5d} epochs: {float(correct) / total} ({time.time() - t0:.2f}sec)")
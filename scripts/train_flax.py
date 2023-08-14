import ssl
ssl._create_default_https_context = ssl._create_unverified_context # Fixed problem with downloading CIFAR10 dataset

from flax import linen as nn
import torch, einx, os, jax, optax
import torchvision
import torchvision.transforms as transforms
import jax.numpy as jnp
from flax.training import train_state

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



class Net(nn.Module):
    @nn.compact
    def __call__(self, x, training):
        for c in [1024, 512, 256]:
            x = einx.flax.Linear("b [...|c]")(x, c=c)
            x = einx.flax.Norm("b [c]")(x)
            x = nn.gelu(x)
            x = nn.Dropout(rate=0.2, deterministic=not training)(x)
        x = einx.flax.Linear("b [...|c]")(x, c=10)
        return x

net = Net()
inputs, labels = next(iter(trainloader))
params = net.init(next_rng(), jnp.asarray(inputs), training=False)['params']
tx = optax.adam(3e-4)
state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)


@jax.jit
def train_step(state, images, labels, rng):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, images, training=True, rngs={"dropout": rng})
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

@jax.jit
def test_step(state, images):
    logits = state.apply_fn({'params': state.params}, images, training=False)
    return logits




print("Starting training")
for epoch in range(100):
    # Train
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        grads, loss, accuracy = train_step(state, jnp.asarray(inputs), jnp.asarray(labels), next_rng())
        state = update_model(state, grads)

    # Test
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        logits = test_step(state, jnp.asarray(images))
        total += logits.shape[0]
        correct += jnp.sum(jnp.argmax(logits, axis=1) == jnp.asarray(labels))
    
    print(f"Test accuracy after {epoch + 1:5d} epoch: ", float(correct) / total)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context # Fixed problem with downloading CIFAR10 dataset

from flax import linen as nn
import torch, einx, os, jax, optax, time
import torchvision
import torchvision.transforms as transforms
import jax.numpy as jnp
from flax.training import train_state
import einx.nn.flax as einn
from functools import partial

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
            x = einn.Linear("b [...|c]", c=c)(x)
            x = einn.Norm("b [c]")(x)
            x = nn.gelu(x)
            x = einn.Dropout("[...]", drop_rate=0.2)(x, training=training)
        x = einn.Linear("b [...|c]", c=10)(x)
        return x

net = Net()
inputs, labels = next(iter(trainloader))
params = net.init(next_rng(), jnp.asarray(inputs), training=False)["params"]
tx = optax.adam(3e-4)
state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)


@partial(jax.jit, donate_argnums=(0,))
def update_step(state, images, labels, rng):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images, training=True, rngs={"dropout": rng})
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss

    grads = jax.grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state

@jax.jit
def test_step(state, images, labels):
    logits = state.apply_fn({"params": state.params}, images, training=False)
    accurate = jnp.argmax(logits, axis=1) == jnp.asarray(labels)
    return accurate



print("Starting training")
for epoch in range(100):
    t0 = time.time()

    # Train
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        state = update_step(state, jnp.asarray(inputs), jnp.asarray(labels), next_rng())

    # Test
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        accurate = test_step(state, jnp.asarray(images), jnp.asarray(labels))
        total += accurate.shape[0]
        correct += jnp.sum(accurate)
    
    print(f"Test accuracy after {epoch + 1:5d} epochs: {float(correct) / total} ({time.time() - t0:.2f}sec)")
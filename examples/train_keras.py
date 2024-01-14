import ssl
ssl._create_default_https_context = ssl._create_unverified_context # Fixed problem with downloading CIFAR10 dataset

import torch, keras, einx, os, torchvision, time
import torchvision.transforms as transforms
import einx.nn.keras as einn
import numpy as np
import tensorflow as tf

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





blocks = []
for c in [1024, 512, 256]:
    blocks.append(einn.Linear("b [...|c]", c=c))
    blocks.append(einn.Norm("[b] c", decay_rate=0.99))
    blocks.append(keras.layers.Activation(keras.activations.gelu))
    blocks.append(einn.Dropout("[...]", drop_rate=0.2))
blocks.append(einn.Linear("b [...|c]", c=10))
model = keras.Sequential(blocks)

inputs, _ = next(iter(trainloader))
inputs = np.asarray(inputs)
model(inputs)


optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss_value = loss_fn(labels, logits)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply(grads, model.trainable_weights)

@tf.function
def test_step(inputs, labels):
    outputs = model(inputs, training=False)
    predicted = tf.math.argmax(outputs, axis=1)
    return predicted == labels

print("Starting training")
for epoch in range(100):
    t0 = time.time()

    # Train
    for data in trainloader:
        inputs, labels = data
        inputs = np.array(inputs)
        labels = np.array(labels)

        train_step(inputs, labels)

    # Test
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = np.array(images)
        labels = np.array(labels)

        accurate = test_step(images, labels)
        total += accurate.shape[0]
        correct += tf.math.count_nonzero(accurate)

    print(f"Test accuracy after {epoch + 1:5d} epochs {float(correct) / total} ({time.time() - t0:.2f}sec)")
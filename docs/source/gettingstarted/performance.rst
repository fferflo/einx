Performance
###########

When an einx function is invoked, the required backend operations are determined from the given Einstein expressions and traced into graph representation. The graph is
then just-in-time compiled into a regular Python function using Python's `exec() <https://docs.python.org/3/library/functions.html#exec>`_.

As a simple example, consider the following einx call:

>>> x = np.zeros((10, 10))
>>> einx.sum("a [b]", x).shape
(10,)

We can inspect the compiled function by passing ``graph=True``:

>>> graph = einx.sum("a [b]", x, graph=True)
>>> print(graph)
def reduce(i0, backend):
    x1 = backend.to_tensor(i0)
    x2 = backend.sum(x1, axis=1)
    return x2

einx passes this string to `exec() <https://docs.python.org/3/library/functions.html#exec>`_ to just-in-time compile the function. It then invokes the function using the
required arguments and backend (i.e. Numpy, Torch, Jax or Tensorflow). The traced function is cached, such that subsequent calls with the same signature of inputs can
reuse it and incur **no overhead other than for cache lookup**.

When using just-in-time compilation like `jax.jit <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_
and `torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_, **einx incurs zero overhead** (other than during
initialization).

Inspecting operations
---------------------

In addition to reducing the overhead, the just-in-time compiled function also allows verifying that the correct backend calls are made. For example:

A sum-reduction that requires a reshape operation:

>>> x = np.zeros((10, 10))
>>> print(einx.sum("b... (g [c])", x, g=2, graph=True))
def reduce(i0, backend):
    x1 = backend.to_tensor(i0)
    x2 = backend.reshape(x1, (10, 2, 5))
    x3 = backend.sum(x2, axis=2)
    return x3

A call to ``einx.dot`` that forwards computation to ``backend.einsum``:

>>> x = np.zeros((10, 10))
>>> print(einx.dot("b... (g [c1|c2])", x, np.ones, g=2, c2=8, graph=True))
def dot(i0, i1, backend):
    x1 = backend.to_tensor(i0)
    x2 = backend.reshape(x1, (10, 2, 5))
    x4 = einx.param.instantiate(i1, shape=(5, 8), in_axis=(0,), out_axis=(1,), batch_axis=(), name="weight", init="dot", backend=backend)
    x5 = backend.einsum("abc,cd->abd", x2, x4)
    x6 = backend.reshape(x5, (10, 16))
    return x6

An operation that requires concatenation of tensors:

>>> x = np.zeros((10, 10, 3))
>>> y = np.ones((10, 10))
>>> print(einx.rearrange("h w c, h w -> h w (c + 1)", x, y, graph=True))
def rearrange(i0, i1, backend):
    x1 = backend.to_tensor(i0)
    x3 = backend.to_tensor(i1)
    x4 = backend.reshape(x3, (10, 10, 1))
    x5 = backend.concatenate([x1, x4], 2)
    return x5

The just-in-time compiled function can also be called directly with the correct arguments to avoid a cache lookup:

>>> graph = einx.rearrange("h w c, h w -> h w (c + 1)", x, y, graph=True)
>>> z = graph(x, y, backend=einx.backend.numpy)
>>> z.shape
(10, 10, 4)
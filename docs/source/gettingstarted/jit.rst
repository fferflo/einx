Just-in-time compilation
########################

When an einx function is invoked, the required backend operations are determined from the given Einstein expressions and traced into graph representation. The graph is
then just-in-time compiled into a regular Python function using Python's `exec() <https://docs.python.org/3/library/functions.html#exec>`_.

As a simple example, consider the following einx call:

>>> x = np.zeros((10, 10))
>>> einx.sum("a [b]", x).shape
(10,)

We can inspect the compiled function by passing ``graph=True``:

>>> graph = einx.sum("a [b]", x, graph=True)
>>> print(graph)
# backend: einx.backend.numpy
def op0(i0):
    x0 = backend.sum(i0, axis=1)
    return x0

einx passes this string and variables such as ``backend`` to `exec() <https://docs.python.org/3/library/functions.html#exec>`_ to just-in-time compile the function.
It then invokes the function using the required arguments. The traced function is cached, such that subsequent calls with the same signature of inputs can
reuse it and incur no overhead other than for cache lookup.

When using just-in-time compilation like `jax.jit <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_, einx incurs zero overhead (other than during
initialization).

Inspecting operations
---------------------

In addition to reducing the overhead, the just-in-time compiled function also allows verifying that the correct backend calls are made. For example:

A sum-reduction that requires a reshape operation:

>>> x = np.zeros((10, 10))
>>> print(einx.sum("b... (g [c])", x, g=2, graph=True))
# backend: einx.backend.numpy
def op0(i0):
    x1 = backend.reshape(i0, (10, 2, 5))
    x0 = backend.sum(x1, axis=2)
    return x0

A call to ``einx.dot`` that forwards computation to ``backend.einsum``:

>>> x = np.zeros((10, 10))
>>> print(einx.dot("b... (g [c1|c2])", x, np.ones, g=2, c2=8, graph=True))
# backend: einx.backend.numpy
def op0(i0, i1):
    x2 = backend.reshape(i0, (10, 2, 5))
    x3 = einx.param.instantiate(i1, shape=(5, 8), in_axis=(0,), out_axis=(1,), batch_axis=(), name="weight", init="dot", backend=backend)
    assert x3.shape == (5, 8)
    x1 = backend.einsum("abc,cd->abd", x2, x3)
    x0 = backend.reshape(x1, (10, 16))
    return x0

A call to ``einx.get_at`` that applies ``backend.vmap`` to handle batch axes:

>>> x = np.zeros((4, 128, 128, 3))
>>> y = np.zeros((4, 1024, 2), "int32")
>>> print(einx.get_at("b [h w] c, b p [2] -> b p c", x, y, graph=True))
# backend: einx.backend.numpy
def op1(i0, i1):
    x1 = i1[:, 0]
    x2 = i1[:, 1]
    x0 = backend.get_at(i0, (x1, x2))
    return (x0,)
def op0(i0, i1, op1=op1):
    op2 = backend.vmap(op1, in_axes=(0, 0), out_axes=(0,))
    op3 = backend.vmap(op2, in_axes=(3, None), out_axes=(2,))
    x0 = op3(i0, i1)
    return x0[0]

An operation that requires concatenation of tensors:

>>> x = np.zeros((10, 10, 3))
>>> y = np.ones((10, 10))
>>> print(einx.rearrange("h w c, h w -> h w (c + 1)", x, y, graph=True))
# backend: einx.backend.numpy
def op0(i0, i1):
    x1 = backend.reshape(i1, (10, 10, 1))
    x0 = backend.concatenate([i0, x1], 2)
    return x0

The just-in-time compiled function can also be called directly with the correct arguments to avoid a cache lookup:

>>> graph = einx.rearrange("h w c, h w -> h w (c + 1)", x, y, graph=True)
>>> z = graph(x, y)
>>> z.shape
(10, 10, 4)
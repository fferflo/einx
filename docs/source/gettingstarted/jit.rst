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
import numpy as np
def op0(i0):
    x0 = np.sum(i0, axis=1)
    return x0

einx passes this string to `exec() <https://docs.python.org/3/library/functions.html#exec>`_ to just-in-time compile the function.
It then invokes the function using the required arguments. The traced function is cached, such that subsequent calls with the same signature of inputs can
reuse it and incur no overhead other than for cache lookup.

The function signature includes the types of the input arguments as well as their shape. einx therefore retraces a function every time it is called
with different input shapes. The environment variable ``EINX_WARN_ON_RETRACE`` can be used to print a warning when excessive retracing takes place. For example,
``EINX_WARN_ON_RETRACE=10`` will issue a warning when a function is retraced 10 times from the same call site.

When using just-in-time compilation like `jax.jit <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_, einx incurs zero overhead (other than during
initialization).

Inspecting operations
---------------------

In addition to reducing the overhead, the just-in-time compiled function also allows verifying that the correct backend calls are made. For example:

A sum-reduction that requires a reshape operation:

>>> x = np.zeros((10, 10))
>>> print(einx.sum("b... (g [c])", x, g=2, graph=True))
import numpy as np
def op0(i0):
    x0 = np.reshape(i0, (10, 2, 5))
    x1 = np.sum(x0, axis=2)
    return x1

A call to ``einx.dot`` that forwards computation to ``np.einsum``:

>>> x = np.zeros((10, 10))
>>> print(einx.dot("b... (g [c1->c2])", x, np.ones, g=2, c2=8, graph=True))
import numpy as np
def op0(i0, i1):
    x0 = np.reshape(i0, (10, 2, 5))
    x1 = np.einsum("abc,cd->abd", x0, i1((5, 8)))
    x2 = np.reshape(x1, (10, 16))
    return x2

A call to ``einx.get_at`` that applies ``jax.vmap`` to handle batch axes:

>>> x = jnp.zeros((4, 128, 128, 3))
>>> y = jnp.zeros((4, 1024, 2), "int32")
>>> print(einx.get_at("b [h w] c, b p [2] -> b p c", x, y, graph=True))
import jax
def op1(i0, i1):
    x0 = i1[:, 0]
    x1 = i1[:, 1]
    x2 = i0[x0, x1]
    return (x2,)
x3 = jax.vmap(op1, in_axes=(0, 0), out_axes=(0,))
x4 = jax.vmap(x3, in_axes=(3, None), out_axes=(2,))
def op0(i0, i1):
    x0, = x4(i0, i1)
    return x0

An operation that requires concatenation of tensors:

>>> x = np.zeros((10, 10, 3))
>>> y = np.ones((10, 10))
>>> print(einx.rearrange("h w c, h w -> h w (c + 1)", x, y, graph=True))
import numpy as np
def op0(i0, i1):
    x0 = np.reshape(i1, (10, 10, 1))
    x1 = np.concatenate([i0, x0], axis=2)
    return x1

The just-in-time compiled function can also be called directly with the correct arguments to avoid a cache lookup:

>>> graph = einx.rearrange("h w c, h w -> h w (c + 1)", x, y, graph=True)
>>> z = graph(x, y)
>>> z.shape
(10, 10, 4)
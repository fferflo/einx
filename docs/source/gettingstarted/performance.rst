Performance
###########

einx determines the necessary steps to execute a given operation, and forwards the computation to the underlying tensor framework. Excluding this overhead,
einx operations have the same runtime as the corresponding tensor framework operations.

When using just-in-time compilation like `jax.jit <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ the overhead that is introduced by einx appears only once during
initialization and results in zero-overhead for subsequent calls.

To reduce the overhead in eager mode, einx caches operations when called for the first time and reuses them when called with the same signature of inputs.
To cache an operation, einx runs the function with tracer objects instead of the input tensors and accumulates all backend calls into a graph representation. When the function is called again,
the overhead is reduced to the cache lookup and graph execution overhead.

einx tries to use as few backend operations as possible to perform the requested computation. The graph can be inspected to determine the backend calls
that einx makes and to ensure that no needless operations are performed (see :ref:`Inspecting operations <inspectingoperations>`).

.. _inspectingoperations:

Inspecting operations
---------------------

einx functions accept the ``graph=True`` argument to return a graph representation of the backend operations. The graph can be
inspected to verify that the expected index-based calls are made. For example:

>>> x = np.zeros((10, 10))
>>> graph = einx.sum("a [b]", x, graph=True)
>>> print(graph)
Graph reduce_stage0("a [b]", I0, op="sum"):
    X2 := instantiate(I0, shape=(10, 10))
    X1 := sum(X2, axis=1)
    return X1

The ``instantiate`` function executes :ref:`tensor factories <lazytensorconstruction>` and converts tensors to the requested backend if required.
The ``einx.sum("a [b]", x)`` call thus corresponds to a single ``backend.sum`` call with ``axis=1``.

Another example of a sum-reduction that requires a reshape operation:

>>> x = np.zeros((10, 10))
>>> graph = einx.sum("b... (g [c])", x, g=2, graph=True)
>>> print(graph)
Graph reduce_stage0("b... (g [c])", I0, op="sum", g=2):
    X3 := instantiate(I0, shape=(10, 10))
    X2 := reshape(X3, (10, 2, 5))
    X1 := sum(X2, axis=2)
    return X1

An example of a call to ``einx.dot`` that forwards computation to ``backend.einsum``:

>>> x = np.zeros((10, 10))
>>> graph = einx.dot("b... (g [c1|c2])", x, np.ones, g=2, c2=8, graph=True)
>>> print(graph)
Graph dot_stage0("b... (g [c1|c2])", I0, I1, g=2, c2=8):
    X5 := instantiate(I0, shape=(10, 10), in_axis=(), out_axis=(0), batch_axis=(1), name="weight", init="dot")
    X4 := reshape(X5, (10, 2, 5))
    X6 := instantiate(I1, shape=(5, 8), in_axis=(0), out_axis=(1), batch_axis=(), name="weight", init="dot")
    X3 := einsum("a b c, c d -> a b d", X4, X6)
    X2 := reshape(X3, (10, 16))
    return X2

An example of an operation that requires concatenation of tensors:

>>> x = np.zeros((10, 10, 3))
>>> y = np.ones((10, 10))
>>> graph = einx.rearrange("h w c, h w -> h w (c + 1)", x, y, graph=True)
>>> print(graph)
Graph rearrange_stage0("h w c, h w -> h w (c + 1)", I0, I1):
    X3 := instantiate(I0, shape=(10, 10, 3))
    X5 := instantiate(I1, shape=(10, 10))
    X4 := reshape(X5, (10, 10, 1))
    X2 := concatenate([X3, X4], 2)
    return X2

The graph can also be called directly with the same arguments to avoid a cache lookup:

>>> z = graph("h w c, h w -> h w (c + 1)", x, y)
>>> z.shape
(10, 10, 4)
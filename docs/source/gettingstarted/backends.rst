Backends
########

Overview
========

einx executes tensor operations by just-in-time compiling them to Python code snippets and invoking them with the given tensor arguments.
The details of the compilation depend on the backend that is selected for an operation.

einx provides different backends for this purpose: There is at least one backend per supported tensor framework (*e.g.*, Numpy, PyTorch, Jax), and some
tensor frameworks have multiple backends available. This allows invoking different implementations for the same einx operation.

Every einx operation supports a ``backend`` keyword parameter that may be used to specify which backend to use for the operation. If no backend
is specified, it is determined implicitly from the types of the given tensor arguments. For example:

..  code:: python

    x_numpy = np.ones((2, 3))
    x_torch = torch.ones(2, 3)
    x_jax = jnp.ones((2, 3))

    y = einx.sum("a [b]", x_numpy)                          # Uses the default numpy backend
    y = einx.sum("a [b]", x_torch)                          # Uses the default torch backend
    y = einx.sum("a [b]", x_jax)                            # Uses the default jax backend
    y = einx.sum("a [b]", x_numpy, backend="numpy")         # Uses numpy backend
    y = einx.sum("a [b]", x_numpy, backend="jax.vmap")      # Uses jax.vmap backend

Numpy tensors can be mixed with other frameworks in the same operation, in which case the default backend of the other framework is used.
Frameworks other than Numpy cannot be mixed in the same operation.

..  code:: python

    y = einx.add("a b, a b", x_numpy, x_torch)              # Uses the default torch backend
    y = einx.add("a b, a b", x_numpy, x_jax)                # Uses the default jax backend
    y = einx.add("a b, a b", x_torch, x_jax)                # Raises an exception

Available backends
==================

einx provides at least one backend for each supported tensor framework with a name indicating the framework (*e.g.*, ``backend="torch"``).
These backends are used by default when tensors of the corresponding framework are passed as arguments. einx currently supports the following tensor frameworks:

- **Numpy**: `https://numpy.org/ <https://numpy.org/>`_
- **PyTorch**: `https://pytorch.org/ <https://pytorch.org/>`_
- **Jax**: `https://jax.readthedocs.io/ <https://jax.readthedocs.io/>`_
- **Tensorflow**: `https://www.tensorflow.org/ <https://www.tensorflow.org/>`_
- **MLX**: `https://ml-explore.github.io/mlx <https://ml-explore.github.io/mlx>`_
- **Tinygrad**: `https://tinygrad.org/ <https://tinygrad.org/>`_
- **ArrayAPI**: `https://data-apis.org/array-api/latest/ <https://data-apis.org/array-api/latest/>`_. einx supports any ArrayAPI-compliant framework (including, *e.g.*, `Dask <https://docs.dask.org/en/stable/array.html>`_) via the ``arrayapi`` backend.

For each framework,
there are specialized backends that compile operations to particular types of tensor notations that are supported by the framework.
These include :doc:`Numpy-like notation </comparison/numpylike>` (``backend="{framework}.numpylike"``),
:doc:`einsum notation </comparison/ein>` (``backend="{framework}.einsum"``),
and :doc:`vmap notation </comparison/vmap>` (``backend="{framework}.vmap"``). The default backends for each framework
consist of different choices of these specialized backends for different operations.

Code examples
=============

To inspect how a backend implements an operation, the code representation may be returned by passing ``graph=True`` to the function.
The following sections show several example operations.

Sum-reduction
*************

The following operation represents a simple sum-reduction over the second axis of a 2D tensor:

..  code:: python

    >>> x = jnp.ones((2, 10))

    >>> print(einx.sum("a [b]", x, y, backend="jax.numpylike", graph=True))
    import jax.numpy as jnp
    def op(a):
        a = jnp.sum(a, axis=1)
        return a

    >>> print(einx.sum("a [b]", x, y, backend="jax.einsum", graph=True))
    import jax.numpy as jnp
    def op(a):
        a = jnp.einsum("ab->a", a)
        return a

    >>> print(einx.sum("a [b]", x, y, backend="jax.vmap", graph=True))
    import jax.numpy as jnp
    import jax
    def op(a):
        b = jax.vmap(jnp.sum, in_axes=0, out_axes=0)
        c = b(a)
        return c

Batched matrix multiplication
*****************************

The following operation represents a batched matrix multiplication where the batch dimension is the last dimension in the tensor, and some axes are flattened:

..  code:: python

    >>> x = jnp.zeros((2, 3 * 8))
    >>> y = jnp.zeros((3, 4 * 8))

    >>> print(einx.dot("a ([b] x), [b] (c x) -> a (c x)", x, y, backend="jax.numpylike", graph=True))
    import jax.numpy as jnp
    def op(a, b):
        a = jnp.reshape(a, (2, 3, 8))
        a = jnp.transpose(a, (2, 0, 1))
        b = jnp.reshape(b, (3, 4, 8))
        b = jnp.transpose(b, (2, 0, 1))
        c = jnp.matmul(a, b)
        c = jnp.transpose(c, (1, 2, 0))
        c = jnp.reshape(c, (2, 32))
        return c

    >>> print(einx.dot("a ([b] x), [b] (c x) -> a (c x)", x, y, backend="jax.einsum", graph=True))
    import jax.numpy as jnp
    def op(a, b):
        a = jnp.reshape(a, (2, 3, 8))
        b = jnp.reshape(b, (3, 4, 8))
        c = jnp.einsum("abc,bdc->adc", a, b)
        c = jnp.reshape(c, (2, 32))
        return c

    >>> print(einx.dot("a ([b] x), [b] (c x) -> a (c x)", x, y, backend="jax.vmap", graph=True))
    import jax
    import jax.numpy as jnp
    def op(a, b):
        c = jax.vmap(jnp.dot, in_axes=(1, 1), out_axes=0)
        c = jax.vmap(c, in_axes=(None, 1), out_axes=0)
        c = jax.vmap(c, in_axes=(0, None), out_axes=0)
        a = jnp.reshape(a, (2, 3, 8))
        b = jnp.reshape(b, (3, 4, 8))
        d = c(a, b)
        d = jnp.reshape(d, (2, 32))
        return d

Indexing operation
******************

The following operation gathers pixel colors from a batch of images at the coordinates specified by an index tensor:

..  code:: python

    >>> x = jnp.zeros((2, 128, 128, 3)) # batch of images
    >>> y = jnp.zeros((50, 2)) # set of 50 pixels

    # This backend computes the indices into a flattened value tensor manually
    >>> print(einx.get_at("b [h w] c, p [2] -> b p c", x, y, backend="jax.numpylike", graph=True))
    import jax.numpy as jnp
    def op(a, b):
        a = jnp.reshape(a, (98304,))
        c = jnp.arange(2, dtype="int32")
        c = jnp.multiply(c, 49152)
        c = jnp.reshape(c, (2, 1, 1))
        d = jnp.multiply(b[:, 0], 384)
        d = jnp.reshape(d, (1, 50, 1))
        e = jnp.add(c, d)
        b = jnp.multiply(b[:, 1], 3)
        b = jnp.reshape(b, (1, 50, 1))
        f = jnp.add(e, b)
        g = jnp.arange(3, dtype="int32")
        g = jnp.reshape(g, (1, 1, 3))
        h = jnp.add(f, g)
        i = jnp.take(a, h)
        return i

    # This backend uses vmap to vectorize an elementary indexing operation
    >>> print(einx.get_at("b [h w] c, p [2] -> b p c", x, y, backend="jax.vmap", graph=True))
    import jax
    def c(d, e):
        return d[e[0], e[1]]
    def op(a, b):
        f = jax.vmap(c, in_axes=(0, None), out_axes=0)
        f = jax.vmap(f, in_axes=(None, 0), out_axes=1)
        f = jax.vmap(f, in_axes=(3, None), out_axes=2)
        g = f(a, b)
        return g

    # The einsum backend does not support einx.get_at
    >>> print(einx.get_at("b [h w] c, p [2] -> b p c", x, y, backend="jax.einsum", graph=True))
    raises OperationNotSupportedError: get_at operation is not supported by the jax.einsum backend.

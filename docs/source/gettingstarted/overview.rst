Overview
########

Introduction
------------

einx allows formulating many tensor operations as concise expressions using few powerful abstractions. It is inspired by
`einops <https://github.com/arogozhnikov/einops>`_ and `einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_.
For an introduction to Einstein-notation see
`this great einops tutorial <https://nbviewer.org/github/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb>`_ and a
:doc:`comparison with index-based notation </gettingstarted/cheatsheet>`.

einx can be integrated easily into existing code and seamlessly works with tensors from different frameworks (Numpy, Torch, Jax, Tensorflow):

..  code::

    import einx

    import numpy as np
    x = np.ones((3, 4))
    y = einx.sum("a [b]", x)

    import torch
    x = torch.ones(3, 4)
    y = einx.sum("a [b]", x)

It provides several powerful abstractions, as well as easy-to-use specializations:

* :func:`einx.rearrange` maps tensors from input to output expressions by permuting axes, inserting new
  broadcasted axes, concatenating and splitting the tensors as requested. Other einx functions support the same rearranging, but also
  compute additional operations on the tensors (see :doc:`How does einx handle input and output tensors? </faq/flatten>`).
* :func:`einx.reduce` applies a reduction operation on tensors along the axes specified in the input expressions, such as
  `np.sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_, `np.mean <https://numpy.org/doc/stable/reference/generated/numpy.mean.html>`_
  or `np.any <https://numpy.org/doc/stable/reference/generated/numpy.any.html>`_. Specializations are provided as top-level functions
  in the ``einx.*`` namespace following Numpy naming: ``einx.{sum|prod|mean|any|all|max|min|count_nonzero|...}``.
* :func:`einx.elementwise` applies element-by-element operations on tensors, such as
  `np.add <https://numpy.org/doc/stable/reference/generated/numpy.add.html>`_, `np.multiply <https://numpy.org/doc/stable/reference/generated/numpy.multiply.html>`_
  or `np.where <https://numpy.org/doc/stable/reference/generated/numpy.where.html>`_. Specializations are provided as top-level functions
  in the ``einx.*`` namespace following Numpy naming: ``einx.{add|multiply|logical_and|where|equal|...}``.
* :func:`einx.dot` applies general dot-products between tensors similar to `np.einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_.
* :func:`einx.vmap` allows vectorizing arbitrary functions over batched tensors (see e.g. `vectorization in jax <https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html>`_).

Einstein expressions
--------------------

An einx expression that describes a tensor's shape consists of named and unnamed axes (``a``, ``8``), compositions ``(a b)``, ellipses ``a...``
and concatenations ``(a + b)``. Unlike in einops, an ellipsis always repeats the expression that appears directly in front of it

..  code::

    einx.rearrange("b c h w  -> b h w  c", x)
    # same as
    einx.rearrange("b c s... -> b s... c", x)

and can appear multiple times per expression and be composed with other expressions arbitrarily:

..  code::

    # Divide image into patches (space-to-depth)
    einx.rearrange("b (h h2) (w w2) c -> b h w  (h2 w2 c)", x, h2=2, w2=2)
    # same as
    einx.rearrange("b (s s2)...     c -> b s... (s2... c)", x, s2=2) # or s2=(2, 2)

This facilitates writing dimension-agnostic code even for complex operations. To be fully compatible with einops-style notation, einx implicitly
converts anonymous ellipses (that do not have a preceeding expression) by adding a name in front:

..  code::

    einx.rearrange("b ... -> ... b", x)
    # same as
    einx.rearrange("b _anonymous_ellipsis_variable... -> _anonymous_ellipsis_variable... b", x)

einx introduces concatenations as a way to specify operations such as `np.concatenate <https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html>`_,
`np.split <https://numpy.org/doc/stable/reference/generated/numpy.split.html>`_,
`np.stack <https://numpy.org/doc/stable/reference/generated/numpy.stack.html>`_,
`einops.pack and einops.unpack <https://einops.rocks/4-pack-and-unpack/>`_ in pure Einstein notation:

..  code::

    # Pack/ unpack 
    z    = einx.rearrange("h w c, h w -> h w (c + 1)", x, y)
    x, y = einx.rearrange("h w (c + 1) -> h w c, h w", z)

    # Append number to channels
    einx.rearrange("... c, 1 -> ... (c + 1)", x, [42])

einx uses a `SymPy <https://www.sympy.org/en/index.html>`_ solver to determine the values of named axes in Einstein expressions.
In many cases, the shapes of the input tensors provide enough constraints to determine the values of all named axes. For other cases, einx functions accept
``**parameters`` that can be used to specify the values of some or all named axes and provide additional constraints to the solver:

..  code::

    x = np.zeros((10,))
    einx.rearrange("(a b) -> a b", x)           # Fails: Values of a and b cannot be determined
    einx.rearrange("(a b) -> a b", x, a=5)      # Succeeds: b determined by solver
    einx.rearrange("(a b) -> a b", x, b=2)      # Succeeds: a determined by solver
    einx.rearrange("(a b) -> a b", x, a=5, b=2) # Succeeds
    einx.rearrange("(a b) -> a b", x, a=5, b=5) # Fails: Conflicting constraints

Bracket notation
----------------

einx introduces the ``[]``-notation to specify how operations should be vectorized. ``[]`` denotes axes that an operation is applied on, while all other
axes are batch axes and vectorized over.

This corresponds to the ``axis`` argument of numpy functions:

..  code::

    einx.sum("a [b]", x)
    # same as
    np.sum(x, axis=1)

    einx.sum("a [...]", x)
    # same as
    np.sum(x, axis=tuple(range(1, x.ndim)))

    einx.sum("b... (g [c])", x)
    # requires reshapes in numpy

Operations are sensitive to the positioning of brackets, e.g. allowing for flexible ``keepdims=True`` behavior out-of-the-box:

..  code::

    einx.sum("b... [c]", x)                # Shape: b...
    einx.sum("b... ([c])", x)              # Shape: b... 1
    einx.sum("b... [c]", x, keepdims=True) # Shape: b... 1

In the second example, ``c`` is reduced within the composition ``(c)``, resulting in an empty composition ``()``, i.e. a trivial axis with size 1.

Other examples of bracket notation:

..  code::

    # Add bias onto channels
    einx.add("b... [c]", x, bias) # bias has shape c

    # Map from c1 to c2 channels using a linear map
    einx.dot("b [c1] -> b [c2]", x, w)
    # Same call in shorter notation:
    einx.dot("b [c1|c2]", x, w)

    # Mean pooling with kernel_size=4 and stride=4 (must be evenly divisible)
    einx.mean("b (s [s2])... c", x, s2=4)

``einx.vmap`` allows vectorizing arbitrary functions using the same bracket notation, e.g.:

..  code::

    # Compute the mean of the first tensor and the max of the second
    def op(x, y): # c, d -> 2
        return np.stack([np.mean(x), np.max(y)])

    einx.vmap("b1 [c] b2, b2 [d] -> b2 [2] b1", x, y, op=op)

The arguments that arrive at ``op`` have shapes that match the marked subexpressions. Other einx functions can similarly be formulated using ``einx.vmap``:

..  code::

    einx.mean("a b [c]", x)
    einx.vmap("a b [c] -> a b", x, op=np.mean)

    einx.add("a b, b", x, y)
    einx.vmap("a b, b -> a b", x, y, op=np.add) # Function is applied on scalars

    einx.dot("a b, b c -> a c", x, y)
    einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot)

While using the option without ``einx.vmap`` is often faster, ``einx.vmap`` also allows vectorizing functions that do not support
batch axes (e.g. `map_coordinates <https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.ndimage.map_coordinates.html>`_).

.. _lazytensorconstruction:

Lazy tensor construction
------------------------

Instead of passing tensors, all operations also accept tensor factories (e.g. a function ``lambda shape: tensor``) that are
called to create the corresponding tensor when the shape is resolved.

..  code::

    einx.dot("b... [c1|c2]", x, np.ones, c2=32) # Second input is constructed using np.ones

This is especially useful in the context of deep learning modules, where the shapes of a layer's weights are chosen to match with the desired
input and output shapes (see :doc:`Neural Networks </gettingstarted/neuralnetworks>`).

Performance
-----------

einx determines the necessary steps to execute a given operation, and forwards the computation to the underlying tensor framework. Excluding this overhead,
einx operations have the same runtime as the corresponding tensor framework operations.

When using just-in-time compilation like `jax.jit <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ or
`torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_, the overhead that is introduced by einx appears only once during
initialization and results in zero-overhead for subsequent calls.

To reduce the overhead in eager mode, einx caches operations when called for the first time and reuses them when called with the same signature of inputs.
To cache an operation, einx runs the function with tracer objects instead of the input tensors and accumulates all backend calls into a graph representation. When the function is called again,
the overhead is reduced to the cache lookup and graph execution overhead.

einx tries to use as few backend operations as possible to perform the requested computation. The graph can be used to examine the specific backend calls
that einx makes and to ensure that no needless operations are performed. The graph can be accessed by passing ``graph=True`` to an einx function, and can be
converted to string representation:

..  code:: python

    >>> x = np.zeros((10, 10))
    >>> graph = einx.sum("a [b]", x, graph=True)
    >>> print(str(graph))

    Graph reduce_stage0("a [b]", I0, op="sum"):
        X2 := instantiate(I0, shape=(10, 10))
        X1 := sum(X2, (1), keepdims=False)
        return X1

The ``instantiate`` function executes tensor factories if they are given, and converts tensors to the requested backend. The ``einx.sum("a [b]", x)`` call
thus reduces to a single ``backend.sum`` call with ``axis=1``.



Another example of a sum-reduction that requires a reshape operation:

..  code:: python

    >>> x = np.zeros((10, 10))
    >>> graph = einx.sum("b... (g [c])", x, g=2, graph=True)
    >>> print(str(graph))

    Graph reduce_stage0("b... (g [c])", I0, op="sum", g=2):
        X3 := instantiate(I0, shape=(10, 10))
        X2 := reshape(X3, (10, 2, 5))
        X1 := sum(X2, (2), keepdims=False)
        return X1

An example of a call to ``einx.dot`` that forwards computation to ``backend.einsum``:

..  code:: python

    >>> x = np.zeros((10, 10))
    >>> graph = einx.dot("b... (g [c1|c2])", x, np.ones, g=2, c2=8, graph=True)
    >>> print(str(graph))

    Graph dot_stage0("b... (g [c1|c2])", I0, I1, g=2, c2=8):
        X5 := instantiate(I0, shape=(10, 10), in_axis=(), out_axis=(0), batch_axis=(1))
        X4 := reshape(X5, (10, 2, 5))
        X6 := instantiate(I1, shape=(5, 8), in_axis=(0), out_axis=(1), batch_axis=())
        X3 := einsum("a b c, c d -> a b d", X4, X6)
        X2 := reshape(X3, (10, 16))
        return X2

.. note::

    ``einx.dot`` also passes the ``in_axis``, ``out_axis`` and ``batch_axis`` arguments to tensor factories, e.g. to determine the fan-in and fan-out
    of neural network layers and initialize the weights accordingly (see `Neural Networks </gettingstarted/neuralnetworks>`).

An example of an operation that requires concatenation of tensors:

..  code:: python

    >>> x = np.zeros((10, 10, 3))
    >>> y = np.ones((10, 10))
    >>> graph = einx.rearrange("h w c, h w -> h w (c + 1)", x, y, graph=True)
    >>> print(str(graph))

    Graph rearrange_stage0("h w c, h w -> h w (c + 1)", I0, I1):
        X3 := instantiate(I0, shape=(10, 10, 3))
        X5 := instantiate(I1, shape=(10, 10))
        X4 := reshape(X5, (10, 10, 1))
        X2 := concatenate([X3, X4], 2)
        return X2

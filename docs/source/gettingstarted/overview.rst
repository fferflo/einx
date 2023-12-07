.. toctree::
   :caption: Overview
   :maxdepth: 3

Overview
########

Introduction
------------

einx allows formulating many tensor operations as concise expressions using few powerful abstractions. It is inspired by
`einops <https://github.com/arogozhnikov/einops>`_ and `einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_. einx introduces:

* Composable :ref:`Einstein expressions <einsteinexpressions>` with ``[]``-:ref:`notation <bracketnotation>` compatible with einops-notation
  (see :doc:`Comparison with einops </faq/einops>`).
* Generalized :doc:`neural network layers </gettingstarted/neuralnetworks>` that are formulated using einx expressions.
* Numpy-like naming convention: ``einx.{sum|any|max|count_nonzero|where|add|logical_and|flip|...}``
* :ref:`Inspection of backend operations <inspectingoperations>` in index-based notation that are invoked in a given einx call.

einx can be integrated easily into existing code and works with tensor frameworks Numpy, Torch, Jax and Tensorflow:

..  code::

    import einx

    import numpy as np
    x = np.ones((3, 4))
    y = einx.sum("a [b]", x)

    import torch
    x = torch.ones(3, 4)
    y = einx.sum("a [b]", x)

It incurs no overhead in just-in-time compiled code, and only a marginal overhead in eager mode by caching operations on the first call (see :ref:`Performance <performance>`).

.. _einsteinexpressions:

Einstein expressions
--------------------

For an introduction to the basics of Einstein-notation for tensor manipulation see
`this great einops tutorial <https://nbviewer.org/github/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb>`_, a
:doc:`comparison with index-based notation </gettingstarted/cheatsheet>` and an :doc:`overview of how einx parses Einstein expressions </faq/solver>`. The following gives a short
summary of the most important concepts.

An Einstein expression describes a tensor's shape and consists of named and unnamed axes (``a``, ``1``), axis lists ``a b``, compositions ``(a b)``, ellipses ``a...``
and concatenations ``(a + b)``.

An axis **list** specifies the axes of a tensor in order, separated by spaces:

>>> x = np.ones((2, 3, 4)) # Expression: a b c
>>> einx.rearrange("a b c  -> a c b", x).shape
(2, 4, 3)

A **composition** represents multiple axes as a single axis in `row-major order <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_:

>>> x = np.ones((2, 3, 4)) # Expression: a b c
>>> einx.rearrange("a b c  -> (a b) c", x).shape
(6, 4)

This uses a `reshape <https://numpy.org/doc/stable/reference/generated/numpy.reshape.html>`_ operation which does not change the underlying data.
The value of the new axis is the product of the composed axes. See `this einops tutorial <https://nbviewer.org/github/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb>`_
for hands-on illustrations of axis composition using a batch of images. Axes can be decomposed analogously:

>>> x = np.ones((6, 4)) # Expression: (a b) c
>>> einx.rearrange("(a b) c  -> a b c", x, a=2).shape
(2, 3, 4)

Since a decomposition is ambiguous w.r.t. the values of the individual axes (only the product is known), additional constraints for the axis values
must be passed as parameters, e.g. ``a=2``. 

An **ellipsis** repeats the expression that appears directly in front of it:

>>> x = np.ones((2, 3, 4)) # Expression: a x...
>>> einx.rearrange("a x...  -> x... a", x).shape # Expands to "a x.0 x.1 -> x.0 x.1 a"
(3, 4, 2)

The number of repetitions is determined from the rank of the input tensors. This simplifies expressions and facilitates writing dimension-agnostic code.
Ellipses can be composed with other expressions arbitrarily which allows formulating complex multi-dimensional operations in a concise way. For example:

..  code::

    # Mean-pooling with stride 4 (if evenly divisible)
    einx.mean("b (s [s2])... c", x, s2=4) # [] marks axes along which the mean is computed

    # Divide an into a list of patches with size 4
    einx.rearrange("(s s2)... c -> (s...) s2... c", x, s2=4)

To be fully compatible with einops-style notation where an ellipsis can only appear once without a preceding expression, einx implicitly
converts anonymous ellipses by adding an axis in front:

..  code::

    einx.rearrange("b ... -> ... b", x)
    # same as
    einx.rearrange("b _anonymous_ellipsis_variable... -> _anonymous_ellipsis_variable... b", x)

einx introduces axis **concatenations** as a way to specify operations such as `np.concatenate <https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html>`_,
`np.split <https://numpy.org/doc/stable/reference/generated/numpy.split.html>`_,
`np.stack <https://numpy.org/doc/stable/reference/generated/numpy.stack.html>`_,
`einops.pack and einops.unpack <https://einops.rocks/4-pack-and-unpack/>`_ in pure Einstein notation:

..  code::

    # Pack/ unpack 
    z    = einx.rearrange("h w c, h w -> h w (c + 1)", x, y)
    x, y = einx.rearrange("h w (c + 1) -> h w c, h w", z)

    # Append number to channels
    einx.rearrange("... c, 1 -> ... (c + 1)", x, [42])

einx uses a `SymPy <https://www.sympy.org/en/index.html>`_-based solver to determine the values of named axes in Einstein expressions (see :doc:`How does einx parse Einstein expressions? </faq/solver>`).
In many cases, the shapes of the input tensors provide enough constraints to determine the values of all named axes. For other cases, einx functions accept
``**parameters`` that can be used to specify the values of some or all named axes and provide **additional constraints** to the solver:

..  code::

    x = np.zeros((10,))
    einx.rearrange("(a b) -> a b", x)           # Fails: Values of a and b cannot be determined
    einx.rearrange("(a b) -> a b", x, a=5)      # Succeeds: b determined by solver
    einx.rearrange("(a b) -> a b", x, b=2)      # Succeeds: a determined by solver
    einx.rearrange("(a b) -> a b", x, a=5, b=2) # Succeeds
    einx.rearrange("(a b) -> a b", x, a=5, b=5) # Fails: Conflicting constraints

Internally, einx uses **Einstein expression trees** to represent the shapes of tensors, for example:

.. figure:: /images/stage3-tree.png
  :height: 240
  :align: center

  Einstein expression tree for ``b (s [r])... c`` for tensor with shape ``(2, 4, 8, 3)`` and constraint ``r=4``.

For more details, see :doc:`How does einx parse Einstein expressions? </faq/solver>`

.. _bracketnotation:

Bracket notation
----------------

einx introduces the ``[]``-notation to specify how operations should be vectorized. ``[]`` denotes axes that an operation is applied on, while all other
axes are batch axes and vectorized over. This corresponds to the ``axis`` argument in index-based notation:

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

    # Map from c1 to c2 channels using a linear map w
    einx.dot("b [c1] -> b [c2]", x, w)
    # Same call in shorter notation:
    einx.dot("b [c1|c2]", x, w)

    # Mean pooling with stride 4 (if evenly divisible)
    einx.mean("b (s [s2])... c", x, s2=4)

    # Reverse elements along the last two axes
    einx.flip("... [b c]", x)

:func:`einx.vmap` allows vectorizing arbitrary functions using the same bracket notation, e.g.:

..  code::

    # Compute the mean of the first tensor and the max of the second
    def op(x, y): # c, d -> 2
        return np.stack([np.mean(x), np.max(y)])

    # Apply op to batched tensors x and y
    einx.vmap("b1 [c] b2, b2 [d] -> b2 [2] b1", x, y, op=op)

The arguments that are passed to ``op`` have shapes that match the marked subexpressions. Other einx functions can similarly be formulated using :func:`einx.vmap`:

..  code::

    einx.mean("a b [c]", x)
    einx.vmap("a b [c] -> a b", x, op=np.mean)

    einx.add("a b, b", x, y)
    einx.vmap("a b, b -> a b", x, y, op=np.add) # Function is applied on scalars

    einx.dot("a b, b c -> a c", x, y)
    einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot)

    einx.flip("a [b]", x)
    einx.vmap("a [b] -> a [b]", x, op=np.flip)

While using the option without :func:`einx.vmap` is often faster, :func:`einx.vmap` also allows vectorizing functions that do not inherently support
batch axes (e.g. `map_coordinates <https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.ndimage.map_coordinates.html>`_).

.. _apisummary:

API
---

einx provides several powerful abstractions:

1. :func:`einx.rearrange` transforms tensors between Einstein expressions by reshaping, permuting axes, inserting new
   broadcasted axes, concatenating and splitting as required.

2. :func:`einx.vmap_with_axis` applies functions that accept the ``axis`` argument and follow
   `numpy broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ (e.g. `np.multiply <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_,
   `np.flip <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_, `np.sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_).

3. :func:`einx.vmap` applies arbitrary functions using vectorization
   (see `this jax tutorial <https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html>`_ for an introduction to vectorization).

4. :func:`einx.dot` applies general dot-products similar to `np.einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_.

Many easy-to-use specializations are also included as top-level functions in the ``einx.*`` namespace following Numpy naming conventions:

* ``einx.{sum|prod|mean|any|all|max|min|count_nonzero|...}`` (see :func:`einx.reduce`).
* ``einx.{add|multiply|logical_and|where|equal|...}`` (see :func:`einx.elementwise`).
* ``einx.{flip|roll|...}`` (see :func:`einx.vmap_with_axis`).
* ``einx.{get_at|set_at|add_at|...}`` (see :func:`einx.index`).

See the :doc:`API reference </api>` for a list of functions.

.. _lazytensorconstruction:

Lazy tensor construction
------------------------

Instead of passing tensors, all operations also accept tensor factories (e.g. a function ``lambda shape: tensor``) that are
called to create the corresponding tensor when the shape is resolved.

..  code::

    einx.dot("b... [c1|c2]", x, np.ones, c2=32) # Second input is constructed using np.ones

This is especially useful in the context of deep learning modules, where the shapes of a layer's weights are chosen to match with the desired
input and output shapes (see :doc:`Neural networks </gettingstarted/neuralnetworks>`).

.. _performance:

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

einx tries to use as few backend operations as possible to perform the requested computation. The graph can be inspected to determine the backend calls
that einx makes and to ensure that no needless operations are performed (see :ref:`Inspecting operations <inspectingoperations>`).

.. _inspectingoperations:

Inspecting operations
---------------------

einx functions accept the ``graph=True`` argument to return a graph representation of the backend operations. The graph can be
inspected to verify that the expected index-based calls are made. For example:

..  code:: python

    >>> x = np.zeros((10, 10))
    >>> graph = einx.sum("a [b]", x, graph=True)
    >>> print(str(graph))

    Graph reduce_stage0("a [b]", I0, op="sum"):
        X2 := instantiate(I0, shape=(10, 10))
        X1 := sum(X2, axis=1)
        return X1

The ``instantiate`` function executes :ref:`tensor factories <lazytensorconstruction>` and converts tensors to the requested backend if required.
The ``einx.sum("a [b]", x)`` call thus corresponds to a single ``backend.sum`` call with ``axis=1``.

Another example of a sum-reduction that requires a reshape operation:

..  code:: python

    >>> x = np.zeros((10, 10))
    >>> graph = einx.sum("b... (g [c])", x, g=2, graph=True)
    >>> print(str(graph))

    Graph reduce_stage0("b... (g [c])", I0, op="sum", g=2):
        X3 := instantiate(I0, shape=(10, 10))
        X2 := reshape(X3, (10, 2, 5))
        X1 := sum(X2, axis=2)
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

    ``einx.dot`` passes the ``in_axis``, ``out_axis`` and ``batch_axis`` arguments to tensor factories, e.g. to determine the fan-in and fan-out
    of neural network layers and initialize the weights accordingly (see :doc:`Neural networks </gettingstarted/neuralnetworks>`).

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

Related libraries
-----------------

* `einops <https://github.com/arogozhnikov/einops>`_
* `einsum in Numpy <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_
* `eindex <https://github.com/arogozhnikov/eindex>`_
* `torchdim <https://github.com/facebookresearch/torchdim>`_
* `einindex <https://github.com/malmaud/einindex>`_
* `einshape <https://github.com/google-deepmind/einshape>`_
* `einop <https://github.com/cgarciae/einop>`_
* `eingather <https://twitter.com/francoisfleuret/status/1661372730241953793>`_
* `Named tensors in PyTorch <https://pytorch.org/docs/stable/named_tensor.html>`_
* `Named axes in Jax <https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html>`_
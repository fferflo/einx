Tutorial: Einstein notation
###########################

This tutorial introduces the Einstein notation that is used in einx. It is inspired by and compatible with the notation used in `einops <https://github.com/arogozhnikov/einops>`_,
but follows a novel design based on a full composability of expressions, and the introduction of ``[]``-notation and intuitive shorthands. When combined, these features
allow for a concise and expressive formulation of a large variety of tensor operations. (See :doc:`How does einx compare with einops? </faq/einops>` for a complete list 
of differences.)

Introduction
------------

An Einstein expression provides a description of the dimensions of a given tensor. In the simplest case, each dimension is given a unique name (``a``, ``b``, ``c``), and the names
are listed to form an Einstein expression:

>>> x = np.ones((2, 3, 4))
>>> einx.matches("a b c", x) # Check whether expression matches the tensor's shape
True
>>> einx.matches("a b", x)
False

One application of Einstein expressions is to formulate tensor operations such as reshaping and permuting axes in an intuitive way. Instead of defining an
operation in classical index-based notation

>>> y = np.transpose(x, (0, 2, 1))
>>> y.shape
(2, 4, 3)

we instead provide the input and output expressions in Einstein notation and let einx determine the necessary operations:

>>> y = einx.rearrange("a b c -> a c b", x)
>>> y.shape
(2, 4, 3)

The purpose of :func:`einx.rearrange` is to map tensors between different Einstein expressions. It does not perform any computation itself, but rather forwards the computation
to the respective backend, e.g. Numpy.

To verify that the correct backend calls are made, the just-in-time compiled function that einx invokes for this expression can be printed using ``graph=True``:

>>> graph = einx.rearrange("a b c -> a c b", x, graph=True)
>>> print(graph)
# backend: einx.backend.numpy
def op0(i0):
    x0 = backend.transpose(i0, (0, 2, 1))
    return x0

The function shows that einx performs the expected call to ``np.transpose``.

.. note::

    einx traces the backend calls made for a given operation and just-in-time compiles them into a regular Python function using Python's
    `exec() <https://docs.python.org/3/library/functions.html#exec>`_. When the function is called with the same signature of arguments, the compiled function is reused and
    therefore incurs no additional overhead other than for cache lookup (see :doc:`Just-in-time compilation </gettingstarted/jit>`)

.. _axiscomposition:

Axis composition
----------------

Multiple axes can be wrapped in parentheses to indicate that they represent an *axis composition*.

>>> x = np.ones((6, 4))
>>> einx.matches("(a b) c", x)
True

The composition ``(a b)`` is an axis itself and comprises the subaxes ``a`` and ``b`` which are layed out in
`row-major order <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_. This corresponds to ``a`` chunks of ``b`` elements each.
The length of the composed axis is the product of the subaxis lengths.

We can use :func:`einx.rearrange` to compose and decompose axes in a tensor by passing the respective Einstein expressions:

>>> # Stack 2 chunks of 3 elements into a single dimension with length 6
>>> x = np.ones((2, 3, 4))
>>> einx.rearrange("a b c -> (a b) c", x).shape
(6, 4)

>>> # Divide a dimension of length 6 into 2 chunks of 3 elements each
>>> x = np.ones((6, 4))
>>> einx.rearrange("(a b) c -> a b c", x, a=2).shape
(2, 3, 4)

Since the decomposition is ambiguous w.r.t. the values of ``a`` and ``b`` (for example ``a=2 b=3`` and ``a=1 b=6`` would be valid), additional constraints have to be passed
to find unique axis values, e.g. ``a=2`` as in the example above.

Composing and decomposing axes is a cheap operation and e.g. preferred over calling ``np.split``. The graph of these functions shows
that it uses a `np.reshape <https://numpy.org/doc/stable/reference/generated/numpy.reshape.html>`_
operation with the requested shape:

>>> print(einx.rearrange("(a b) c -> a b c", x, a=2, graph=True))
# backend: einx.backend.numpy
def op0(i0):
    x0 = backend.reshape(i0, (2, 3, 4))
    return x0

>>> print(einx.rearrange("a b c -> (a b) c", x, graph=True))
# backend: einx.backend.numpy
def op0(i0):
    x0 = backend.reshape(i0, (6, 4))
    return x0

.. note::

    See `this great einops tutorial <https://nbviewer.org/github/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb>`_ for hands-on illustrations of axis
    composition using a batch of images.

Axis compositions are used for example to divide the channels of a tensor into equally sized groups (as in multi-headed attention),
or to divide an image into patches by decomposing the spatial dimensions (if the image resolution is evenly divisible by the patch size).

Ellipsis
--------

An *ellipsis* repeats the expression that appears directly in front of it:

>>> x = np.ones((2, 3, 4))
>>> einx.matches("a b...", x) # Expands to "a b.0 b.1"
True

The number of repetitions is determined from the rank of the input tensors: 

>>> einx.rearrange("a b... -> b... a", x).shape # Expands to "a b.0 b.1 -> b.0 b.1 a"
(3, 4, 2)

Using ellipses e.g. for spatial dimensions often results in simpler and more readable expressions, and allows using the same expression for tensors with different dimensionality:

>>> # Divide an image into a list of patches with size p=8
>>> x = np.ones((256, 256, 3), dtype="uint8")
>>> einx.rearrange("(s p)... c -> (s...) p... c", x, p=8)
(1024, 8, 8, 3)

>>> # Divide a volume into a list of cubes with size p=8
>>> x = np.ones((256, 256, 256, 3), dtype="uint8")
>>> einx.rearrange("(s p)... c -> (s...) p... c", x, p=8)
(32768, 8, 8, 8, 3)

This operation requires multiple backend calls in index-based notation that might be difficult to understand on first glance. The einx call on the other hand clearly conveys
the intent of the operation and requires less code:

>>> print(einx.rearrange("(s p)... c -> (s...) p... c", x, p=8, graph=True))
# backend: einx.backend.numpy
def op0(i0):
    x2 = backend.reshape(i0, (32, 8, 32, 8, 3))
    x1 = backend.transpose(x2, (0, 2, 1, 3, 4))
    x0 = backend.reshape(x1, (1024, 8, 8, 3))
    return x0

In einops-style notation, an ellipsis can only appear once at root level without a preceding expression. To be fully compatible with einops notation, einx implicitly
converts anonymous ellipses by adding an axis in front:

..  code::

    einx.rearrange("b ... -> ... b", x)
    # same as
    einx.rearrange("b _anonymous_ellipsis_axis... -> _anonymous_ellipsis_axis... b", x)

Unnamed axes
------------

An *unnamed axis* is a number in the Einstein expression and similar to using a new unique axis name with an additional constraint specifying its length:

>>> x = np.ones((2, 3, 4))
>>> einx.matches("2 b c", x)
True
>>> einx.matches("a b c", x, a=2)
True
>>> einx.matches("a 1 c", x)
False

Unnamed axes can be used for example as an alternative to ``np.expand_dims``, ``np.squeeze``, ``np.newaxis``, ``np.broadcast_to``:

>>> x = np.ones((2, 1, 3))
>>> einx.rearrange("a 1 b -> 1 1 a b 1 5 6", x).shape
(1, 1, 2, 3, 1, 5, 6)

Since each unnamed axis is given a unique name, multiple unnamed axes do not refer to the same underlying tensor dimension. This can lead to unexpected behavior:

>>> einx.rearrange("a b c -> a c b", x).shape
(2, 4, 3)
>>> einx.rearrange("2 b c -> 2 c b", x).shape # Raises an exception

Concatenation
-------------

A *concatenation* represents an axis in Einstein notation along which two or more subtensors are concatenated. Using axis concatenations, we can describe operations such as
`np.concatenate <https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html>`_,
`np.split <https://numpy.org/doc/stable/reference/generated/numpy.split.html>`_,
`np.stack <https://numpy.org/doc/stable/reference/generated/numpy.stack.html>`_,
`einops.pack and einops.unpack <https://einops.rocks/4-pack-and-unpack/>`_ in pure Einstein notation. A concatenation axis is marked with ``+`` and wrapped in parentheses,
and its length is the sum of the subaxis lengths.

>>> x = np.ones((5, 4))
>>> einx.matches("(a + b) c", x)
True

This can be used for example to concatenate tensors that do not have compatible dimensions:

>>> x = np.ones((256, 256, 3))
>>> y = np.ones((256, 256))
>>> einx.rearrange("h w c, h w -> h w (c + 1)", x, y).shape
(256, 256, 4)

The graph shows that einx first reshapes ``y`` by adding a channel dimension, and then concatenates the tensors along that axis:

>>> print(einx.rearrange("h w c, h w -> h w (c + 1)", x, y, graph=True))
# backend: einx.backend.numpy
def op0(i0, i1):
    x1 = backend.reshape(i1, (256, 256, 1))
    x0 = backend.concatenate([i0, x1], 2)
    return x0

Splitting is supported analogously:

>>> z = np.ones((256, 256, 4))
>>> x, y = einx.rearrange("h w (c + 1) -> h w c, h w", z)
>>> x.shape, y.shape
((256, 256, 3), (256, 256))

Unlike the index-based `np.concatenate <https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html>`_, einx also broadcasts subtensors if required:

>>> # Append a number to all channels
>>> x = np.ones((256, 256, 3))
>>> einx.rearrange("... c, 1 -> ... (c + 1)", x, [42]).shape
(256, 256, 4)

Additional constraints
----------------------

einx uses a `SymPy <https://www.sympy.org/en/index.html>`_-based solver to determine the values of named axes in Einstein expressions (see :doc:`How does einx parse Einstein expressions? </faq/solver>`).
In many cases, the shapes of the input tensors provide enough constraints to determine the values of all named axes in the solver. For other cases, einx functions accept
``**parameters`` that can be used to specify the values of some or all named axes and provide additional constraints to the solver:

..  code::

    x = np.zeros((10,))
    einx.rearrange("(a b) -> a b", x)           # Fails: Values of a and b cannot be determined
    einx.rearrange("(a b) -> a b", x, a=5)      # Succeeds: b determined by solver
    einx.rearrange("(a b) -> a b", x, b=2)      # Succeeds: a determined by solver
    einx.rearrange("(a b) -> a b", x, a=5, b=2) # Succeeds
    einx.rearrange("(a b) -> a b", x, a=5, b=5) # Fails: Conflicting constraints

.. _bracketnotation:

Bracket notation
----------------

einx introduces the ``[]``-notation to denote axes that an operation is applied on. This corresponds to the ``axis`` argument in index-based notation:

..  code::

    einx.sum("a [b]", x)
    # same as
    np.sum(x, axis=1)

    einx.sum("a [...]", x)
    # same as
    np.sum(x, axis=tuple(range(1, x.ndim)))

:func:`einx.sum` is part of a family of functions that specialize :func:`einx.reduce` and apply a reduction operation to the input tensor
(see :doc:`Tutorial: Tensor manipulation </gettingstarted/tensormanipulation>`). In this case, ``[]`` denotes axes
that are reduced.

Bracket notation is fully compatible with expression rearranging and can therefore be placed anywhere inside a nested Einstein expression:

>>> # Compute sum over pairs of values along the last axis
>>> x = np.ones((2, 2, 16))
>>> einx.sum("... (g [c])", x, c=2).shape
(2, 2, 8)

>>> # Mean-pooling with stride 4 (if evenly divisible)
>>> x = np.ones((4, 256, 256, 3))
>>> einx.mean("b (s [ds])... c", x, ds=4).shape
(4, 64, 64, 3)

>>> print(einx.mean("b (s [ds])... c", x, ds=4, graph=True))
def reduce(i0, backend):
    x1 = backend.to_tensor(i0)
    x2 = backend.reshape(x1, (4, 64, 4, 64, 4, 3))
    x3 = backend.mean(x2, axis=(2, 4))
    return x3

See :doc:`How does einx handle input and output tensors? </faq/flatten>` for details on how operations are applied to tensors with nested Einstein expressions.

Operations are sensitive to the positioning of brackets, e.g. allowing for flexible ``keepdims=True`` behavior out-of-the-box:

>>> x = np.ones((16, 4))
>>> einx.sum("b [c]", x).shape
(16,)
>>> einx.sum("b ([c])", x).shape
(16, 1)
>>> einx.sum("b [c]", x, keepdims=True).shape
(16, 1)

In the second example, ``c`` is reduced within the composition ``(c)``, resulting in an empty composition ``()``, i.e. a trivial axis with size 1.

The operation :func:`einx.vmap` can be used to apply arbitrary functions to tensors. Analogous to the above examples, ``[]`` denotes axes that the function is applied on:

>>> x = np.ones((16, 8))
>>> def op(x): # c1 -> c2
>>>     return x[:-1]
>>> einx.vmap("b [c1] -> b [c2]", x, op=op, c2=7).shape
(16, 7)

.. note::

    :func:`einx.vmap` does not know the shape of the function output until the function is invoked, and thus requires specifying the additional constraint ``c2=7``.

The bracket notation also allows using a shorthand with ``[..|..]``-notation where two expressions are specified jointly:

..  code::

    einx.vmap("b [c1|c2]", x, op=op, c2=7)
    # same as
    einx.vmap("b [c1] -> b [c2]", x, op=op, c2=7)

The left and right options inside the bracket are selected for the input and output expressions, while all other parts are kept as-is. See the
documentation of the respective functions for more details on how bracket notation is used.

einx provides a wide range of tensor operations that accept arguments in Einstein notation as described in this document.
The following tutorial gives an overview of these functions and their usage.

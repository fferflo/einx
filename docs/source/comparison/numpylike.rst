Numpy-like notation
###################

Numpy and similar frameworks (*e.g.* PyTorch, Jax, Tensorflow) provide a large set of operation-specific rules
to express the vectorization of tensor operations. These rules include among others special function parameters (such as
``axis``, ``dim``, and ``keepdims``), `implicit broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__,
`advanced indexing rules <https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing>`__, and function-specific
behavior specified only in their documentation. einx subsumes these rules under a single, consistent notation.

In the following, we discuss several challenges of Numpy-like notation for expressing tensor operations, and how
these challenges are addressed with einx notation. We then provide a list of functions from Numpy-like frameworks that reduce to
a small set of einx operations with varying expressions for their vectorization.

Comparison of notations
***********************

High mental load
================

**Complex expressions in Numpy-like notation often incur mental load to read and write.** This is among others due to (1) the imperative rather than declarative
nature of the notation, and (2) the usage of positional axis indices rather than axis names. For example:

*   Which axes of ``x`` and ``y`` in the following expression are vectorized jointly or separately?

    ..  code-block:: python

        z = x[:, np.newaxis, :, np.newaxis] + y[:, :, np.newaxis, :]

    This information is more clearly conveyed in the equivalent einx expression:

    ..  code-block:: python

        z = einx.add("a c, a b d -> a b c d", x, y)

*   Which input and output axes in the following operation correspond to each other?

    ..  code-block:: python

        y = np.transpose(x, (2, 1, 3, 0))

    This information is more clearly conveyed in the equivalent einx expression:

    ..  code-block:: python

        y = einx.id("a b c d -> c b d a", x)

Separate shape alignment
========================

**Separate shape alignment is often required in Numpy-like notation to align input and output shapes of an operation with the operation's signature** (using *e.g.*
``np.newaxis``, ``np.reshape`` and ``np.transpose``). For example:

*   Scalar operations require shapes that align based on Numpy's implicit broadcasting rules. In the following case, we
    need to explicitly align the shapes by inserting unitary dimensions with ``np.newaxis`` first:

    ..  code-block:: python

        # x has shape (2, 4)
        # y has shape (2, 3, 5)

        # einx
        z = einx.add("a c, a b d -> a b c d")

        # Numpy
        z = x[:, np.newaxis, :, np.newaxis] + y[:, :, np.newaxis, :]

*   ``np.concatenate`` only works if the shapes of the tensors already match (up to the concatenated axis). This is not the case for example
    when appending a vector to an image tensor along the channel dimension:

    ..  code-block:: python

        # img has shape (4, 3, 64, 64) = (batch, channel, height, width)
        # vec has shape (2)

        # einx
        out = einx.id("b c1 h w, c2 -> b (c1 + c2) h w", img, vec)

        # Numpy
        vec_as_img = np.broadcast_to(vec[np.newaxis, :, np.newaxis, np.newaxis], (img.shape[0], vec.shape[0], img.shape[2], img.shape[3]))
        out = np.concatenate([img, vec_as_img], axis=1)

*   ``np.matmul`` supports only specific input and output shapes and requires separate alignment
    with ``np.newaxis`` and ``np.squeeze``:

    ..  code-block:: python

        # x has shape (2, 3)
        # y has shape (2, 3)

        # einx
        z = einx.dot("b [x], b [x] -> b", x, y)

        # Numpy
        z = np.matmul(x[:, np.newaxis, :], y[:, :, np.newaxis]).squeeze(-1).squeeze(-1)

*   ``np.broadcast_to`` requires source and target shapes that align based on Numpy's broadcasting rules. In the following case, we
    need to explicitly align the shapes by inserting unitary dimensions with ``np.newaxis`` first:

    ..  code-block:: python

        # x has shape (4)

        # einx
        y = einx.id("c -> 3 c 5", x)

        # Numpy
        y = np.broadcast_to(x[np.newaxis, :, np.newaxis], (3, 4, 5))

Non-descriptive functions
=========================

**Function names and arguments alone often do not reflect the vectorization behavior without
reading their documentation or writing comments.** For example:

*   All of ``torch.{take|gather|index_select}`` perform vectorized indexing. However, the functions apply to different input and output
    shapes, and the names do not indicate what the differences are:

    ..  code-block:: python

        z = torch.take(x, y) # Op 1
        z = torch.gather(x, 0, y) # Op 2
        z = torch.index_select(x, 0, y) # Op 3

    In contrast, einx provides a single function (*i.e.* ``einx.get_at``) for the gather/take/index-retrieval operation, with the vectorization behavior
    being represented explicitly in the vectorization expression:

    ..  code-block:: python

        z = einx.get_at("[x], ... -> ...", x, y) # Op 1
        z = einx.get_at("[x] b c, i b c -> i b c", x, y) # Op 2
        z = einx.get_at("[x] b c, i -> i b c", x, y) # Op 3

*   All of ``np.{matmul|dot|tensordot|inner}`` perform a vectorized dot-product. However, the functions follow different signatures: For instance,
    ``np.tensordot`` specifies the contracted axes using an ``axes`` argument, while the others implicitly choose contracted axes
    based on rules that are specified in their documentation. Of these functions, only ``np.matmul`` supports shared axes in the input tensors
    (such as in batched matrix multiplication or batched dot-products)

    .. code-block:: python

        z = np.matmul(x, y)
        z = einx.dot("b i [j], b [j] k -> b i k", x, y) # axis b is shared between both inputs

    while the others do not:

    .. code-block:: python

        z = np.dot(x, y)
        z = einx.dot("b1 i [j], b2 [j] k -> b1 i b2 k", x, y) # no axis is shared between inputs

        z = np.tensordot(x, y, axes=[2, 1])
        z = einx.dot("b1 i [j], b2 [j] k -> b1 i b2 k", x, y) # no axis is shared between inputs

        z = np.inner(x, y)
        z = einx.dot("b1 i [j], b2 k [j] -> b1 i b2 k", x, y) # no axis is shared between inputs

    The different functions names do not indicate the different behavior clearly. In contrast, einx provides a single function (*i.e.* ``einx.dot``)
    for the vectorized dot-product operation, with the vectorization behavior being represented explicitly in the vectorization expression.

*   ``np.kron`` and ``scipy.linalg.khatri_rao`` perform a vectorized scalar multiplication. However, the functions names do not communicate the behavior
    without reading their documentation (for anyone not familiar with the specific mathematical definitions). In contrast, the einx expression makes
    the vectorization behavior explicit:

    ..  code-block:: python

        z = np.kron(x, y)
        z = einx.multiply("a..., b... -> (a b)...", x, y)

        z = scipy.linalg.khatri_rao(x, y)
        z = einx.multiply("a c, b c -> (a b) c", x, y)

*   ``np.stack`` and ``np.concatenate`` (as well as variants such as ``np.{vstack|hstack|column_stack}``) perform concatenation-like operations.
    However, for anyone not familiar with the difference between "stacking" and "concatenating", the function names do not communicate the different behavior.
    In contrast, einx provides a single concept (*i.e.* axis compositions with ``+``) for such operations that clearly conveys the meaning:

    ..  code-block:: python

        z = np.stack([x, y], axis=-1)
        z = einx.id("a b, a b -> a b (1 + 1)", x, y)

        z = np.concatenate([x, y], axis=-1)
        z = einx.id("a b1, a b2 -> a (b1 + b2)", x, y)

*   ``np.meshgrid`` supports different indexing conventions (*i.e.* "ij" and "xy") that change the vectorization behavior.
    However, the function name does not communicate the difference without reading the documentation. In contrast, the einx expression
    makes the vectorization behavior explicit:

    ..  code-block:: python

        xn, yn = np.meshgrid(x, y, indexing="ij")
        xn, yn = einx.id("a, b -> a b, a b", x, y)

        xn, yn = np.meshgrid(x, y, indexing="xy")
        xn, yn = einx.id("a, b -> b a, b a", x, y)

Inconsistent behavior across frameworks
=======================================

**The rules for specifying how operations are vectorized sometimes differ across frameworks.** For example:

*   ``torch.gather`` and ``tf.gather`` both perform an indexing operation, but follow different vectorization behavior. This is
    indicated by listing the corresponding einx expression:

    ..  code-block:: python

        # PyTorch
        z = torch.gather(x, 0, y)
        z = einx.get_at("[x] b c, i b c -> i b c", x, y)

        # Tensorflow
        z = tf.gather(x, y, axis=0)
        z = einx.get_at("[x] b c, i -> i b c", x, y)

*   ``np.split`` and ``torch.split`` both perform a split/unconcatenate operation, but follow different vectorization behavior. This is
    indicated by listing the corresponding einx expression:

    ..  code-block:: python

        # Numpy
        x1, x2, x3 = np.split(x, [4, 9], axis=-1) # argument lists positions where split occurs
        x1, x2, x3 = einx.id("a (b1 + b2 + b3) -> a b1, a b2, a b3", x, b1=4, b2=5)

        # PyTorch
        x1, x2, x3 = torch.split(x, [4, 5, 6], dim=-1) # argument lists sizes of each split
        x1, x2, x3 = einx.id("a (b1 + b2 + b3) -> a b1, a b2, a b3", x, b1=4, b2=5)





List of operations in einx notation
***********************************

The following section provides examples of functions calls in Numpy-like frameworks and their equivalent einx expressions.
The functions are grouped based on the elementary operation they compute.

Vectorized identity map
=======================

np.reshape
----------

..  code-block:: python

    # x has shape (3, 4, 5)

    # Numpy
    y = np.reshape(x, (3, 4 * 5))

    # einx
    y = einx.id("a b c -> a (b c)", x)

..  code-block:: python

    # x has shape (3, 4 * 5)

    # Numpy
    y = np.reshape(x, (3, 4, 5))

    # einx
    y = einx.id("a (b c) -> a b c", x)

np.transpose
------------

..  code-block:: python

    # x has shape (3, 4, 5)

    # Numpy
    y = np.transpose(x, (0, 2, 1))

    # einx
    y = einx.id("a b c -> a c b", x)

np.squeeze
----------

..  code-block:: python

    # x has shape (3, 1, 5)

    # Numpy
    y = np.squeeze(x, axis=1)

    # einx
    y = einx.id("a 1 c -> a c", x)

np.expand_dims
--------------

..  code-block:: python

    # x has shape (3, 5)

    # Numpy
    y = np.expand_dims(x, axis=1)

    # einx
    y = einx.id("a c -> a 1 c", x)

np.newaxis
----------

..  code-block:: python

    # x has shape (3, 5)

    # Numpy
    y = x[:, np.newaxis, :]

    # einx
    y = einx.id("a c -> a 1 c", x)

np.broadcast_to
---------------

..  code-block:: python

    # x has shape (5)

    # Numpy
    y = np.broadcast_to(x, (3, 4, 5))

    # einx
    y = einx.id("c -> 3 4 c", x)

np.concatenate
--------------

..  code-block:: python

    # x has shape (3, 4)
    # y has shape (3, 5)

    # Numpy
    z = np.concatenate([x, y], axis=-1)

    # einx
    y = einx.id("a b1, a b2 -> a (b1 + b2)", x, y)

np.stack
--------

..  code-block:: python

    # x has shape (3, 4)
    # y has shape (3, 4)

    # Numpy
    z = np.stack([x, y], axis=-1)

    # einx
    z = einx.id("a b, a b -> a b (1 + 1)", x, y)

np.split
--------

..  code-block:: python

    # x has shape (3, 15)

    # Numpy
    x1, x2, x3 = np.split(x, [4, 9], axis=-1)

    # einx
    x1, x2, x3 = einx.id("a (b1 + b2 + b3) -> a b1, a b2, a b3", x, b1=4, b2=5)

torch.split
-----------

..  code-block:: python

    # x has shape (3, 15)

    # PyTorch
    x1, x2, x3 = torch.split(x, [4, 5, 6], dim=-1)

    # einx
    x1, x2, x3 = einx.id("a (b1 + b2 + b3) -> a b1, a b2, a b3", x, b1=4, b2=5)

np.meshgrid
-----------

..  code-block:: python

    # x has shape (3)
    # x has shape (4)

    # Numpy
    xn, yn = np.meshgrid(x, y, indexing="ij")

    # einx
    xn, yn = einx.id("a, b -> a b, a b", x, y)

..  code-block:: python

    # x has shape (3)
    # x has shape (4)

    # Numpy
    xn, yn = np.meshgrid(x, y, indexing="xy")

    # einx
    xn, yn = einx.id("a, b -> b a, b a", x, y)

np.zeros
--------

..  code-block:: python

    # Numpy
    x = np.zeros((3, 4, 5))

    # einx
    x = einx.id("-> a...", 0.0, a=(3, 4, 5))

np.ones
-------

..  code-block:: python

    # Numpy
    x = np.ones((3, 4, 5))

    # einx
    x = einx.id("-> a...", 1.0, a=(3, 4, 5))

np.full
-------

..  code-block:: python

    # Numpy
    x = np.full((3, 4, 5), 3.14)

    # einx
    x = einx.id("-> a...", 3.14, a=(3, 4, 5))

np.diagonal
-----------

..  code-block:: python

    # x has shape (3, 4, 4)

    # Numpy
    y = np.diagonal(x, axis1=1, axis2=2)

    # einx
    y = einx.id("a b b -> a b", x)

Vectorized dot-product
======================

np.dot
------

``np.dot`` performs a contraction over the two input tensors following a set of rules specified in its
`documentation <https://numpy.org/doc/stable/reference/generated/numpy.dot.html>`__. For instance, the
case described as

    If a is an N-D array and b is an M-D array (where M>=2), it is a sum product over the last axis of a and the second-to-last axis of b.

is expressed as follows:

..  code-block:: python

    # x has shape (3, 4)
    # y has shape (8, 4, 5)

    # Numpy
    z = np.dot(x, y)

    # einx
    z = einx.dot("x [a], y [a] b -> x y b", x, y)

np.matmul
---------

``np.matmul`` performs a contraction over the two input tensors following a set of rules specified in its
`documentation <https://numpy.org/doc/stable/reference/generated/numpy.matmul.html>`__. For instance, the
case described as

    If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.

is expressed as follows:

..  code-block:: python

    # x has shape (8, 3, 4)
    # y has shape (8, 4, 5)

    # Numpy
    z = np.matmul(x, y)

    # einx
    z = einx.dot("x a [b], x [b] c -> x a c", x, y)

np.inner
--------

``np.inner`` performs a contraction over the last axes of the two input tensors.

..  code-block:: python

    # x has shape (3, 4)
    # y has shape (5, 4)

    # Numpy
    z = np.inner(x, y)

    # einx
    z = einx.dot("a [c], b [c] -> a b", x, y)

np.tensordot
------------

``np.tensordot`` performs a contraction over the two input tensors along the positional axes specified in the ``axes`` argument.

..  code-block:: python

    # x has shape (4, 3)
    # y has shape (5, 4)

    # Numpy
    z = np.tensordot(x, y, axes=(0, 1))

    # einx
    z = einx.dot("[a] b, c [a] -> b c", x, y)

Vectorized scalar operations
============================

np.multiply
-----------

``np.multiply`` performs an element-wise multiplication of two tensors following Numpy's `implicit broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__.

..  code-block:: python

    # x has shape (3, 4)
    # y has shape (3)

    # Numpy
    z = np.multiply(x, y[:, np.newaxis])

    # einx
    z = einx.multiply("a b, a -> a b", x, y)

np.outer
--------

..  code-block:: python

    # x has shape (3)
    # y has shape (4)

    # Numpy
    z = np.outer(x, y)

    # einx
    z = einx.multiply("a, b -> a b", x, y)

np.kron
-------

``np.kron`` performs a Kronecker product of two tensors. The equivalent einx expression shows that it
is a vectorized scalar multiplication.

..  code-block:: python

    # x has shape (2, 3)
    # y has shape (4, 5)

    # Numpy
    z = np.kron(x, y)

    # einx
    z = einx.multiply("a..., b... -> (a b)...", x, y)

scipy.linalg.khatri_rao
-----------------------

``scipy.linalg.khatri_rao`` performs a Kathri-Rao product of two tensors. The equivalent einx expression shows that it
is a vectorized scalar multiplication.

..  code-block:: python

    # x has shape (2, 3)
    # y has shape (4, 3)

    # Scipy
    z = scipy.linalg.khatri_rao(x, y)

    # einx
    z = einx.multiply("a c, b c -> (a b) c", x, y)

Vectorized indexing operations
==============================

torch.take
----------

..  code-block:: python

    # x has shape (16)
    # y has shape (3, 4)

    # Torch
    z = torch.take(x, y)

    # einx
    z = einx.get_at("[x], ... -> ...", x, y)

torch.gather
------------

..  code-block:: python

    # x has shape (16, 4, 5)
    # y has shape (3, 4, 5)

    # PyTorch
    z = torch.gather(x, 0, y)

    # einx
    z = einx.get_at("[x] b c, i b c -> i b c", x, y)

torch.take_along_dim
--------------------

..  code-block:: python

    # x has shape (16, 4, 5)
    # y has shape (3, 4, 5)

    # PyTorch
    z = torch.take_along_dim(x, y, dim=0)

    # einx
    z = einx.get_at("[x] b c, i b c -> i b c", x, y)

torch.index_select
------------------

..  code-block:: python

    # x has shape (3, 16, 5)
    # y has shape (4)

    # PyTorch
    z = torch.index_select(x, 1, y)

    # einx
    z = einx.get_at("a [x] c, i -> a i c", x, y)

tf.gather
---------

..  code-block:: python

    # x has shape (3, 16, 5)
    # y has shape (4)

    # Tensorflow
    z = tf.gather(x, y, axis=1)

    # einx
    z = einx.get_at("a [x] c, i -> a i c", x, y)

tf.gather_nd
------------

..  code-block:: python

    # x has shape (16, 17)
    # y has shape (4, 2)

    # Tensorflow
    z = tf.gather_nd(x, y)

    # einx
    z = einx.get_at("[...], b [i] -> b", x, y)

..  code-block:: python

    # x has shape (5, 16, 17)
    # y has shape (5, 4, 2)

    # Tensorflow
    z = tf.gather_nd(x, y, batch_dims=1)

    # einx
    z = einx.get_at("a [...], a b [i] -> a b", x, y)

__getitem__
-----------

..  code-block:: python

    # x has shape (16, 17)
    # y1 has shape (4)
    # y2 has shape (4)

    # Numpy
    z = x[y1, y2]

    # einx
    z = einx.get_at("[x y], a, a -> a", x, y1, y2)

..  code-block:: python

    # x has shape (16, 17)
    # y has shape (4, 2)

    # Numpy
    z = x[y[:, 0], y[:, 1]]

    # einx
    z = einx.get_at("[x y], a [2] -> a", x, y)

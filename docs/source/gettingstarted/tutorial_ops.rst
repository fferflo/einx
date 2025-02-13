Tutorial: Operations
####################

einx represents tensor operations using a set of elementary operations that are vectorized according to the given einx expressions.
Internally, einx does not implement the operations from scratch, but forwards computation to the respective backend, e.g. by
calling `np.reshape <https://numpy.org/doc/stable/reference/generated/numpy.reshape.html>`_,
`np.transpose <https://numpy.org/doc/stable/reference/generated/numpy.transpose.html>`_ or 
`np.sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_ with the appropriate arguments.

This tutorial gives an overview of these operations and their usage. For a complete list of provided functions, see the :doc:`API reference </api>`.

Rearranging
-----------

The function :func:`einx.rearrange` transforms tensors between einx expressions without modifying their values. Conceptually,
:func:`einx.rearrange` applies a vectorized identity mapping. It is implemented on the backend using operations such as
reshaping, permuting, stacking, unstacking and broadcasting.

For example:

>>> x = np.ones((4, 256, 17))
>>> y, z = einx.rearrange("b (s p) (c + 1) -> (b s) p c, (b p) s 1", x, p=8)
>>> y.shape, z.shape
((128, 8, 16), (32, 32, 1))

The index-based calls can be inspected using the just-in-time compiled function that einx creates for this
expression (see :doc:`Just-in-time compilation </more/jit>`):

>>> print(einx.rearrange("b (s p) (c + 1) -> (b s) p c, (b p) s 1", x, p=8, graph=True))
import numpy as np
def op0(i0):
    x0 = np.reshape(i0, (4, 32, 8, 17))
    x1 = np.reshape(x0[:, :, :, 0:16], (128, 8, 16))
    x2 = np.reshape(x0[:, :, :, 16:17], (4, 32, 8))
    x3 = np.transpose(x2, (0, 2, 1))
    x4 = np.reshape(x3, (32, 32, 1))
    return [x1, x4]

Reduction
---------

einx provides a :ref:`family of elementary operations <apireductionops>` that reduce tensors along one or more axes. For example:

.. code::

   einx.sum("a [b] -> a", x)
   # same as
   np.sum(x, axis=1)

   einx.mean("a [...] c -> a c", x)
   # same as
   np.mean(x, axis=tuple(range(1, x.ndim - 1)))

The elementary operations of these functions have the signature ``[...] -> []``; they take an input tensor with arbitrary shape and return a scalar.
The operation is vectorized over all other axes in the expressions.

Reduction operations support the following variants:

* If no brackets are found, brackets are placed implicitly around all axes that do not appear in the output:

  .. code::

     einx.sum("a b c -> a c", x) # Expands to: "a [b] c -> a c"

  This corresponds with `einops-notation for reduction operations <https://einops.rocks/api/reduce/>`_.

* If no output expression is specified, it is determined implicitly by removing marked subexpressions from the input:

  ..  code::

     einx.sum("a [b] c", x) # Expands to: "a [b] c -> a c"

All operations support expression rearranging:

>>> x = np.ones((16, 8))
>>> einx.prod("a (b [c]) -> b a", x, c=2).shape
(4, 16)

Reduction functions are implemented as specializations of :func:`einx.reduce` and use backend operations like `np.sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_,
`np.prod <https://numpy.org/doc/stable/reference/generated/numpy.prod.html>`_ or `np.any <https://numpy.org/doc/stable/reference/generated/numpy.any.html>`_ as the ``op`` argument:

.. code::

   einx.reduce("a [b]", x, op=np.sum)
   # same as
   einx.sum("a [b]", x)

:func:`einx.reduce` also allows custom reduction operations that accept the ``axis`` argument similar to `np.sum <https://numpy.org/doc/stable/reference/generated/numpy.sum.html>`_:

.. code::

   def custom_mean(x, axis):
       return np.sum(x, axis=axis) / x.shape[axis]
   einx.reduce("a [b] c", x, op=custom_mean)

Element-by-element
------------------

einx provides a :ref:`family of elementary operations <apielementwiseops>` that apply element-by-element operations to tensors. For example:

.. code::

   einx.add("a b, b -> a b", x, y)
   # same as
   x + y[np.newaxis, :]

   einx.multiply("a, a b -> a b", x, y)
   # same as
   x[:, np.newaxis] * y

   einx.subtract("a, (a b) -> b a", x, y)
   # requires reshape and transpose in index-based notation

The elementary operations of these functions have the signature ``[], [], ... -> []``; they take one or more scalars as input and return a single scalar as output.
The operation is vectorized over all axes in the expressions.

Element-by-element operations support the following variants:

* If the output is left out, it is determined implicitly if one of the input expressions contains the named axes of all other inputs and if this choice is unique:

  .. code::

     einx.add("a b, a", x, y)         # Expands to: "a b, a -> a b"

     einx.where("b a, b, a", x, y, z) # Expands to "b a, b, a -> b a"

     einx.subtract("a b, b a", x, y)  # Raises an exception

     einx.add("a b, a b", x, y)       # Expands to: "a b, a b -> a b"

* Bracket notation can be used to indicate that the second input is a subexpression of the first:

  .. code::

     einx.add("a [b]", x, y) # Expands to: "a b, b"

  .. note::

     Conceptually, a different elementary operation is used in this case which is applied to tensors of equal shape rather than just scalars.
     This variant will likely be removed in future versions.

All operations support expression rearranging:

>>> x = np.ones((16, 16, 32))
>>> y = np.ones((4,))
>>> einx.add("... (g c), c -> ... (c g)", x, y).shape
(16, 16, 32)

Internally, the inputs are rearranged such that the operation can be applied using `Numpy broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.
The functions are specializations of :func:`einx.elementwise` and use backend operations like `np.add <https://numpy.org/doc/stable/reference/generated/numpy.add.html>`_,
`np.logical_and <https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html>`_ and `np.where <https://numpy.org/doc/stable/reference/generated/numpy.where.html>`_
as the ``op`` argument:

.. code::

   einx.elementwise("a b, b -> a b", x, y, op=np.add)
   # same as
   einx.add("a b, b -> a b", x, y)

Several operations from classical tensor frameworks can be expressed using universal einx element-by-element functions:

.. list-table:: Example: ``einx.multiply``
   :widths: 42 58
   :header-rows: 1

   * - Classical API
     - einx API

   * - | ``np.multiply(x, y[:, np.newaxis])``
       | ``x * y[:, np.newaxis]``
     - ``einx.multiply("a b, a -> a b", x, y)``
   * - ``np.outer(x, y)``
     - ``einx.multiply("a, b -> a b", x, y)``
   * - ``np.kron(x, y)``
     - ``einx.multiply("a..., b... -> (a b)...", x, y),``
   * - ``scipy.linalg.khatri_rao(x, y)``
     - ``einx.multiply("a c, b c -> (a b) c", x, y)``

Indexing
--------

einx provides a :ref:`family of elementary operations <apiindexingops>` that perform multi-dimensional indexing and update/retrieve values from tensors
at specified coordinates. For example:

.. code::

   image = np.ones((256, 256, 3))
   coordinates = np.ones((100, 2), dtype=np.int32)
   updates = np.ones((100, 3))

   # Retrieve values at the given locations in an image
   y = einx.get_at("[h w] c, i [2] -> i c", image, coordinates)
   # same as
   y = image[coordinates[:, 0], coordinates[:, 1]]

The elementary retrieval operation has the signature ``[...], [i] -> []``; brackets in the first input indicate axes that are indexed,
and a single bracket in the second input indicates the coordinate axis. The output is a single scalar, i.e. the value that is retrieved from the given coordinates.
The length of the coordinate axis should equal the number of indexed axes in the first input. The operation is vectorized over all other axes in the expressions.

When updating values in the tensor, the elementary operation has the signature ``[...], [i], [] -> [...]``; it includes an additional input scalar that is used
to update the given tensor, and the output shape matches the shape of the first input. For example:

.. code::

   updates = np.ones((100, 3))

   # Update values at the given locations in an image
   y = einx.set_at("[h w] c, i [2], i c -> [h w] c", image, coordinates, updates)
   # same as
   image[coordinates[:, 0], coordinates[:, 1]] = updates
   y = image

Indexing operations support the following variants:

* For single-dimensional indexing, the elementary operation supports the signatures ``[d], [1] -> []`` and ``[d], [] -> []``:

  .. code::

     tensor = np.ones((256, 3))
     coordinates_with1 = np.ones((100, 1), dtype=np.int32)
     coordinates_without1 = coordinates_with1[:, 0] # (100,)

     y = einx.get_at("[d] c, i [1] -> i c", tensor, coordinates_with1)
     # same as
     y = einx.get_at("[d] c, i -> i c", tensor, coordinates_without1)
     # same as 
     y = einx.get_at("[d] c, i 1 -> i c", tensor, coordinates_with1)
     # -> uses the signature [d], [] -> [] and vectorizes over the 1 axis

* Coordinates can be passed in as separate tensors and are stacked to form the coordinate tensor:

  .. code::

     coordinates_x = np.ones((100,), dtype=np.int32)
     coordinates_y = np.ones((100,), dtype=np.int32)

     y = einx.get_at("[h w] c, i, i -> i c", image, coordinates_x, coordinates_y)

All operations support expression rearranging:

.. code::

   einx.add_at("b ([h w]) c, ([2] b) i, c i -> c [h w] b", image, coordinates, updates, h=256)

Several operations from classical tensor frameworks can be expressed using universal einx indexing functions:

.. list-table:: Example: ``einx.get_at``
   :widths: 42 58
   :header-rows: 1

   * - Classical API
     - einx API

   * - | ``torch.gather(x, 0, y)``
       | ``torch.take_along_dim(x, y, dim=0)``
     - ``einx.get_at("[_] b c, i b c -> i b c", x, y)``
   * - | ``torch.gather(x, 1, y)``
       | ``torch.take_along_dim(x, y, dim=1)``
     - ``einx.get_at("a [_] c, a i c -> a i c", x, y)``
   * - | ``torch.index_select(x, 0, y)``
       | ``tf.gather(x, y, axis=0)``
     - ``einx.get_at("[_] b c, i -> i b c", x, y)``
   * - | ``torch.index_select(x, 1, y)``
       | ``tf.gather(x, y, axis=1)``
     - ``einx.get_at("a [_] c, i -> a i c", x, y)``
   * - ``tf.gather(x, y, axis=1, batch_dims=1)``
     - ``einx.get_at("a [_] c, a i -> a i c", x, y)``
   * - ``torch.take(x, y)``
     - ``einx.get_at("[_], ... -> ...", x, y)``
   * - ``tf.gather_nd(x, y)``
     - ``einx.get_at("[...], b [i] -> b", x, y)``
   * - ``tf.gather_nd(x, y, batch_dims=1)``
     - ``einx.get_at("a [...], a b [i] -> a b", x, y)``

Dot-product
-----------

The function :func:`einx.dot` computes a dot-product along the marked axes:

>>> # Matrix multiplication between x and y
>>> x = np.ones((4, 16))
>>> y = np.ones((16, 8))
>>> einx.dot("a [b], [b] c -> a c", x, y).shape
(4, 8)

The elementary operation of this functions has the signature ``[...], [...], ... -> []``; it takes several tensors as input and reduces to a single scalar value.

While operations such as matrix multiplications are represented conceptually as a vectorized dot-products in einx, they are still implemented using
efficient matmul calls in the respective backend rather than a vectorized evaluation of the dot-product.

The operation supports the following variants:

* If no brackets are found, brackets are placed implicitly around all axes that do not appear in the output:

  .. code::

     einx.dot("a b, b c -> a c", x, y) # Expands to: "a [b], [b] c -> a c"

  This allows using einsum-like notation with :func:`einx.dot`. In fact, :func:`einx.dot` internally forwards computation
  to the ``einsum`` implementation of the respective backend, but additionally supports rearranging of expressions:

  >>> # Simple grouped linear layer
  >>> x = np.ones((20, 16))
  >>> w = np.ones((8, 4))
  >>> print(einx.dot("b (g c1), c1 c2 -> b (g c2)", x, w, g=2, graph=True))
  import numpy as np
  def op0(i0, i1):
      x0 = np.reshape(i0, (20, 2, 8))
      x1 = np.einsum("abc,cd->abd", x0, i1)
      x2 = np.reshape(x1, (20, 8))
      return x2

* If given two input tensors, the expression of the second input is determined implicitly by marking
  its components in the input and output expression:

  .. code::

     einx.dot("a [b] -> a [c]", x, y) # Expands to: "a b, b c -> a c"

  .. note::

     Conceptually, the elementary operation in this case is not a simple dot-product, but rather a linear map from
     ``b`` to ``c`` channels. This variant will likely be removed in future versions.

  Axes marked multiple times appear only once in the implicit second input expression:

  .. code::

     einx.dot("[a b] -> [a c]", x, y) # Expands to: "a b, a b c -> a c"

Several operations from classical tensor frameworks can be expressed using the einx dot-product (and similarly using einsum):

.. list-table:: Example: ``einx.dot``
   :widths: 42 58
   :header-rows: 1

   * - Classical API
     - einx API

   * - ``np.matmul(x, y)``
     - | ``einx.dot("... a [b], ... [b] c -> ... a c", x, y)``
       | ``einx.dot("... [a], [a] -> ...", x, y)``
   * - ``np.dot(x, y)``
     - | ``einx.dot("x... [a], y... [a] b -> x... y... b", x, y)``
       | ``einx.dot("... [a], [a] -> ...", x, y)``
   * - ``np.tensordot(x, y, axes=1)``
     - ``einx.dot("a [b], [b] c -> a c", x, y)``
   * - ``np.tensordot(x, y, axes=([2], [1]))``
     - ``einx.dot("a b [c], d [c] e -> a b d e", x, y)``
   * - ``np.inner(x, y)``
     - ``einx.dot("x... [a], y... [a] -> x... y...", x, y)``

Other operations: ``vmap``
--------------------------

If an operation is not provided as a separate einx API, it can still be applied in einx using :func:`einx.vmap` or :func:`einx.vmap_with_axis`.
Both functions apply the same vectorization rules as other einx functions, but accept an ``op`` argument that specifies the elementary operation to apply.

In :func:`einx.vmap`, the input and output tensors of ``op`` match the marked axes in the input and output expressions:

.. code::

   # A custom operation:
   def op(x):
       # Input: x has shape "b c"
       x = np.sum(x, axis=1)
       x = np.flip(x, axis=0)
       # Output: x has shape "b"
       return x

   einx.vmap("a [b c] -> a [b]", x, op=op)

:func:`einx.vmap` is implemented using automatic vectorization in the respective backend (e.g. 
`jax.vmap <https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html>`_, `torch.vmap <https://pytorch.org/docs/stable/generated/torch.vmap.html>`_). 
einx also implements a simple ``vmap`` function for the Numpy backend for testing/ debugging purposes using Python loops.

In :func:`einx.vmap_with_axis`, ``op`` is instead given an ``axis`` argument and must follow
`Numpy broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_:

.. code::

   # A custom operation:
   def op(x, axis):
       # Input: x has shape "a b c", axis is (1, 2)
       x = np.sum(x, axis=axis[1])
       x = np.flip(x, axis=axis[0])
       # Output: x has shape "a b"
       return x

   einx.vmap_with_axis("(a [b c]) -> (a [b])", x, op=op, a=2, b=3, c=4)

Both :func:`einx.reduce` and :func:`einx.elementwise` are adaptations of :func:`einx.vmap_with_axis`.

Since most backend operations that accept an ``axis`` argument operate on the entire input tensor when ``axis`` is not given, :func:`einx.vmap_with_axis` can often
analogously be expressed using :func:`einx.vmap`:

>>> x = np.ones((4, 16))
>>> einx.vmap_with_axis("a [b] -> a", x, op=np.sum).shape
(4,)
>>> einx.vmap          ("a [b] -> a", x, op=np.sum).shape
(4,)

>>> x = np.ones((4, 16))
>>> y = np.ones((4,))
>>> einx.vmap_with_axis("a b, a -> a b", x, y, op=np.add).shape
(4, 16)
>>> einx.vmap          ("a b, a -> a b", x, y, op=np.add).shape
(4, 16)

:func:`einx.vmap` provides more general vectorization capabilities than :func:`einx.vmap_with_axis`, but might in some cases be slower if the latter relies on a
specialized implementation.

.. _lazytensorconstruction:

Misc: Tensor factories
----------------------------

All einx operations also accept tensor factories instead of tensors as arguments. A tensor factory is a function that accepts a ``shape``
argument and returns a tensor with that shape. This allows deferring the construction of a tensor to the point inside
an einx operation where its shape has been resolved, and avoids having to manually determine the shape in advance:

..  code::

    einx.dot("b... c1, c1 c2 -> b... c2", x, lambda shape: np.random.uniform(shape), c2=32)

In this example, the shape of ``x`` is used by the expression solver to determine the values of ``b...`` and ``c1``. Since the tensor factory provides no shape
constraints to the solver, the remaining axis values have to be specified explicitly, i.e. ``c2=32``.

Tensor factories are particularly useful in the context of deep learning modules: The shapes of a layer's weights are typically chosen to align with the shapes
of the layer input and outputs (e.g. the number of input channels in a linear layer must match the corresponding axis in the layer's weight matrix).
This can be achieved implicitly by constructing layer weights using tensor factories.

The following tutorial describes in more detail how this is used in einx to implement deep learning models.
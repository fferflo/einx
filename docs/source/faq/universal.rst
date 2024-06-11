How is einx notation universal?
###############################

To address this question, let's first look at how tensor operations are commonly expressed in existing tensor frameworks.

Classical notation
------------------

Tensor operations can be dissected into two distinct components:

1. An **elementary operation** that is performed.

   * Example: ``np.sum`` computes a sum-reduction.

2. A division of the input tensor into sub-tensors. The elementary operation is applied to each sub-tensor independently. We refer to this as **vectorization**.

   * Example: Sub-tensors in ``np.sum`` span the dimensions specified by the ``axis`` parameter. The sum-reduction is vectorized over all other dimensions.

In common tensor frameworks like Numpy, PyTorch, Tensorflow or Jax, different elementary operations are implemented with different vectorization rules.
For example, to express vectorization

* ``np.sum`` uses the ``axis`` parameter,
* ``np.add`` follows `implicit broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ (e.g. in combination with ``np.newaxis``), and
* ``np.matmul`` provides `an implicit and custom set of rules <https://numpy.org/doc/stable/reference/generated/numpy.matmul.html>`_.

Furthermore, an elementary operation is sometimes implemented in multiple APIs in order to offer vectorization rules for different use cases.
For example, the retrieve-at-index operation can be implemented in PyTorch using ``tensor[coords]``, ``torch.gather``, ``torch.index_select``, ``torch.take``,
``torch.take_along_dim``, which conceptually apply the same low-level operation, but follow different vectorization rules (see below).
Still, these interfaces sometimes do not cover all desirable use cases.

einx notation
-------------

einx provides an interface to tensor operations where vectorization is expressed entirely using einx notation, and each elementary operation
is represented by exactly one API. The einx notation is:

* **Consistent**: The same type of notation is used for all elementary operations. Each elementary operation is represented by exactly one API.
* **Complete**: Any operation that can be expressed with existing vectorization tools such as
  `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_ can also be expressed in einx notation.

The following tables show examples of classical API calls that can be expressed using universal einx operations.

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
   * - | ``tf.gather_nd(x, y, batch_dims=1)``
       | ``x[y[..., 0], y[..., 1]]``
     - ``einx.get_at("a [...], a b [i] -> a b", x, y)``

.. list-table:: Example: ``einx.dot`` (similar to einsum)
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

.. list-table:: Example: ``einx.flip``
   :widths: 42 58
   :header-rows: 1

   * - Classical API
     - einx API

   * - | ``np.flip(x, y, axis=0)``
       | ``np.flipud(x, y)``
     - ``einx.flip("[a] b", x)``
   * - ``np.fliplr(x, y)``
     - ``einx.flip("a [b]", x)``

..
   * - ``einx.rearrange``
     - ``np.reshape`` ``np.transpose`` ``np.squeeze`` ``np.expand_dims`` ``tensor[np.newaxis]`` ``np.stack`` ``np.hstack`` ``np.concatenate``

While elementary operations and vectorization are decoupled conceptually to provide a universal API, the implementation of the operations
in the respective backend do not necessarily follow the same decoupling. For example, a matrix multiplication is represented as a vectorized
dot-product in einx (using ``einx.dot``), but still invokes an efficient matmul operation on the backend instead of a vectorized evaluation of the dot product.
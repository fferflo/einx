How is einx notation universal?
###############################

To address this question, let's first look at how tensor operations are commonly expressed in existing tensor frameworks.

Classical notation
------------------

Tensor operations can be dissected into two distinct components:

1. An **elementary operation** that is performed.

   * Example: ``np.sum`` computes a sum-reduction.

2. A division of the input tensor into sub-tensors. The elementary operation is applied to each sub-tensor independently. We refer to this as **vectorization**.

   * Example: Sub-tensors in ``np.sum`` span the dimensions specified by the ``axis`` parameter. The sum-reduction is repeated over all other dimensions.

In common tensor frameworks like Numpy, PyTorch, Tensorflow or Jax, different elementary operations are implemented with different vectorization rules.
For example, to express vectorization

* ``np.sum`` uses the ``axis`` parameter,
* ``np.add`` follows `implicit broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_ (e.g. in combination with ``np.newaxis``), and
* ``np.matmul`` provides `an implicit and custom set of rules <https://numpy.org/doc/stable/reference/generated/numpy.matmul.html>`_.

Furthermore, an elementary operation is sometimes implemented in multiple APIs in order to offer vectorization rules for different use cases.
For example, the retrieve-at-index operation can be implemented in PyTorch using ``tensor[coords]``, ``torch.gather``, ``torch.index_select``, ``torch.take``,
``torch.take_along_dim``, which conceptually apply the same low-level operation, but follow different vectorization rules.
Still, these interfaces sometimes do not cover all desirable use cases.

einx notation
-------------

einx provides an interface to tensor operations where vectorization is expressed entirely using einx notation, and each elementary operation
is represented by exactly one API. The einx notation is:

* **Consistent**: The same type of notation is used for all elementary operations.
* **Unique**: Each elementary operation is represented by exactly one API.
* **Complete**: Any operation that can be expressed with existing vectorization tools such as
  `jax.vmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html>`_ can also be expressed in einx notation.

The following table shows examples of universal einx functions that implement the same elementary operations as a variety of existing tensor operations:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - einx API
     - Classical API
   * - ``einx.get_at``
     - ``torch.gather`` ``torch.index_select`` ``torch.take`` ``torch.take_along_dim`` ``tf.gather`` ``tf.gather_nd`` ``tensor[coords]``
   * - ``einx.dot`` (similar to einsum)
     - ``np.matmul`` ``np.dot`` ``np.tensordot`` ``np.inner``
   * - ``einx.add``
     - ``np.add`` with ``np.newaxis``
   * - ``einx.rearrange``
     - ``np.reshape`` ``np.transpose`` ``np.squeeze`` ``np.expand_dims`` ``tensor[np.newaxis]`` ``np.stack`` ``np.hstack`` ``np.concatenate``
   * - ``einx.flip``
     - ``np.flip`` ``np.fliplr`` ``np.flipud``

While elementary operations and vectorization are decoupled conceptually to provide a universal API, the implementation of the operations
in the respective backend do not necessarily follow the same decoupling. For example, a matrix multiplication is represented as a vectorized
dot-product in einx (using ``einx.dot``), but still invokes an efficient matmul operation instead of a vectorized evaluation of the dot product.
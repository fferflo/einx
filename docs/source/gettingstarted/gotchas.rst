Gotchas
#######

1. **Unnamed axes are always unique** and cannot refer to the same axis in different expressions. E.g. ``3 -> 3`` refers to two different axes, both
with length 3. This can lead to unexpected behavior in some cases: ``einx.reduce("3 -> 3", x)`` will reduce the first ``3`` axis and insert
a new axis broadcasted to length 3.

2. **Spaces in expressions are important.** E.g. in ``(a b)...`` the ellipsis repeats ``(a b)``, while in ``(a b) ...``  the ellipsis repeats a new
axis that is inserted in front of it.

3. **einx.dot is not called einx.einsum** despite providing einsum-like functionality to avoid confusion with ``einx.sum``. The name is 
motivated by the fact that the function computes a generalized dot-product, and is in line with expressing the same operation using :func:`einx.vmap`:

..  code::

    einx.dot("a b, b c -> a c", x, y)
    einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot)

4. **Neural network layers in** ``einx.nn.*`` **have to be initialized with a single forward-pass on a dummy batch** to determine the shapes and construct the layer weights.
This is already common practice in jax-based frameworks like `Flax <https://github.com/google/flax>`_ and `Haiku <https://github.com/google-deepmind/dm-haiku>`_,
but may require modification of `PyTorch <https://pytorch.org/>`_ training scripts. ``torch.compile`` should be applied after this
first forward pass (see :doc:`Neural networks </gettingstarted/neuralnetworks>`).

5. **einx does not support dynamic shapes** that can occur for example when tracing some types of functions
(e.g. `tf.unique <https://www.tensorflow.org/api_docs/python/tf/unique>`_) in Tensorflow using ``tf.function``. As a workaround, the shape can be specified statically
using `tf.ensure_shape <https://www.tensorflow.org/api_docs/python/tf/ensure_shape>`_.

6. **einx implements a custom vmap for Numpy using Python loops**. This is slower than ``vmap``
in other backends, but is included for debugging and testing purposes.
Gotchas
#######

1. **Unnamed axes are always unique** and cannot refer to the same axis in different expressions. E.g. ``3 -> 3`` refers to two different axes, both
with length 3. This can lead to unexpected behavior in some cases: ``einx.sum("3 -> 3", x)`` will reduce the first ``3`` axis and insert
a new axis broadcasted to length 3.

2. **Spaces in expressions are important.** E.g. in ``(a b)...`` the ellipsis repeats ``(a b)``, while in ``(a b) ...``  the ellipsis repeats a new
axis that is inserted in front of it.

3. **einx.dot is not called einx.einsum** despite providing einsum-like functionality to avoid confusion with ``einx.sum``. The name is 
motivated by the fact that the function computes a generalized dot-product, and is in line with expressing the same operation using :func:`einx.vmap`:

..  code::

    einx.dot("a b, b c -> a c", x, y)
    einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot)

4. **einx does not support dynamic shapes** that can occur for example when tracing some types of functions
(e.g. `tf.unique <https://www.tensorflow.org/api_docs/python/tf/unique>`_) in Tensorflow using ``tf.function``. As a workaround, the shape can be specified statically,
e.g. using `tf.ensure_shape <https://www.tensorflow.org/api_docs/python/tf/ensure_shape>`_. In Keras, when constructing a model using the
`functional API <https://keras.io/guides/functional_api/>`_, the batch size argument is dynamic by default and should be specified with some dummy value,
e.g. ``keras.Input(shape=(...), batch_size=1)``.

5. **einx implements a custom vmap for Numpy using Python loops**. This is slower than ``vmap``
in other backends, but is included for debugging and testing purposes.

6. **In einx.nn layers, weights are created on the first forward pass** (see :doc:`Tutorial: Neural networks </gettingstarted/neuralnetworks>`). This is common practice in jax-based frameworks like Flax and Haiku where the
model is initialized using a forward pass on a dummy batch. In other frameworks, an initial forward pass should be added before using the model. (In some
circumstances the first actual training batch might be sufficient, but it is safer to always include the initial forward pass.) In PyTorch,
`torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_ should only be applied after the initial forward pass.

7. **einx.nn.equinox does not support stateful layers** since Equinox requires the shape of states to be known in the layer's ``__init__``
method.

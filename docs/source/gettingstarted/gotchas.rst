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

4. **Compatibility of torch.compile and einx** is not tested well and may cause silent bugs. When
`torch.compile <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_ is used with ``einn`` modules, it raises an exception unless the module
is first invoked with a dummy batch (possibly due to a buggy interaction with tracing and caching of einx operations and instantiation of weights, although later calls where
a cache miss occurs and a graph is traced do not cause the same error). It is not clear at this point how ``torch.compile`` interacts with einx. ``torch.compile`` should be used at
your own discretion. einx can be used with PyTorch in eager mode and adds only negligible overhead due to the tracing and caching of operations. Feedback on this issue is welcome.

5. **einx does not support dynamic shapes** that can occur for example when tracing some types of functions
(e.g. `tf.unique <https://www.tensorflow.org/api_docs/python/tf/unique>`_) in Tensorflow using ``tf.function``. As a workaround, the shape can be specified statically,
e.g. using `tf.ensure_shape <https://www.tensorflow.org/api_docs/python/tf/ensure_shape>`_.

6. **einx implements a custom vmap for Numpy using Python loops**. This is slower than ``vmap``
in other backends, but is included for debugging and testing purposes.
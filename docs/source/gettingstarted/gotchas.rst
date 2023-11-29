Gotchas
#######

1. **Unnamed axes are always unique** and cannot refer to the same axis in different expressions. E.g. ``3 -> 3`` refers to two different axes, both
with length 3. This can lead to unexpected behavior in some cases: ``einx.reduce("3 -> 3", x)`` will reduce the first ``3`` axis and insert
a new axis broadcasted to length 3.

2. **Spaces in expressions are important.** E.g. in ``(a b)...`` the ellipsis repeats ``(a b)``, while in ``(a b) ...``  the ellipsis repeats a new
axis that is inserted in front of it.

3. While ``jax.jit`` is fully compatible with einx, ``torch.compile`` causes problems form some functionality in ``einx.nn.torch.*``, for example
when using ``decay_rate`` argument in ``einx.nn.torch.Norm``. This might be fixed in the future.

4. **einx.dot is not called einx.einsum** despite providing einsum-like functionality to avoid confusion with ``einx.sum``. The name is 
motivated by the fact that the function computes a generalized dot-product, and is in line with expressing the same operation using ``einx.vmap``:

..  code::

    einx.dot("a b, b c -> a c", x, y)
    einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot)
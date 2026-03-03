vmap notation
#############

Short introduction
******************

vmap notation is based on the vmap function which provides a universal means to vectorize operations along a single axis of input and output tensors. vmap
was first introduced by Jax (`jax.vmap <https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html>`__)
and has subsequently been adopted by severeal other tensor libraries (*e.g.*,
`torch.vmap <https://docs.pytorch.org/docs/stable/generated/torch.vmap.html>`__,
`mlx.core.vmap <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.vmap.html>`__).

Given a tensor operation like the following

..  code-block:: python

    def op(x, y):
        return 0.5 * jnp.sum(x * y, axis=0)
    # Operation op:
    # - x has shape (a)
    # - y has shape (a)
    # - output has shape ()

applying vmap produces a new operation that operates on tensors with one additional (leading) dimension:

..  code-block:: python

    vmapped_op = jax.vmap(op)
    # Operation vmapped_op:
    # - x has shape (b, a)
    # - y has shape (b, a)
    # - output has shape (b)

The inner operation (``op``) is applied independently to each sub-tensor stacked along the new dimension.
The vmapped operation can thus similarly be expressed in loop notation:

..  code-block:: python

    for b in range(...):
        z[b] = op(x[b, :], y[b, :])

vmap additionally allows specifing the positions of the vectorized axis in all inputs and outputs
using the ``in_axes`` and ``out_axes`` parameters to vectorize along non-leading dimensions.
For more information, see the `documentation <https://docs.jax.dev/en/latest/automatic-vectorization.html>`__.
vmap may be applied multiple subsequent times to vectorize an operation along multiple axes.

Why use vmap?
*************

The key advantage of vmap over manual looping is that it's much more computationally efficient: It pushes the
vectorization into the underlying, low-level primitive operation that is performed by the backend, rather than
executing it in the less efficient, high-level frontend (*i.e.* Python interpreter).

For example, let's consider a primitive sum-reduction operation ``sum`` with an ``axis`` argument that is implemented
for a given backend, and some custom operation in Python that internally uses ``sum``:

..  code-block:: python

    def op(x):
        # input  has shape (c, d)
        return sum(x, axis=0)
        # output has shape (d)

Manually vectorizing this operation with loop notation over a new leading dimension of length 4
(*i.e.* with input shape ``(b, c, d)`` and ``b = 4``) results in the following invocations of the primitive ``sum``:

..  code-block:: python

    # Invocations of primitive sum with loop notation:
    y[0] = sum(x[0, :, :], axis=0)
    y[1] = sum(x[1, :, :], axis=0)
    y[2] = sum(x[2, :, :], axis=0)
    y[3] = sum(x[3, :, :], axis=0)

In contrast, vmap pushes the vectorization into the primitive operation ``sum`` itself by using its vectorization interface, *i.e.* the ``axis`` parameter:

..  code-block:: python

    # Invocations of primitive sum with vmap notation:
    y = sum(x, axis=1)

This is accomplished as follows: vmap internally creates a *symbolic tensor* that ``op`` is invoked with only once. The symbolic tensor
internally represents the entire tensor with shape ``(b, c, d)``, but appears to ``op`` as if it had shape ``(c, d)``.
Every time the primitive operation ``sum`` is invoked on the symbolic tensor inside of ``op``, the primitive is instead
invoked with the full tensor, and its vectorization interface (*i.e.* the ``axis`` argument)
is modified to account for the additional, hidden axis ``b``. Thus, ``sum`` is only invoked once, and the vectorization along
axis ``b`` is handled internally by the primitive operation.

To support vmap, a tensor framework implements this behavior for all of its primitive operations and their respective vectorization interfaces.

vmap as a universal notation
****************************

vmap allows expressing the vectorization of arbitrary tensor operations, with the exception of axis compositions and vectorized axes that appear only on the input side.
Using vmap as a universal notation for tensor operations, however, typically results in verbose, obscure and error-prone expressions that are
difficult to read and write in all but few cases. For instance, the operation expressed in einx notation as

..  code-block:: python

    # einx notation
    z = einx.dot("b q [k] h, b [k] h c -> b q h c", x, y)

vectorizes along four dimensions and thus requires four subsequent applications of vmap with appropriate ``in_axes`` and ``out_axes`` arguments:

..  code-block:: python

    # vmap notation
    op = jnp.dot
    op = jax.vmap(op, in_axes=(0, 0), out_axes=0)
    op = jax.vmap(op, in_axes=(None, 2), out_axes=1)
    op = jax.vmap(op, in_axes=(1, None), out_axes=1)
    op = jax.vmap(op, in_axes=(3, 2), out_axes=2)
    z = op(x, y)

In particular, the order of the vmap applications may not be changed arbitrarily without also adjusting the ``in_axes`` and ``out_axes`` arguments accordingly.

While vmap may be used as a general notation for vectorization in this way, it is not suited to do so in practice.
Instead, it is typically used for simple vectorization cases along a single axis and in combination with existing, Numpy-like notation:

..  code-block:: python

    def op(x, y):
        # Use Numpy-like notation here
        return 0.5 * jnp.sum(x, axis=1) + jnp.flip(y)

    z = jax.vmap(op)(x, y)
    # Ok, it is clear what is happening here:
    # op is vectorized along a single leading dimension of x, y and z



vmap as a backend in einx
*************************

While vmap notation might not be suited as a general, readable and writable notation for tensor operations, it is still able to represent most cases of vectorization.
einx utilizes this by providing the option to compile operations to vmap notation.

This includes operations that are already included in einx's API (by passing ``backend="{framework}.vmap"``)

..  code-block:: python

    >>> x = jnp.ones((4, 8, 2))
    >>> y = jnp.ones((2, 4))
    >>> print(einx.add("a b c, c a -> b a c", x, y, backend="jax.vmap", graph=True))
    import jax.numpy as jnp
    import jax
    def op(a, b):
        c = jax.vmap(jnp.add, in_axes=(0, 0), out_axes=0)
        c = jax.vmap(c, in_axes=(1, 0), out_axes=1)
        c = jax.vmap(c, in_axes=(1, None), out_axes=0)
        d = c(a, b)
        return d

as well as custom operations which may be adapted to einx notation using ``adapt_with_vmap``:

..  code-block:: python

    >>> def op(x, y):
    >>>     return 0.5 * jnp.sum(x, axis=1) + jnp.flip(y)
    >>> einop = einx.jax.adapt_with_vmap(op)
    >>> 
    >>> x = jnp.ones((2, 3, 4, 5))
    >>> y = jnp.ones((2, 4))
    >>> print(einop("a b [c d], a [c] -> b [c] a", x, y, graph=True))
    # Constant const1: <function op at 0x77d5fe8437e0>
    import jax.numpy as jnp
    import jax
    def op(a, b):
        def c(d, e):
            f = const1(d, e)
            assert isinstance(f, jnp.ndarray), "Expected 1st return value of the adapted function to be a tensor"
            assert (tuple(f.shape) == (4,)), "Expected 1st return value of the adapted function to be a tensor with shape (4,)"
            return f
        c = jax.vmap(c, in_axes=(0, None), out_axes=0)
        c = jax.vmap(c, in_axes=(0, 0), out_axes=2)
        g = c(a, b)
        return g

einx additionally uses Numpy-like functions to handle axis compositions in these cases:

..  code-block:: python

    >>> x = jnp.ones((4 * 8, 2))
    >>> y = jnp.ones((2 * 4,))
    >>> print(einx.add("(a b) c, (c a) -> (b a c)", x, y, backend="jax.vmap", graph=True))
    import jax
    import jax.numpy as jnp
    def op(a, b):
        c = jax.vmap(jnp.add, in_axes=(0, None), out_axes=0)
        c = jax.vmap(c, in_axes=(1, 0), out_axes=1)
        c = jax.vmap(c, in_axes=(0, 1), out_axes=1)
        a = jnp.reshape(a, (4, 8, 2))
        b = jnp.reshape(b, (2, 4))
        d = c(a, b)
        d = jnp.reshape(d, (64,))
        return d

Examples of compiled code
#########################


The following are examples of various einx operations along with the Python code snippet that einx compiles for them, using either
the default backend or explicitly specifying a backend. The compiled code can be inspected by passing ``graph=True`` to the einx operation.

Axis permutation
================

The operation

..  code-block:: Python

    x = np.zeros((10, 5, 2))
    einx.id("a b c -> b c a", x)

compiles to the following code:


With ``backend="numpy"``
------------------------



..  code-block:: Python

    import numpy as np
    def op(a):
        a = np.transpose(a, (1, 2, 0))
        return a


With ``backend="torch"``
------------------------



..  code-block:: Python

    import torch
    def op(a):
        a = torch.asarray(a, device=None)
        a = torch.permute(a, (1, 2, 0))
        return a


With ``backend="jax"``
----------------------



..  code-block:: Python

    import jax.numpy as jnp
    def op(a):
        a = jnp.transpose(a, (1, 2, 0))
        return a


With ``backend="arrayapi"``
---------------------------



..  code-block:: Python

    import array_api_compat
    def op(a):
        b = array_api_compat.array_namespace(a)
        c = b.permute_dims(a, (1, 2, 0))
        return c


Axis flattening
===============

The operation

..  code-block:: Python

    x = np.zeros((10, 5))
    einx.id("(a b) c -> a (b c)", x, b=2)

compiles to the following code:



..  code-block:: Python

    import numpy as np
    def op(a):
        a = np.reshape(a, (5, 10))
        return a


No-op
=====

The operation

..  code-block:: Python

    x = np.zeros((10, 5))
    einx.id("a b -> a b", x)

compiles to the following code:



..  code-block:: Python

    def op(a):
        return a


Element-wise multiplication
===========================

The operation

..  code-block:: Python

    x = jnp.zeros((2, (5 * 6)))
    y = jnp.zeros((4, 3, 6))
    einx.multiply("a (d e), c b e -> a b c d e", x, y)

compiles to the following code:


With ``backend="jax.numpylike"``
--------------------------------



..  code-block:: Python

    import jax.numpy as jnp
    def op(a, b):
        a = jnp.reshape(a, (2, 1, 1, 5, 6))
        b = jnp.transpose(b, (1, 0, 2))
        b = jnp.reshape(b, (1, 3, 4, 1, 6))
        c = jnp.multiply(a, b)
        return c


With ``backend="jax.vmap"``
---------------------------



..  code-block:: Python

    import jax.numpy as jnp
    import jax
    def op(a, b):
        c = jax.vmap(jnp.multiply, in_axes=(None, 0), out_axes=0)
        c = jax.vmap(c, in_axes=(None, 0), out_axes=1)
        c = jax.vmap(c, in_axes=(0, 2), out_axes=2)
        c = jax.vmap(c, in_axes=(0, None), out_axes=2)
        c = jax.vmap(c, in_axes=(0, None), out_axes=0)
        a = jnp.reshape(a, (2, 5, 6))
        d = c(a, b)
        return d


With ``backend="jax.einsum"``
-----------------------------



..  code-block:: Python

    import jax.numpy as jnp
    def op(a, b):
        a = jnp.reshape(a, (2, 5, 6))
        c = jnp.einsum("abc,dec->aedbc", a, b)
        return c


Dot-product
===========

The operation

..  code-block:: Python

    x = jnp.zeros((2, 3))
    y = jnp.zeros((4, 3))
    einx.dot("a [b], c [b] -> c a", x, y)

compiles to the following code:


With ``backend="jax.numpylike"``
--------------------------------



..  code-block:: Python

    import jax.numpy as jnp
    def op(a, b):
        a = jnp.reshape(a, (1, 2, 3))
        b = jnp.transpose(b, (1, 0))
        b = jnp.reshape(b, (1, 3, 4))
        c = jnp.matmul(a, b)
        c = jnp.reshape(c, (2, 4))
        c = jnp.transpose(c, (1, 0))
        return c


With ``backend="jax.vmap"``
---------------------------



..  code-block:: Python

    import jax.numpy as jnp
    import jax
    def op(a, b):
        c = jax.vmap(jnp.dot, in_axes=(None, 0), out_axes=0)
        c = jax.vmap(c, in_axes=(0, None), out_axes=1)
        d = c(a, b)
        return d


With ``backend="jax.einsum"``
-----------------------------



..  code-block:: Python

    import jax.numpy as jnp
    def op(a, b):
        c = jnp.einsum("ab,cb->ca", a, b)
        return c


Indexing
========

The operation

..  code-block:: Python

    x = jnp.zeros((2, 128, 128, 3))
    y = jnp.zeros((50, 2))
    einx.get_at("b [h w] c, p [2] -> b p c", x, y)

compiles to the following code:


With ``backend="jax.numpylike"``
--------------------------------



..  code-block:: Python

    import jax.numpy as jnp
    def op(a, b):
        a = jnp.reshape(a, (98304,))
        c = jnp.arange(2, dtype="int32")
        c = jnp.multiply(c, 49152)
        c = jnp.reshape(c, (2, 1, 1))
        d = jnp.multiply(b[:, 0], 384)
        d = jnp.reshape(d, (1, 50, 1))
        e = jnp.add(c, d)
        b = jnp.multiply(b[:, 1], 3)
        b = jnp.reshape(b, (1, 50, 1))
        f = jnp.add(e, b)
        g = jnp.arange(3, dtype="int32")
        g = jnp.reshape(g, (1, 1, 3))
        h = jnp.add(f, g)
        i = jnp.take(a, h)
        return i


With ``backend="jax.vmap"``
---------------------------



..  code-block:: Python

    import jax
    def c(d, e):
        return d[e[0], e[1]]
    def op(a, b):
        f = jax.vmap(c, in_axes=(None, 0), out_axes=0)
        f = jax.vmap(f, in_axes=(2, None), out_axes=1)
        f = jax.vmap(f, in_axes=(0, None), out_axes=0)
        g = f(a, b)
        return g


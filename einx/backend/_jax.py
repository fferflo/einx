from functools import partial

def make_jax_backend():
    import jax as jax_
    import jax.numpy as jnp
    class jax:
        @staticmethod
        def to_tensor(tensor):
            return jnp.asarray(tensor)

        tensor = jnp.ndarray
        name = "jax"

        cast = lambda tensor, dtype: tensor.astype(dtype)
        reshape = jnp.reshape
        transpose = jnp.transpose
        broadcast_to = jnp.broadcast_to
        einsum = partial(jnp.einsum, optimize="optimal")
        dot = jnp.dot
        swapaxes = jnp.swapaxes

        stack = jnp.stack
        concatenate = jnp.concatenate

        zeros = jnp.zeros
        ones = jnp.ones

        elementwise = lambda *args, op, **kwargs: op(*args, **kwargs)
        add = jnp.add
        subtract = jnp.subtract
        multiply = jnp.multiply
        true_divide = jnp.true_divide
        floor_divide = jnp.floor_divide
        divide = jnp.divide
        logical_and = jnp.logical_and
        logical_or = jnp.logical_or
        where = jnp.where
        less = jnp.less
        less_equal = jnp.less_equal
        greater = jnp.greater
        greater_equal = jnp.greater_equal
        equal = jnp.equal
        not_equal = jnp.not_equal
        maximum = jnp.maximum
        minimum = jnp.minimum

        reduce = lambda *args, op, **kwargs: op(*args, **kwargs)
        sum = jnp.sum
        mean = jnp.mean
        var = jnp.var
        std = jnp.std
        prod = jnp.prod
        count_nonzero = jnp.count_nonzero
        any = jnp.any
        all = jnp.all
        min = jnp.amin
        max = jnp.amax

        map = lambda *args, op, **kwargs: op(*args, **kwargs)
        flip = jnp.flip
        roll = jnp.roll

        sqrt = jnp.sqrt
        rsqrt = jax_.lax.rsqrt
        square = jnp.square

        allclose = jnp.allclose

        vmap = jax_.vmap

        def assert_shape(tensor, shape):
            assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
            return tensor

        class random:
            def bernoulli(rng, p, shape):
                return jax_.random.bernoulli(rng, p, shape)

    return jax
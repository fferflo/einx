from functools import partial
from .base import base_backend

def make_jax_backend():
    import jax as jax_
    import jax.numpy as jnp
    class jax(base_backend):
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

        def get_at(tensor, coordinates):
            return tensor[coordinates]
        def set_at(tensor, coordinates, updates):
            return tensor.at[coordinates].set(updates)
        def add_at(tensor, coordinates, updates):
            return tensor.at[coordinates].add(updates)
        def subtract_at(tensor, coordinates, updates):
            return tensor.at[coordinates].add(-updates)

        flip = jnp.flip
        roll = jnp.roll

        sqrt = jnp.sqrt
        rsqrt = jax_.lax.rsqrt
        square = jnp.square

        allclose = jnp.allclose

        vmap = jax_.vmap

        class random:
            def bernoulli(rng, p, shape):
                return jax_.random.bernoulli(rng, p, shape)

    return jax
from .base import *
import einx.tracer as tracer
from einx.tracer.tensor import op
import einx, types
from functools import partial


def create():
    tjax = tracer.import_("jax")
    tjnp = tracer.import_("jax.numpy", "jnp")
    import jax.numpy as jnp
    import jax as jax_

    class jax(Backend):
        name = "jax"
        tensor_types = [jnp.ndarray]

        @staticmethod
        @einx.trace
        def to_tensor(tensor, shape):
            return einx.tracer.apply(
                tjnp.asarray,
                args=[tensor],
                output=einx.tracer.Tensor(shape),
            )

        reshape = op.reshape(tjnp.reshape)
        transpose = op.transpose(tjnp.transpose)
        broadcast_to = op.broadcast_to(tjnp.broadcast_to)
        einsum = op.einsum(tjnp.einsum)
        arange = op.arange(tjnp.arange)

        stack = op.stack(tjnp.stack)
        concatenate = op.concatenate(tjnp.concatenate)

        add = associative_binary_to_nary(op.elementwise(tjnp.add))
        subtract = op.elementwise(tjnp.subtract)
        multiply = associative_binary_to_nary(op.elementwise(tjnp.multiply))
        true_divide = op.elementwise(tjnp.true_divide)
        floor_divide = op.elementwise(tjnp.floor_divide)
        divide = op.elementwise(tjnp.divide)
        logical_and = associative_binary_to_nary(op.elementwise(tjnp.logical_and))
        logical_or = associative_binary_to_nary(op.elementwise(tjnp.logical_or))
        where = op.elementwise(tjnp.where)
        less = op.elementwise(tjnp.less)
        less_equal = op.elementwise(tjnp.less_equal)
        greater = op.elementwise(tjnp.greater)
        greater_equal = op.elementwise(tjnp.greater_equal)
        equal = op.elementwise(tjnp.equal)
        not_equal = op.elementwise(tjnp.not_equal)
        maximum = associative_binary_to_nary(op.elementwise(tjnp.maximum))
        minimum = associative_binary_to_nary(op.elementwise(tjnp.minimum))

        sum = op.reduce(tjnp.sum)
        mean = op.reduce(tjnp.mean)
        var = op.reduce(tjnp.var)
        std = op.reduce(tjnp.std)
        prod = op.reduce(tjnp.prod)
        count_nonzero = op.reduce(tjnp.count_nonzero)
        any = op.reduce(tjnp.any)
        all = op.reduce(tjnp.all)
        min = op.reduce(tjnp.min)
        max = op.reduce(tjnp.max)
        logsumexp = op.reduce(tjax.scipy.special.logsumexp)

        log = op.elementwise(tjnp.log)
        exp = op.elementwise(tjnp.exp)
        sqrt = op.elementwise(tjnp.sqrt)
        rsqrt = op.elementwise(tjax.lax.rsqrt)
        square = op.elementwise(tjnp.square)

        @staticmethod
        @einx.trace
        def get_at(tensor, coordinates):
            return tensor[coordinates]

        @staticmethod
        @einx.trace
        def set_at(tensor, coordinates, updates):
            return einx.tracer.apply(
                tensor.at[coordinates].set, args=[updates], output=einx.tracer.Tensor(tensor.shape)
            )

        @staticmethod
        @einx.trace
        def add_at(tensor, coordinates, updates):
            return einx.tracer.apply(
                tensor.at[coordinates].add, args=[updates], output=einx.tracer.Tensor(tensor.shape)
            )

        @staticmethod
        @einx.trace
        def subtract_at(tensor, coordinates, updates):
            return einx.tracer.apply(
                tensor.at[coordinates].add, args=[-updates], output=einx.tracer.Tensor(tensor.shape)
            )

        flip = op.keep_shape(tjnp.flip)
        roll = op.keep_shape(tjnp.roll)
        softmax = op.keep_shape(tjax.nn.softmax)
        log_softmax = op.keep_shape(tjax.nn.log_softmax)

        stop_gradient = op.keep_shape(tjax.lax.stop_gradient)

        vmap = op.vmap(tjax.vmap)

        class random:
            @einx.trace
            def bernoulli(rng, p, shape):
                return einx.tracer.apply(
                    tjax.random.bernoulli,
                    args=[rng, p, shape],
                    output=einx.tracer.Tensor(shape),
                )

    return jax()

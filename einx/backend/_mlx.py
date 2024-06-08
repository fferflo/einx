from .base import *
import einx.tracer as tracer
from einx.tracer.tensor import op
import einx, types
from functools import partial


def create():
    import mlx.core as mx

    tmx = tracer.import_("mlx.core", "mx")

    def to_tuple(x):
        if isinstance(x, tuple):
            return x
        elif isinstance(x, list):
            return tuple(x)
        elif isinstance(x, np.ndarray):
            return tuple(x.tolist())
        else:
            raise ValueError(f"Cannot convert {type(x)} to tuple")

    def to_dtype(x):
        if isinstance(x, str):
            if x == "bool":
                return mx.bool_
            else:
                return vars(mx)[x]
        else:
            return x

    to_dtype2 = to_dtype

    class mlx(Backend):
        name = "mlx"
        tensor_types = [mx.array]

        to_dtype = staticmethod(to_dtype2)

        @staticmethod
        @einx.trace
        def to_tensor(tensor, shape):
            return einx.tracer.apply(
                tmx.array,
                args=[tensor],
                output=einx.tracer.Tensor(shape),
            )

        @staticmethod
        @einx.trace
        def reshape(tensor, shape):
            if einx.tracer.is_scalar(tensor):
                tensor = tmx.array(tensor)
            return einx.tracer.apply(
                tmx.reshape, args=[tensor, list(to_tuple(shape))], output=einx.tracer.Tensor(shape)
            )

        transpose = op.transpose(tmx.transpose)
        broadcast_to = op.broadcast_to(tmx.broadcast_to)

        @staticmethod
        @einx.trace
        def einsum(equation, *tensors):
            raise NotImplementedError("mlx does not support einsum yet")

        @staticmethod
        @einx.trace
        def arange(start, stop=None, step=None, dtype="int32"):
            args = [start]
            if stop is not None:
                args.append(stop)
            if step is not None:
                args.append(step)
            return op.arange(tmx.arange)(*args, dtype=to_dtype(dtype))

        stack = op.stack(tmx.stack)
        concatenate = op.concatenate(tmx.concatenate)

        add = associative_binary_to_nary(op.elementwise(tmx.add))
        subtract = op.elementwise(tmx.subtract)
        multiply = associative_binary_to_nary(op.elementwise(tmx.multiply))
        true_divide = op.elementwise(tmx.divide)
        floor_divide = op.elementwise(tmx.floor_divide)
        divide = op.elementwise(tmx.divide)
        mod = op.elementwise(tmx.remainder)
        logical_and = associative_binary_to_nary(op.elementwise(tmx.logical_and))
        logical_or = associative_binary_to_nary(op.elementwise(tmx.logical_or))
        where = op.elementwise(tmx.where)
        less = op.elementwise(tmx.less)
        less_equal = op.elementwise(tmx.less_equal)
        greater = op.elementwise(tmx.greater)
        greater_equal = op.elementwise(tmx.greater_equal)
        equal = op.elementwise(tmx.equal)
        not_equal = op.elementwise(tmx.not_equal)
        maximum = associative_binary_to_nary(op.elementwise(tmx.maximum))
        minimum = associative_binary_to_nary(op.elementwise(tmx.minimum))

        sum = op.reduce(tmx.sum)
        mean = op.reduce(tmx.mean)
        var = op.reduce(tmx.var)
        prod = op.reduce(tmx.prod)
        count_nonzero = op.reduce(tmx.count_nonzero)
        any = op.reduce(tmx.any)
        all = op.reduce(tmx.all)
        min = op.reduce(tmx.min)
        max = op.reduce(tmx.max)
        logsumexp = op.reduce(tmx.logsumexp)

        log = op.elementwise(tmx.log)
        exp = op.elementwise(tmx.exp)
        sqrt = op.elementwise(tmx.sqrt)
        rsqrt = op.elementwise(tmx.rsqrt)
        square = op.elementwise(tmx.square)

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

        softmax = op.keep_shape(tmx.softmax)

        stop_gradient = op.keep_shape(tmx.stop_gradient)

        # vmap = op.vmap(tmx.vmap)
        @staticmethod
        def vmap(op, in_axes, out_axes, input_shapes=None, output_shapes=None):
            raise NotImplementedError("mlx does not fully support vmap yet")

        sqrt = tmx.sqrt
        rsqrt = tmx.rsqrt
        square = tmx.square

        class random:
            @einx.trace
            def bernoulli(rng, p, shape):
                einx.tracer.apply(
                    tmx.random.bernoulli,
                    args=[p, shape, rng],
                    output=einx.tracer.Tensor(shape),
                )

    return mlx()

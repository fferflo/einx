from functools import partial
import functools
from .base import Backend, associative_binary_to_nary


def to_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    elif isinstance(x, np.ndarray):
        return tuple(x.tolist())
    else:
        raise ValueError(f"Cannot convert {type(x)} to tuple")


def make_mlx_backend():
    import mlx.core as mx

    def to_dtype(x):
        if isinstance(x, str):
            if x == "bool":
                return mx.bool_
            else:
                return vars(mx)[x]
        else:
            return x

    class mlx(Backend):
        def to_tensor(tensor):
            return mx.array(tensor)

        tensor = mx.array
        name = "mlx"

        def cast(tensor, dtype):
            return tensor.astype(to_dtype(dtype))

        reshape = mx.reshape
        transpose = mx.transpose
        broadcast_to = mx.broadcast_to

        def einsum(einsum_str, *arrays):
            raise NotImplementedError("mlx does not support einsum yet")

        swapaxes = mx.swapaxes

        def arange(start, stop=None, step=None, dtype="int32"):
            args = [start]
            if stop is not None:
                args.append(stop)
            if step is not None:
                args.append(step)
            return mx.arange(*args, dtype=to_dtype(dtype))

        stack = mx.stack
        concatenate = mx.concatenate

        def zeros(shape, dtype="float32"):
            return mx.zeros(to_tuple(shape), dtype=to_dtype(dtype))

        def ones(shape, dtype="float32"):
            return mx.ones(to_tuple(shape), dtype=to_dtype(dtype))

        add = associative_binary_to_nary(mx.add)
        subtract = mx.subtract
        multiply = associative_binary_to_nary(mx.multiply)
        true_divide = mx.divide
        floor_divide = mx.floor_divide
        divide = mx.divide
        logical_and = associative_binary_to_nary(mx.logical_and)
        logical_or = associative_binary_to_nary(mx.logical_or)
        where = mx.where
        less = mx.less
        less_equal = mx.less_equal
        greater = mx.greater
        greater_equal = mx.greater_equal
        equal = mx.equal
        not_equal = mx.not_equal
        maximum = associative_binary_to_nary(mx.maximum)
        minimum = associative_binary_to_nary(mx.minimum)

        sum = mx.sum
        mean = mx.mean
        var = mx.var

        def std(tensor, axis=None, ddof=0, keepdims=False):
            return mx.sqrt(mx.var(tensor, axis=axis, ddof=ddof, keepdims=keepdims))

        prod = mx.prod
        count_nonzero = mx.sum
        any = mx.any
        all = mx.all
        min = mx.min
        max = mx.max
        logsumexp = mx.logsumexp

        def get_at(tensor, coordinates):
            return tensor[coordinates]

        def set_at(tensor, coordinates, updates):
            tensor[coordinates] = updates
            return tensor

        def add_at(tensor, coordinates, updates):
            tensor[coordinates] += updates
            return tensor

        def subtract_at(tensor, coordinates, updates):
            tensor[coordinates] -= updates
            return tensor

        def flip(tensor, axis):
            if isinstance(axis, int):
                axis = (axis,)
            for axis in axis:
                c = (slice(None),) * axis + (slice(None, None, -1),)
                tensor = tensor[c]
            return tensor

        def roll(tensor, shift, axis):
            if isinstance(axis, int):
                axis = (axis,)
            if isinstance(shift, int):
                shift = (shift,)
            if len(axis) != len(shift):
                raise ValueError(f"Got {len(shift)} shifts, expected {len(axis)}")
            for shift, axis in zip(shift, axis):
                indices = mx.arange(tensor.shape[axis])
                indices = (indices - shift) % tensor.shape[axis]
                c = (slice(None),) * axis + (indices,)
                tensor = tensor[c]
            return tensor

        softmax = mx.softmax

        def log_softmax(x, axis=None):
            x_max = mx.max(x, axis=axis, keepdims=True)
            x = x - mx.stop_gradient(x_max)
            return x - mx.log(mx.sum(mx.exp(x), axis=axis, keepdims=True))

        sqrt = mx.sqrt
        rsqrt = mx.rsqrt
        square = mx.square

        allclose = mx.allclose

        def vmap(op, in_axes, out_axes, input_shapes=None, output_shapes=None):
            raise NotImplementedError("mlx does not fully support vmap yet")
            # return mx.vmap(op, in_axes, out_axes)

        class random:
            def bernoulli(rng, p, shape):
                return mx.random.bernoulli(p, shape, rng)

    return mlx

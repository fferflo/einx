from functools import partial
from .base import Backend, associative_binary_to_nary


def make_dask_backend():
    import dask.array as da

    class dask(Backend):
        def to_tensor(tensor):
            return da.asarray(tensor)

        tensor = da.Array
        name = "dask"

        def cast(tensor, dtype):
            return tensor.astype(dtype)

        reshape = da.reshape
        transpose = da.transpose
        broadcast_to = da.broadcast_to
        einsum = partial(da.einsum, optimize="optimal")
        swapaxes = da.swapaxes
        arange = da.arange

        stack = da.stack
        concatenate = da.concatenate

        zeros = da.zeros
        ones = da.ones

        add = associative_binary_to_nary(da.add)
        subtract = da.subtract
        multiply = associative_binary_to_nary(da.multiply)
        true_divide = da.true_divide
        floor_divide = da.floor_divide
        divide = da.divide
        logical_and = associative_binary_to_nary(da.logical_and)
        logical_or = associative_binary_to_nary(da.logical_or)
        where = da.where
        less = da.less
        less_equal = da.less_equal
        greater = da.greater
        greater_equal = da.greater_equal
        equal = da.equal
        not_equal = da.not_equal
        maximum = associative_binary_to_nary(da.maximum)
        minimum = associative_binary_to_nary(da.minimum)

        sum = da.sum
        mean = da.mean
        var = da.var
        std = da.std
        prod = da.prod
        count_nonzero = da.count_nonzero
        any = da.any
        all = da.all
        min = da.min
        max = da.max

        def logsumexp(x, axis=None, keepdims=False):
            if isinstance(axis, int):
                axis = (axis,)
            x_max1 = da.max(x, axis=axis, keepdims=keepdims)
            x_max2 = x_max1
            if not keepdims:
                for a in sorted(axis):
                    x_max2 = da.expand_dims(x_max2, axis=a)
            return da.log(da.sum(da.exp(x - x_max2), axis=axis, keepdims=keepdims)) + x_max1

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

        flip = da.flip
        roll = da.roll

        def softmax(x, axis=None):
            x = x - da.max(x, axis=axis, keepdims=True)
            return da.exp(x) / da.sum(da.exp(x), axis=axis, keepdims=True)

        def log_softmax(x, axis=None):
            x = x - da.max(x, axis=axis, keepdims=True)
            return x - da.log(da.sum(da.exp(x), axis=axis, keepdims=True))

        sqrt = da.sqrt

        def rsqrt(x):
            return 1.0 / da.sqrt(x)

        square = da.square

        allclose = da.allclose

        def vmap(op, in_axes, out_axes, input_shapes=None, output_shapes=None):
            raise NotImplementedError(
                "Functions relying on vmap are not supported for the dask backend"
            )

    return dask

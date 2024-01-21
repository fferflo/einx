import numpy as np
from functools import partial
from .base import Backend, associative_binary_to_nary

class numpy(Backend):
    @staticmethod
    def to_tensor(tensor):
        return np.asarray(tensor)

    tensor = np.ndarray
    name = "numpy"

    cast = lambda tensor, dtype: tensor.astype(dtype)
    reshape = np.reshape
    transpose = np.transpose
    broadcast_to = np.broadcast_to
    einsum = partial(np.einsum, optimize="optimal")
    dot = np.dot
    swapaxes = np.swapaxes
    arange = np.arange

    stack = np.stack
    concatenate = np.concatenate

    zeros = np.zeros
    ones = np.ones

    add = associative_binary_to_nary(np.add)
    subtract = np.subtract
    multiply = associative_binary_to_nary(np.multiply)
    true_divide = np.true_divide
    floor_divide = np.floor_divide
    divide = np.divide
    logical_and = associative_binary_to_nary(np.logical_and)
    logical_or = associative_binary_to_nary(np.logical_or)
    where = np.where
    less = np.less
    less_equal = np.less_equal
    greater = np.greater
    greater_equal = np.greater_equal
    equal = np.equal
    not_equal = np.not_equal
    maximum = associative_binary_to_nary(np.maximum)
    minimum = associative_binary_to_nary(np.minimum)

    sum = np.sum
    mean = np.mean
    var = np.var
    std = np.std
    prod = np.prod
    count_nonzero = np.count_nonzero
    any = np.any
    all = np.all
    min = np.amin
    max = np.amax
    def logsumexp(x, axis=None, keepdims=False):
        if isinstance(axis, int):
            axis = (axis,)
        x_max1 = np.max(x, axis=axis, keepdims=keepdims)
        x_max2 = x_max1
        if not keepdims:
            for a in sorted(axis):
                x_max2 = np.expand_dims(x_max2, axis=a)
        return np.log(np.sum(np.exp(x - x_max2), axis=axis, keepdims=keepdims)) + x_max1

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

    flip = np.flip
    roll = np.roll

    def softmax(x, axis=None):
        x = x - np.max(x, axis=axis, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    def log_softmax(x, axis=None):
        x = x - np.max(x, axis=axis, keepdims=True)
        return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))

    sqrt = np.sqrt
    rsqrt = lambda x: 1.0 / np.sqrt(x)
    square = np.square

    allclose = np.allclose

    def vmap(op, in_axes, out_axes, input_shapes=None, output_shapes=None):
        if not isinstance(in_axes, (tuple, list)) or not isinstance(out_axes, (tuple, list)):
            raise ValueError("in_axes and out_axes must be tuples or lists of integers")
        def inner(*args):
            if len(args) != len(in_axes):
                raise ValueError(f"Expected {len(in_axes)} arguments, got {len(args)}")
            value = set(arg.shape[axis] for arg, axis in zip(args, in_axes) if not axis is None)
            if len(value) != 1:
                raise ValueError(f"Expected all arguments to have same size along vmap axis, got {value}")
            value = value.pop()
            xs_stacks = [[]] * len(out_axes)
            for i in range(value):
                xs = op(*[arg[(slice(None),) * axis + (i,)] if not axis is None else arg for arg, axis in zip(args, in_axes)])
                if len(xs) != len(out_axes):
                    raise ValueError(f"Expected {len(out_axes)} arguments from vmapped function, got {len(xs)}")
                for xs_stack, x in zip(xs_stacks, xs):
                    xs_stack.append(x)
            xs = tuple(np.stack(xs_stack, axis=out_axis) for out_axis, xs_stack in zip(out_axes, xs_stacks))
            return xs
        inner.__name__ = f"vmap({op.__name__ if '__name__' in dir(op) else str(op)}, in_axes={in_axes}, out_axes={out_axes})"
        return inner

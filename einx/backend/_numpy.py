from .base import *
import einx.tracer as tracer
from einx.tracer.tensor import op
import numpy as np
import einx, types
from functools import partial


def create():
    tnp = tracer.import_("numpy", "np")

    class numpy(Backend):
        name = "numpy"
        tensor_types = [np.ndarray, np.generic, list, tuple, int, float, bool]
        _get_tests = staticmethod(_get_tests)

        @staticmethod
        @einx.trace
        def to_tensor(tensor, shape):
            return einx.tracer.apply(
                tnp.asarray,
                args=[tensor],
                output=einx.tracer.Tensor(shape),
            )

        reshape = op.reshape(tnp.reshape)
        transpose = op.transpose(tnp.transpose)
        broadcast_to = op.broadcast_to(tnp.broadcast_to)
        einsum = op.einsum(tnp.einsum)
        arange = op.arange(tnp.arange)

        stack = op.stack(tnp.stack)
        concatenate = op.concatenate(tnp.concatenate)

        add = associative_binary_to_nary(op.elementwise(tnp.add))
        subtract = op.elementwise(tnp.subtract)
        multiply = associative_binary_to_nary(op.elementwise(tnp.multiply))
        true_divide = op.elementwise(tnp.true_divide)
        floor_divide = op.elementwise(tnp.floor_divide)
        divide = op.elementwise(tnp.divide)
        logical_and = associative_binary_to_nary(op.elementwise(tnp.logical_and))
        logical_or = associative_binary_to_nary(op.elementwise(tnp.logical_or))
        where = op.elementwise(tnp.where)
        less = op.elementwise(tnp.less)
        less_equal = op.elementwise(tnp.less_equal)
        greater = op.elementwise(tnp.greater)
        greater_equal = op.elementwise(tnp.greater_equal)
        equal = op.elementwise(tnp.equal)
        not_equal = op.elementwise(tnp.not_equal)
        maximum = associative_binary_to_nary(op.elementwise(tnp.maximum))
        minimum = associative_binary_to_nary(op.elementwise(tnp.minimum))

        sum = op.reduce(tnp.sum)
        mean = op.reduce(tnp.mean)
        var = op.reduce(tnp.var)
        std = op.reduce(tnp.std)
        prod = op.reduce(tnp.prod)
        count_nonzero = op.reduce(tnp.count_nonzero)
        any = op.reduce(tnp.any)
        all = op.reduce(tnp.all)
        min = op.reduce(tnp.min)
        max = op.reduce(tnp.max)

        log = op.elementwise(tnp.log)
        exp = op.elementwise(tnp.exp)
        sqrt = op.elementwise(tnp.sqrt)
        square = op.elementwise(tnp.square)

        @staticmethod
        @einx.trace
        def get_at(tensor, coordinates):
            return tensor[coordinates]

        @staticmethod
        @einx.trace
        def set_at(tensor, coordinates, updates):
            return tensor.__setitem__(coordinates, updates)

        @staticmethod
        @einx.trace
        def add_at(tensor, coordinates, updates):
            return tensor.__setitem__(
                coordinates, tensor.__getitem__(coordinates).__iadd__(updates)
            )

        @staticmethod
        @einx.trace
        def subtract_at(tensor, coordinates, updates):
            return tensor.__setitem__(
                coordinates, tensor.__getitem__(coordinates).__isub__(updates)
            )

        flip = op.keep_shape(tnp.flip)
        roll = op.keep_shape(tnp.roll)

    numpy.vmap = op.vmap(partial(vmap_forloop, backend=numpy))

    return numpy()


def _get_tests():
    test = types.SimpleNamespace(
        full=lambda shape, value=0.0, dtype="float32": np.full(shape, value, dtype=dtype),
        to_tensor=np.asarray,
        to_numpy=lambda x: x,
    )
    return [(create(), test)]

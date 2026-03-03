import einx._src.adapter as adapter
import numpy as np
from functools import partial
from einx._src.util.functools import use_name_of
from .._util import _associative_binary_to_nary, _to_tensor


def dot(tensordot, to_tensor):
    return adapter.classical_from_numpy.dot(lambda *args: tensordot(*args, axes=1), to_tensor=to_tensor)


class ops:
    def __init__(self, mlx):
        mx = mlx.core

        def to_tensor_all(*args):
            to_tensor_one = _to_tensor(mx.array, forward=[mx.array], convert=["scalar", "numpy"])
            return [to_tensor_one(arg) for arg in args]

        def _to_dtype(x):
            if isinstance(x, str):
                return getattr(mx, x)
            else:
                return x

        self.reshape = adapter.classical_from_numpy.reshape(mx.reshape, to_tensor=to_tensor_all)
        self.transpose = adapter.classical_from_numpy.transpose(mx.transpose, to_tensor=to_tensor_all)
        self.broadcast_to = adapter.classical_from_numpy.broadcast_to(mx.broadcast_to, to_tensor=to_tensor_all)
        self.diagonal = adapter.classical_from_numpy.diagonal(mx.diagonal, self.transpose, to_tensor=to_tensor_all)
        self.stop_gradient = mx.stop_gradient

        self.add = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(mx.add), to_tensor=to_tensor_all)
        self.subtract = adapter.classical_from_numpy.elementwise(mx.subtract, to_tensor=to_tensor_all)
        self.multiply = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(mx.multiply), to_tensor=to_tensor_all)
        self.true_divide = adapter.classical_from_numpy.elementwise(mx.divide, to_tensor=to_tensor_all)
        self.floor_divide = adapter.classical_from_numpy.elementwise(mx.floor_divide, to_tensor=to_tensor_all)
        self.divide = adapter.classical_from_numpy.elementwise(mx.divide, to_tensor=to_tensor_all)
        self.logaddexp = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(mx.logaddexp), to_tensor=to_tensor_all)
        self.logical_and = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(mx.logical_and), to_tensor=to_tensor_all)
        self.logical_or = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(mx.logical_or), to_tensor=to_tensor_all)
        self.where = adapter.classical_from_numpy.elementwise(mx.where, to_tensor=to_tensor_all)
        self.maximum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(mx.maximum), to_tensor=to_tensor_all)
        self.minimum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(mx.minimum), to_tensor=to_tensor_all)
        self.less = adapter.classical_from_numpy.elementwise(mx.less, to_tensor=to_tensor_all)
        self.less_equal = adapter.classical_from_numpy.elementwise(mx.less_equal, to_tensor=to_tensor_all)
        self.greater = adapter.classical_from_numpy.elementwise(mx.greater, to_tensor=to_tensor_all)
        self.greater_equal = adapter.classical_from_numpy.elementwise(mx.greater_equal, to_tensor=to_tensor_all)
        self.equal = adapter.classical_from_numpy.elementwise(mx.equal, to_tensor=to_tensor_all)
        self.not_equal = adapter.classical_from_numpy.elementwise(mx.not_equal, to_tensor=to_tensor_all)
        self.exp = adapter.classical_from_numpy.elementwise(mx.exp, to_tensor=to_tensor_all)
        self.log = adapter.classical_from_numpy.elementwise(mx.log, to_tensor=to_tensor_all)
        self.negative = adapter.classical_from_numpy.elementwise(mx.negative, to_tensor=to_tensor_all)
        self.divmod = adapter.classical_from_numpy.elementwise(mx.divmod, to_tensor=to_tensor_all)

        self.sum = adapter.classical_from_numpy.reduce(mx.sum, to_tensor=to_tensor_all)
        self.mean = adapter.classical_from_numpy.reduce(mx.mean, to_tensor=to_tensor_all)
        self.var = adapter.classical_from_numpy.reduce(mx.var, to_tensor=to_tensor_all)
        self.std = adapter.classical_from_numpy.reduce(mx.std, to_tensor=to_tensor_all)
        self.prod = adapter.classical_from_numpy.reduce(mx.prod, to_tensor=to_tensor_all)
        self.count_nonzero = adapter.classical_from_classical.count_nonzero(self)
        self.any = adapter.classical_from_numpy.reduce(mx.any, to_tensor=to_tensor_all)
        self.all = adapter.classical_from_numpy.reduce(mx.all, to_tensor=to_tensor_all)
        self.max = adapter.classical_from_numpy.reduce(mx.max, to_tensor=to_tensor_all)
        self.min = adapter.classical_from_numpy.reduce(mx.min, to_tensor=to_tensor_all)
        self.logsumexp = adapter.classical_from_numpy.reduce(mx.logsumexp, to_tensor=to_tensor_all)
        self.argmax = adapter.classical_from_numpy.reduce(mx.argmax, to_tensor=to_tensor_all)
        self.argmin = adapter.classical_from_numpy.reduce(mx.argmin, to_tensor=to_tensor_all)

        self.sort = adapter.classical_from_numpy.sort(mx.sort, to_tensor=to_tensor_all)
        self.argsort = adapter.classical_from_numpy.sort(mx.argsort, to_tensor=to_tensor_all)
        self.roll = adapter.classical_from_numpy.roll(mx.roll, to_tensor=to_tensor_all)
        self.flip = adapter.classical_from_classical.flip(self, mx.array.__getitem__)
        self.softmax = adapter.classical_from_numpy.preserve_shape(mlx.nn.softmax, to_tensor=to_tensor_all)
        self.log_softmax = adapter.classical_from_numpy.preserve_shape(mlx.nn.log_softmax, to_tensor=to_tensor_all)

        self.get_at = adapter.classical_from_numpy.get_at(mx.array.__getitem__, mx.take, to_tensor=to_tensor_all)
        self.set_at = adapter.classical_from_numpy.update_at(mx.array.__setitem__, to_tensor=to_tensor_all)
        self.add_at = adapter.classical_from_numpy.update_at(
            lambda x, indices, updates: mx.array.at(x, indices).add(updates), to_tensor=to_tensor_all, broadcast=self.broadcast_to
        )
        self.subtract_at = adapter.classical_from_numpy.update_at(
            lambda x, indices, updates: mx.array.at(x, indices).subtract(updates), to_tensor=to_tensor_all, broadcast=self.broadcast_to
        )

        self.arange = adapter.classical_from_numpy.arange(mx.arange, to_dtype=_to_dtype)
        self.split = adapter.classical_from_numpy.split(mx.split, to_tensor=to_tensor_all)
        self.concatenate = adapter.classical_from_numpy.concatenate(mx.concatenate, to_tensor=to_tensor_all)
        self.dot = adapter.classical_from_mlx.dot(mx.tensordot, to_tensor=to_tensor_all)
        self.matmul = adapter.classical_from_numpy.matmul(mx.matmul, to_tensor=to_tensor_all)

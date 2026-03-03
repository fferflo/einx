import einx._src.adapter as adapter
import numpy as np
from functools import partial
from einx._src.util.functools import use_name_of
from .._util import _associative_binary_to_nary, _to_tensor, _unsupported_op


class ops:
    def __init__(self, xp):
        def to_tensor_all(*args):
            to_tensor_one = _to_tensor(xp.asarray, forward=[adapter.tensortype_from_arrayapi(xp)], convert=["numpy", "scalar"])
            return [to_tensor_one(arg) for arg in args]

        def _to_dtype(x):
            if isinstance(x, str):
                return getattr(xp, x)
            else:
                return x

        self.reshape = adapter.classical_from_numpy.reshape(xp.reshape, to_tensor=to_tensor_all)
        self.transpose = adapter.classical_from_numpy.transpose(xp.permute_dims, to_tensor=to_tensor_all)
        self.broadcast_to = adapter.classical_from_numpy.broadcast_to(xp.broadcast_to, to_tensor=to_tensor_all)
        self.diagonal = adapter.classical_from_numpy.diagonal(xp.linalg.diagonal, self.transpose, to_tensor=to_tensor_all, axis_always_last=True)
        self.stop_gradient = lambda x: x

        self.add = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(xp.add), to_tensor=to_tensor_all)
        self.subtract = adapter.classical_from_numpy.elementwise(xp.subtract, to_tensor=to_tensor_all)
        self.multiply = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(xp.multiply), to_tensor=to_tensor_all)
        self.true_divide = adapter.classical_from_numpy.elementwise(xp.true_divide, to_tensor=to_tensor_all)
        self.floor_divide = adapter.classical_from_numpy.elementwise(xp.floor_divide, to_tensor=to_tensor_all)
        self.divide = adapter.classical_from_numpy.elementwise(xp.divide, to_tensor=to_tensor_all)
        self.logaddexp = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(xp.logaddexp), to_tensor=to_tensor_all)
        self.logical_and = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(xp.logical_and), to_tensor=to_tensor_all)
        self.logical_or = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(xp.logical_or), to_tensor=to_tensor_all)
        self.where = adapter.classical_from_numpy.elementwise(xp.where, to_tensor=to_tensor_all)
        self.maximum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(xp.maximum), to_tensor=to_tensor_all)
        self.minimum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(xp.minimum), to_tensor=to_tensor_all)
        self.less = adapter.classical_from_numpy.elementwise(xp.less, to_tensor=to_tensor_all)
        self.less_equal = adapter.classical_from_numpy.elementwise(xp.less_equal, to_tensor=to_tensor_all)
        self.greater = adapter.classical_from_numpy.elementwise(xp.greater, to_tensor=to_tensor_all)
        self.greater_equal = adapter.classical_from_numpy.elementwise(xp.greater_equal, to_tensor=to_tensor_all)
        self.equal = adapter.classical_from_numpy.elementwise(xp.equal, to_tensor=to_tensor_all)
        self.not_equal = adapter.classical_from_numpy.elementwise(xp.not_equal, to_tensor=to_tensor_all)
        self.exp = adapter.classical_from_numpy.elementwise(xp.exp, to_tensor=to_tensor_all)
        self.log = adapter.classical_from_numpy.elementwise(xp.log, to_tensor=to_tensor_all)
        self.negative = adapter.classical_from_numpy.elementwise(xp.negative, to_tensor=to_tensor_all)
        self.divmod = adapter.classical_from_numpy.elementwise(lambda x, y: (xp.floor_divide(x, y), xp.remainder(x, y)), to_tensor=to_tensor_all)

        self.sum = adapter.classical_from_numpy.reduce(xp.sum, to_tensor=to_tensor_all)
        self.mean = adapter.classical_from_numpy.reduce(xp.mean, to_tensor=to_tensor_all)
        self.var = adapter.classical_from_numpy.reduce(xp.var, to_tensor=to_tensor_all)
        self.std = adapter.classical_from_numpy.reduce(xp.std, to_tensor=to_tensor_all)
        self.prod = adapter.classical_from_numpy.reduce(xp.prod, to_tensor=to_tensor_all)
        self.count_nonzero = adapter.classical_from_numpy.reduce(xp.count_nonzero, to_tensor=to_tensor_all)
        self.any = adapter.classical_from_numpy.reduce(xp.any, to_tensor=to_tensor_all)
        self.all = adapter.classical_from_numpy.reduce(xp.all, to_tensor=to_tensor_all)
        self.max = adapter.classical_from_numpy.reduce(xp.max, to_tensor=to_tensor_all)
        self.min = adapter.classical_from_numpy.reduce(xp.min, to_tensor=to_tensor_all)
        self.logsumexp = adapter.classical_from_classical.logsumexp(self)
        self.argmax = adapter.classical_from_numpy.reduce(xp.argmax, to_tensor=to_tensor_all)
        self.argmin = adapter.classical_from_numpy.reduce(xp.argmin, to_tensor=to_tensor_all)

        self.sort = adapter.classical_from_numpy.sort(xp.sort, to_tensor=to_tensor_all)
        self.argsort = adapter.classical_from_numpy.sort(xp.argsort, to_tensor=to_tensor_all)
        self.roll = adapter.classical_from_numpy.roll(xp.roll, to_tensor=to_tensor_all)
        self.flip = adapter.classical_from_numpy.preserve_shape(xp.flip, to_tensor=to_tensor_all)
        self.softmax = adapter.classical_from_classical.softmax(self)
        self.log_softmax = adapter.classical_from_classical.log_softmax(self)

        def to_tensor_index(x, *args):
            x = _to_tensor(xp.asarray, forward=[adapter.tensortype_from_arrayapi(xp)], convert=["numpy", "scalar"])(x)
            args = [_to_tensor(xp.asarray, forward=[adapter.tensortype_from_arrayapi(xp), "scalar"], convert=["numpy"])(a) for a in args]
            return x, *args

        if hasattr(xp, "getitem"):
            getitem = xp.getitem
        else:
            getitem = lambda x, indices: x[indices]

        self.get_at = adapter.classical_from_numpy.get_at(getitem, xp.take, to_tensor=to_tensor_index, reshape=self.reshape)
        self.set_at = _unsupported_op("set_at", "arrayapi")
        self.add_at = _unsupported_op("add_at", "arrayapi")
        self.subtract_at = _unsupported_op("subtract_at", "arrayapi")

        self.arange = adapter.classical_from_numpy.arange(xp.arange, to_dtype=_to_dtype)
        self.split = _unsupported_op("split", "arrayapi")
        self.concatenate = adapter.classical_from_numpy.concatenate(xp.concat, to_tensor=to_tensor_all)
        self.dot = adapter.classical_from_numpy.dot(xp.dot, to_tensor=to_tensor_all)
        self.matmul = adapter.classical_from_numpy.matmul(xp.matmul, to_tensor=to_tensor_all)

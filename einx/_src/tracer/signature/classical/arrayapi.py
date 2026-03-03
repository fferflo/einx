import einx._src.tracer as tracer
import einx._src.tracer.signature as signature
from einx._src.util.functools import use_name_of
import types


class Lazy:
    def __init__(self, get_xp):
        self.get_xp = get_xp

    def __getattr__(self, name):
        @use_name_of(name)
        def inner(*args, **kwargs):
            xp = self.get_xp()
            op = getattr(xp, name)
            return op(*args, **kwargs)

        return inner


class arrayapi:
    def __init__(self, get_xp):
        self._get_xp = get_xp
        xp = Lazy(get_xp)

        self.asarray = signature.classical.preserve_shape(xp.asarray)
        self.reshape = signature.classical.set_shape(xp.reshape)
        self.permute_dims = signature.classical.transpose(xp.permute_dims)
        self.broadcast_to = signature.classical.set_shape(xp.broadcast_to)
        self.arange = signature.classical.arange(xp.arange)
        self.concat = signature.classical.concatenate(xp.concat)

        self.add = signature.classical.elementwise(xp.add, num_outputs=1)
        self.subtract = signature.classical.elementwise(xp.subtract, num_outputs=1)
        self.multiply = signature.classical.elementwise(xp.multiply, num_outputs=1)
        self.true_divide = signature.classical.elementwise(xp.true_divide, num_outputs=1)
        self.floor_divide = signature.classical.elementwise(xp.floor_divide, num_outputs=1)
        self.divide = signature.classical.elementwise(xp.divide, num_outputs=1)
        self.logical_and = signature.classical.elementwise(xp.logical_and, num_outputs=1)
        self.logical_or = signature.classical.elementwise(xp.logical_or, num_outputs=1)
        self.where = signature.classical.elementwise(xp.where, num_outputs=1)
        self.maximum = signature.classical.elementwise(xp.maximum, num_outputs=1)
        self.minimum = signature.classical.elementwise(xp.minimum, num_outputs=1)
        self.less = signature.classical.elementwise(xp.less, num_outputs=1)
        self.less_equal = signature.classical.elementwise(xp.less_equal, num_outputs=1)
        self.greater = signature.classical.elementwise(xp.greater, num_outputs=1)
        self.greater_equal = signature.classical.elementwise(xp.greater_equal, num_outputs=1)
        self.equal = signature.classical.elementwise(xp.equal, num_outputs=1)
        self.not_equal = signature.classical.elementwise(xp.not_equal, num_outputs=1)
        self.logaddexp = signature.classical.elementwise(xp.logaddexp, num_outputs=1)
        self.exp = signature.classical.elementwise(xp.exp, num_outputs=1)
        self.log = signature.classical.elementwise(xp.log, num_outputs=1)
        self.negative = signature.classical.elementwise(xp.negative, num_outputs=1)
        self.remainder = signature.classical.elementwise(xp.remainder, num_outputs=1)

        self.sum = signature.classical.reduce(xp.sum)
        self.mean = signature.classical.reduce(xp.mean)
        self.var = signature.classical.reduce(xp.var)
        self.std = signature.classical.reduce(xp.std)
        self.prod = signature.classical.reduce(xp.prod)
        self.count_nonzero = signature.classical.reduce(xp.count_nonzero)
        self.all = signature.classical.reduce(xp.all)
        self.any = signature.classical.reduce(xp.any)
        self.min = signature.classical.reduce(xp.min)
        self.max = signature.classical.reduce(xp.max)
        self.argmax = signature.classical.reduce(xp.argmax)
        self.argmin = signature.classical.reduce(xp.argmin)

        self.getitem = signature.classical.getitem()
        self.take = signature.classical.take(xp.take)

        self.dot = signature.classical.dot(xp.dot)
        self.matmul = signature.classical.matmul(xp.matmul)
        self.einsum = signature.classical.einsum(xp.einsum)

        self.roll = signature.classical.preserve_shape(xp.roll)
        self.flip = signature.classical.preserve_shape(xp.flip)
        self.sort = signature.classical.preserve_shape(xp.sort)
        self.argsort = signature.classical.preserve_shape(xp.argsort)

        self.linalg = types.SimpleNamespace(diagonal=signature.classical.diagonal(lambda x: self._get_xp().linalg.diagonal(x), axis_always_last=True))

    @property
    def int32(self):
        return self._get_xp().int32

    @property
    def int64(self):
        return self._get_xp().int64

    @property
    def float32(self):
        return self._get_xp().float32

    @property
    def float64(self):
        return self._get_xp().float64

    @property
    def bool(self):
        return self._get_xp().bool

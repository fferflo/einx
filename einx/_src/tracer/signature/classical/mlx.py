import einx._src.tracer as tracer
import einx._src.tracer.signature as signature
from functools import partial


class nn:
    def __init__(self, nn):
        self.softmax = signature.classical.preserve_shape(nn.softmax)
        self.log_softmax = signature.classical.preserve_shape(nn.log_softmax)


class core:
    def __init__(self, mx):
        self.array = mx.array
        self.array.__getitem__ = signature.classical.getitem()
        self.array.__setitem__ = signature.classical.setitem()
        self.array.at = signature.classical.at

        self.reshape = signature.classical.set_shape(mx.reshape)
        self.transpose = signature.classical.transpose(mx.transpose)
        self.broadcast_to = signature.classical.set_shape(mx.broadcast_to)
        self.arange = signature.classical.arange(mx.arange)
        self.concatenate = signature.classical.concatenate(mx.concatenate)
        self.split = signature.classical.split(mx.split, cumulative=True)
        self.diagonal = signature.classical.diagonal(mx.diagonal)

        self.add = signature.classical.elementwise(mx.add, num_outputs=1)
        self.subtract = signature.classical.elementwise(mx.subtract, num_outputs=1)
        self.multiply = signature.classical.elementwise(mx.multiply, num_outputs=1)
        self.true_divide = signature.classical.elementwise(mx.true_divide, num_outputs=1)
        self.floor_divide = signature.classical.elementwise(mx.floor_divide, num_outputs=1)
        self.divide = signature.classical.elementwise(mx.divide, num_outputs=1)
        self.logical_and = signature.classical.elementwise(mx.logical_and, num_outputs=1)
        self.logical_or = signature.classical.elementwise(mx.logical_or, num_outputs=1)
        self.where = signature.classical.elementwise(mx.where, num_outputs=1)
        self.maximum = signature.classical.elementwise(mx.maximum, num_outputs=1)
        self.minimum = signature.classical.elementwise(mx.minimum, num_outputs=1)
        self.less = signature.classical.elementwise(mx.less, num_outputs=1)
        self.less_equal = signature.classical.elementwise(mx.less_equal, num_outputs=1)
        self.greater = signature.classical.elementwise(mx.greater, num_outputs=1)
        self.greater_equal = signature.classical.elementwise(mx.greater_equal, num_outputs=1)
        self.equal = signature.classical.elementwise(mx.equal, num_outputs=1)
        self.not_equal = signature.classical.elementwise(mx.not_equal, num_outputs=1)
        self.logaddexp = signature.classical.elementwise(mx.logaddexp, num_outputs=1)
        self.exp = signature.classical.elementwise(mx.exp, num_outputs=1)
        self.log = signature.classical.elementwise(mx.log, num_outputs=1)
        self.negative = signature.classical.elementwise(mx.negative, num_outputs=1)
        self.divmod = signature.classical.elementwise(mx.divmod, num_outputs=2)

        self.sum = signature.classical.reduce(mx.sum)
        self.mean = signature.classical.reduce(mx.mean)
        self.var = signature.classical.reduce(mx.var)
        self.std = signature.classical.reduce(mx.std)
        self.prod = signature.classical.reduce(mx.prod)
        self.count_nonzero = signature.classical.reduce(mx.count_nonzero)
        self.all = signature.classical.reduce(mx.all)
        self.any = signature.classical.reduce(mx.any)
        self.min = signature.classical.reduce(mx.min)
        self.max = signature.classical.reduce(mx.max)
        self.argmax = signature.classical.reduce(mx.argmax)
        self.argmin = signature.classical.reduce(mx.argmin)
        self.logsumexp = signature.classical.reduce(mx.logsumexp)

        self.take = signature.classical.take(mx.take)

        self.tensordot = signature.classical.tensordot(mx.tensordot)
        self.matmul = signature.classical.matmul(mx.matmul)
        self.einsum = signature.classical.einsum(mx.einsum)

        self.roll = signature.classical.preserve_shape(mx.roll)
        self.flip = signature.classical.preserve_shape(mx.flip)
        self.sort = signature.classical.preserve_shape(mx.sort)
        self.argsort = signature.classical.preserve_shape(mx.argsort)

        self.vmap = signature.classical.vmap(mx.vmap)

        self.int32 = mx.int32
        self.int64 = mx.int64
        self.float32 = mx.float32
        self.float64 = mx.float64
        self.bool = mx.bool_

        self.stop_gradient = signature.classical.preserve_shape(mx.stop_gradient)


class mlx:
    def __init__(self):
        self.core = core(tracer.signature.python.import_("mlx.core", as_="mx"))
        self.nn = nn(tracer.signature.python.import_("mlx.nn", as_="mnn"))

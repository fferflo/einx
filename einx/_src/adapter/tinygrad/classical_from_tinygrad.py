import einx._src.adapter as adapter
import einx._src.tracer as tracer
import numpy as np
from functools import partial
from einx._src.util.functools import use_name_of
from .._util import _associative_binary_to_nary, _to_tensor, _unsupported_op


class ops:
    def __init__(self, tinygrad):
        def to_tensor_all(*args):
            to_tensor_one = _to_tensor(tinygrad.Tensor, forward=[tinygrad.Tensor], convert=["numpy", "scalar"])
            return [to_tensor_one(arg) for arg in args]

        self.reshape = adapter.classical_from_numpy.reshape(tinygrad.Tensor.reshape, to_tensor=to_tensor_all)
        self.transpose = adapter.classical_from_numpy.transpose(tinygrad.Tensor.permute, to_tensor=to_tensor_all)
        self.broadcast_to = adapter.classical_from_numpy.broadcast_to(tinygrad.Tensor.expand, to_tensor=to_tensor_all)
        self.diagonal = _unsupported_op("diagonal", "tinygrad")
        self.stop_gradient = tinygrad.Tensor.detach

        self.add = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tinygrad.Tensor.add), to_tensor=to_tensor_all)
        self.subtract = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.sub, to_tensor=to_tensor_all)
        self.multiply = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tinygrad.Tensor.mul), to_tensor=to_tensor_all)
        self.true_divide = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.div, to_tensor=to_tensor_all)
        self.floor_divide = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.idiv, to_tensor=to_tensor_all)
        self.divide = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.div, to_tensor=to_tensor_all)
        self.logaddexp = adapter.classical_from_classical.logaddexp(
            self
        )  # adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tinygrad.Tensor.logaddexp), to_tensor=to_tensor_all)
        self.logical_and = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tinygrad.Tensor.mul), to_tensor=to_tensor_all)
        self.logical_or = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tinygrad.Tensor.add), to_tensor=to_tensor_all)
        self.where = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.where, to_tensor=to_tensor_all)
        self.maximum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tinygrad.Tensor.maximum), to_tensor=to_tensor_all)
        self.minimum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(tinygrad.Tensor.minimum), to_tensor=to_tensor_all)
        self.less = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.__lt__, to_tensor=to_tensor_all)
        self.less_equal = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.__le__, to_tensor=to_tensor_all)
        self.greater = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.__gt__, to_tensor=to_tensor_all)
        self.greater_equal = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.__ge__, to_tensor=to_tensor_all)
        self.equal = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.__eq__, to_tensor=to_tensor_all)
        self.not_equal = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.__ne__, to_tensor=to_tensor_all)
        self.exp = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.exp, to_tensor=to_tensor_all)
        self.log = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.log, to_tensor=to_tensor_all)
        self.negative = adapter.classical_from_numpy.elementwise(tinygrad.Tensor.neg, to_tensor=to_tensor_all)

        def divmod(x, y):
            q = tinygrad.Tensor.idiv(x, y)
            r = tinygrad.Tensor.sub(x, tinygrad.Tensor.mul(q, y))
            return q, r

        self.divmod = adapter.classical_from_numpy.elementwise(divmod, to_tensor=to_tensor_all)

        self.sum = adapter.classical_from_numpy.reduce(tinygrad.Tensor.sum, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.mean = adapter.classical_from_numpy.reduce(tinygrad.Tensor.mean, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.var = adapter.classical_from_numpy.reduce(tinygrad.Tensor.var, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.std = adapter.classical_from_numpy.reduce(tinygrad.Tensor.std, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.prod = adapter.classical_from_numpy.reduce(tinygrad.Tensor.prod, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.count_nonzero = adapter.classical_from_numpy.reduce(tinygrad.Tensor.sum, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.any = adapter.classical_from_numpy.reduce(tinygrad.Tensor.any, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.all = adapter.classical_from_numpy.reduce(tinygrad.Tensor.all, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.max = adapter.classical_from_numpy.reduce(tinygrad.Tensor.max, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.min = adapter.classical_from_numpy.reduce(tinygrad.Tensor.min, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.logsumexp = adapter.classical_from_classical.logsumexp(
            self
        )  # adapter.classical_from_numpy.reduce(tinygrad.Tensor.logsumexp, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.argmax = adapter.classical_from_numpy.reduce(tinygrad.Tensor.argmax, to_tensor=to_tensor_all, argname_keepdims="keepdim")
        self.argmin = adapter.classical_from_numpy.reduce(tinygrad.Tensor.argmin, to_tensor=to_tensor_all, argname_keepdims="keepdim")

        self.sort = adapter.classical_from_numpy.sort(lambda x, **kwargs: tinygrad.Tensor.sort(x, **kwargs)[0], to_tensor=to_tensor_all, argname_axis="dim")
        self.argsort = adapter.classical_from_numpy.sort(lambda x, **kwargs: tinygrad.Tensor.sort(x, **kwargs)[1], to_tensor=to_tensor_all, argname_axis="dim")
        self.roll = adapter.classical_from_numpy.roll(tinygrad.Tensor.roll, to_tensor=to_tensor_all, argname_axis="dims", argname_shift="shifts")
        self.flip = adapter.classical_from_numpy.preserve_shape(tinygrad.Tensor.flip, to_tensor=to_tensor_all)
        self.softmax = adapter.classical_from_numpy.preserve_shape(tinygrad.Tensor.softmax, to_tensor=to_tensor_all)
        self.log_softmax = adapter.classical_from_numpy.preserve_shape(tinygrad.Tensor.log_softmax, to_tensor=to_tensor_all)

        def to_tensor_index(x, *args):
            x = _to_tensor(tinygrad.Tensor, forward=[tinygrad.Tensor], convert=["numpy", "scalar"])(x)
            args = [_to_tensor(tinygrad.Tensor, forward=[tinygrad.Tensor, "scalar"], convert=["numpy"])(a) for a in args]
            return x, *args

        self.get_at = adapter.classical_from_numpy.get_at(tinygrad.Tensor.__getitem__, tinygrad.Tensor.gather, to_tensor=to_tensor_index, reshape=self.reshape)

        def update_at(x, indices, updates, op):
            if op == "set":
                op = tinygrad.Tensor.scatter
            elif op == "add":
                op = partial(tinygrad.Tensor.scatter_reduce, reduce="sum")
            elif op == "subtract":
                op = partial(tinygrad.Tensor.scatter_reduce, reduce="sum")
                updates = self.negative(updates)
            else:
                assert False
            assert x.ndim == 1
            assert indices.ndim == 2
            assert indices.shape[1] == 1
            assert updates.ndim == 1
            indices = self.reshape(indices, (indices.shape[0],))
            x = op(x, 0, indices, updates)
            return x

        self.set_at = adapter.classical_from_numpy.update_at(
            partial(update_at, op="set"), to_tensor=to_tensor_index, broadcast=self.broadcast_to, reshape=self.reshape
        )
        self.add_at = adapter.classical_from_numpy.update_at(
            partial(update_at, op="add"), to_tensor=to_tensor_index, broadcast=self.broadcast_to, reshape=self.reshape
        )
        self.subtract_at = adapter.classical_from_numpy.update_at(
            partial(update_at, op="subtract"), to_tensor=to_tensor_index, broadcast=self.broadcast_to, reshape=self.reshape
        )

        self.arange = adapter.classical_from_numpy.arange(tinygrad.Tensor.arange)
        self.split = adapter.classical_from_numpy.split(tinygrad.Tensor.split, to_tensor=to_tensor_all, cumulative=False, argname_axis="dim")
        self.concatenate = adapter.classical_from_numpy.concatenate(tinygrad.Tensor.cat, to_tensor=to_tensor_all, argname_axis="dim")
        self.dot = adapter.classical_from_numpy.dot(tinygrad.Tensor.dot, to_tensor=to_tensor_all)

        def matmul(x, y):
            # 1. Shapes here:
            # x: (..., i, j)
            # y: (..., j, k)

            # Swap contracted axis in y to last axis
            perm = list(range(y.ndim))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            y = self.transpose(y, perm)

            # 2. Shapes here:
            # x: (..., i, j)
            # y: (..., k, j)

            # Insert unitary dimensions
            x = self.reshape(x, x.shape[:-2] + (x.shape[-2], 1, x.shape[-1]))
            y = self.reshape(y, y.shape[:-2] + (1, y.shape[-2], y.shape[-1]))

            # 3. Shapes here:
            # x: (..., i, 1, j)
            # y: (..., 1, k, j)

            z = self.multiply(x, y)

            # 4. Shapes here:
            # z: (..., i, k, j)

            z = self.sum(z, axis=-1)

            # 5. Shapes here:
            # z: (..., i, k)

            return z

        self.matmul = adapter.classical_from_numpy.matmul(matmul, to_tensor=to_tensor_all)

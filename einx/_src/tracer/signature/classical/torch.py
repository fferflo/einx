import einx._src.tracer as tracer
import einx._src.tracer.signature as signature
import types

from functools import partial


class torch:
    def __init__(self, torch=None):
        if torch is None:
            torch = tracer.signature.python.import_("torch")

        self.Tensor = torch.Tensor
        self.Tensor.__getitem__ = signature.classical.getitem()
        self.Tensor.detach = signature.classical.preserve_shape(torch.Tensor.detach)

        self.asarray = signature.classical.preserve_shape(torch.asarray)
        self.reshape = signature.classical.set_shape(torch.reshape)
        self.permute = signature.classical.transpose(torch.permute)
        self.broadcast_to = signature.classical.set_shape(torch.broadcast_to)
        self.arange = signature.classical.arange(torch.arange)
        self.cat = signature.classical.concatenate(torch.cat, argname_axis="dim")
        self.split = signature.classical.split(torch.split, cumulative=False, argname_axis="dim")
        self.diagonal = signature.classical.diagonal(torch.diagonal, argname_axis1="dim1", argname_axis2="dim2")

        self.add = signature.classical.elementwise(torch.add, num_outputs=1)
        self.subtract = signature.classical.elementwise(torch.subtract, num_outputs=1)
        self.multiply = signature.classical.elementwise(torch.multiply, num_outputs=1)
        self.true_divide = signature.classical.elementwise(torch.true_divide, num_outputs=1)
        self.floor_divide = signature.classical.elementwise(torch.floor_divide, num_outputs=1)
        self.divide = signature.classical.elementwise(torch.divide, num_outputs=1)
        self.remainder = signature.classical.elementwise(torch.remainder, num_outputs=1)
        self.logical_and = signature.classical.elementwise(torch.logical_and, num_outputs=1)
        self.logical_or = signature.classical.elementwise(torch.logical_or, num_outputs=1)
        self.where = signature.classical.elementwise(torch.where, num_outputs=1)
        self.maximum = signature.classical.elementwise(torch.maximum, num_outputs=1)
        self.minimum = signature.classical.elementwise(torch.minimum, num_outputs=1)
        self.less = signature.classical.elementwise(torch.less, num_outputs=1)
        self.less_equal = signature.classical.elementwise(torch.less_equal, num_outputs=1)
        self.greater = signature.classical.elementwise(torch.greater, num_outputs=1)
        self.greater_equal = signature.classical.elementwise(torch.greater_equal, num_outputs=1)
        self.eq = signature.classical.elementwise(torch.eq, num_outputs=1)
        self.ne = signature.classical.elementwise(torch.ne, num_outputs=1)
        self.logaddexp = signature.classical.elementwise(torch.logaddexp, num_outputs=1)
        self.exp = signature.classical.elementwise(torch.exp, num_outputs=1)
        self.log = signature.classical.elementwise(torch.log, num_outputs=1)
        self.neg = signature.classical.elementwise(torch.neg, num_outputs=1)

        self.sum = signature.classical.reduce(torch.sum, argname_axis="dim", argname_keepdims="keepdim")
        self.mean = signature.classical.reduce(torch.mean, argname_axis="dim", argname_keepdims="keepdim")
        self.var = signature.classical.reduce(torch.var, argname_axis="dim", argname_keepdims="keepdim")
        self.std = signature.classical.reduce(torch.std, argname_axis="dim", argname_keepdims="keepdim")
        self.prod = signature.classical.reduce(torch.prod, argname_axis="dim", argname_keepdims="keepdim")
        self.count_nonzero = signature.classical.reduce(torch.count_nonzero, argname_axis="dim", argname_keepdims="keepdim")
        self.all = signature.classical.reduce(torch.all, argname_axis="dim", argname_keepdims="keepdim")
        self.any = signature.classical.reduce(torch.any, argname_axis="dim", argname_keepdims="keepdim")
        self.amin = signature.classical.reduce(torch.amin, argname_axis="dim", argname_keepdims="keepdim")
        self.amax = signature.classical.reduce(torch.amax, argname_axis="dim", argname_keepdims="keepdim")
        self.logsumexp = signature.classical.reduce(torch.logsumexp, argname_axis="dim", argname_keepdims="keepdim")
        self.argmax = signature.classical.reduce(torch.argmax, argname_axis="dim", argname_keepdims="keepdim")
        self.argmin = signature.classical.reduce(torch.argmin, argname_axis="dim", argname_keepdims="keepdim")

        self.take = signature.classical.take(torch.take)
        self.index_put_ = signature.classical.setitem(
            lambda x, key, value, *, accumulate: tracer.signature.python.getattr(x, "index_put_")((key,), value, accumulate=accumulate)
        )

        self.dot = signature.classical.dot(torch.dot)
        self.matmul = signature.classical.matmul(torch.matmul)
        self.einsum = signature.classical.einsum(torch.einsum)

        self.roll = signature.classical.preserve_shape(torch.roll)
        self.flip = signature.classical.preserve_shape(torch.flip)
        self.sort = signature.classical.preserve_shape(lambda *args, **kwargs: torch.sort(*args, **kwargs)[0])
        self.argsort = signature.classical.preserve_shape(torch.argsort)

        self.vmap = lambda func, in_dims, out_dims: signature.classical.vmap(
            lambda func, in_axes, out_axes: torch.vmap(func, in_dims=in_axes, out_dims=out_axes)
        )(func, in_axes=in_dims, out_axes=out_dims)

        self.nn = types.SimpleNamespace(
            functional=types.SimpleNamespace(
                softmax=signature.classical.preserve_shape(torch.nn.functional.softmax),
                log_softmax=signature.classical.preserve_shape(torch.nn.functional.log_softmax),
            )
        )

        self.int32 = torch.int32
        self.int64 = torch.int64
        self.float32 = torch.float32
        self.float64 = torch.float64
        self.bool = torch.bool

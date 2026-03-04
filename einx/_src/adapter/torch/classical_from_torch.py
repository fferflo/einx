import numpy as np
from .._util import _associative_binary_to_nary
from .._util import _axis_to_axistuple
from .._util import _to_tensor, _has_type
from einx._src.util.functools import use_name_of
import einx._src.adapter as adapter


def reduce(op, to_tensor, no_none_axis=False, no_axis_tuple=False, scalar_op=None):
    @use_name_of(op)
    def inner(x, **kwargs):
        if "axis" in kwargs:
            kwargs["dim"] = kwargs.pop("axis")
        if "keepdims" in kwargs:
            kwargs["keepdim"] = kwargs.pop("keepdims")
        return op(x, **kwargs)

    return adapter.classical_from_numpy.reduce(inner, to_tensor=to_tensor, no_none_axis=no_none_axis, no_axis_tuple=no_axis_tuple, scalar_op=scalar_op)


def sort(op, to_tensor):
    return adapter.classical_from_numpy.sort(lambda x, *, axis=None, **kwargs: op(x, dim=axis, **kwargs), to_tensor=to_tensor)


def softmax(torch_op, my_op, to_tensor):
    @use_name_of(torch_op)
    def inner(x, *, axis=None):
        (x,) = to_tensor(x)
        if axis is None:
            axis = tuple(range(x.ndim))
        axis = _axis_to_axistuple(axis)
        if len(axis) == 0:
            return x
        elif len(axis) == 1:
            # Use torch's softmax directly for single axis
            return torch_op(x, dim=axis[0])
        else:
            # Use custom implementation for multiple axes
            return my_op(x, axis=axis)

    return inner


class ops:
    def __init__(self, torch, get_device):
        asarray_with_device = lambda x: torch.asarray(x, device=get_device())

        def to_tensor_all(*args):
            to_tensor_one = _to_tensor(asarray_with_device, forward=[torch.Tensor], convert=["numpy", "scalar"])
            return [to_tensor_one(arg) for arg in args]

        def to_tensor_all_notallscalar(*args):
            if all(_has_type(arg, ["scalar"]) for arg in args):
                args = list(args)
                args[0] = _to_tensor(asarray_with_device, forward=[torch.Tensor], convert=["scalar"])(args[0])
            else:
                args = [_to_tensor(asarray_with_device, forward=[torch.Tensor, "scalar"], convert=["numpy"])(arg) for arg in args]

            return tuple(args)

        def _to_dtype(x):
            if isinstance(x, str):
                return getattr(torch, x)
            else:
                return x

        self.reshape = adapter.classical_from_numpy.reshape(torch.reshape, to_tensor=to_tensor_all)
        self.transpose = adapter.classical_from_numpy.transpose(torch.permute, to_tensor=to_tensor_all)
        self.broadcast_to = adapter.classical_from_numpy.broadcast_to(torch.broadcast_to, to_tensor=to_tensor_all)
        self.diagonal = adapter.classical_from_numpy.diagonal(
            torch.diagonal, self.transpose, to_tensor=to_tensor_all, argname_axis1="dim1", argname_axis2="dim2"
        )
        self.stop_gradient = torch.Tensor.detach

        self.add = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(torch.add), to_tensor=to_tensor_all_notallscalar)
        self.subtract = adapter.classical_from_numpy.elementwise(torch.subtract, to_tensor=to_tensor_all_notallscalar)
        self.multiply = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(torch.multiply), to_tensor=to_tensor_all_notallscalar)
        self.true_divide = adapter.classical_from_numpy.elementwise(torch.true_divide, to_tensor=to_tensor_all_notallscalar)
        self.floor_divide = adapter.classical_from_numpy.elementwise(torch.floor_divide, to_tensor=to_tensor_all_notallscalar)
        self.divide = adapter.classical_from_numpy.elementwise(torch.divide, to_tensor=to_tensor_all_notallscalar)

        self.logaddexp = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(torch.logaddexp), to_tensor=to_tensor_all)
        self.logical_and = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(torch.logical_and), to_tensor=to_tensor_all)
        self.logical_or = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(torch.logical_or), to_tensor=to_tensor_all)

        def to_tensor_where(x, *args):
            x = _to_tensor(asarray_with_device, forward=[torch.Tensor], convert=["numpy", "scalar"])(x)
            args = [_to_tensor(asarray_with_device, forward=[torch.Tensor, "scalar"], convert=["numpy"])(a) for a in args]
            return (x, *args)

        self.where = adapter.classical_from_numpy.elementwise(torch.where, to_tensor=to_tensor_where)

        self.maximum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(torch.maximum), to_tensor=to_tensor_all)
        self.minimum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(torch.minimum), to_tensor=to_tensor_all)

        def to_tensor_compare(x, y):
            x = _to_tensor(asarray_with_device, forward=[torch.Tensor], convert=["numpy", "scalar"])(x)
            y = _to_tensor(asarray_with_device, forward=[torch.Tensor, "scalar"], convert=["numpy"])(y)
            return (x, y)

        self.less = adapter.classical_from_numpy.elementwise(torch.less, to_tensor=to_tensor_compare)
        self.less_equal = adapter.classical_from_numpy.elementwise(torch.less_equal, to_tensor=to_tensor_compare)
        self.greater = adapter.classical_from_numpy.elementwise(torch.greater, to_tensor=to_tensor_compare)
        self.greater_equal = adapter.classical_from_numpy.elementwise(torch.greater_equal, to_tensor=to_tensor_compare)
        self.equal = adapter.classical_from_numpy.elementwise(torch.eq, to_tensor=to_tensor_compare)
        self.not_equal = adapter.classical_from_numpy.elementwise(torch.ne, to_tensor=to_tensor_compare)

        self.exp = adapter.classical_from_numpy.elementwise(torch.exp, to_tensor=to_tensor_all)
        self.log = adapter.classical_from_numpy.elementwise(torch.log, to_tensor=to_tensor_all)
        self.negative = adapter.classical_from_numpy.elementwise(torch.neg, to_tensor=to_tensor_all)

        self.divmod = adapter.classical_from_numpy.elementwise(
            lambda x, y: (torch.floor_divide(x, y), torch.remainder(x, y)), to_tensor=to_tensor_all_notallscalar
        )

        self.sum = adapter.classical_from_torch.reduce(torch.sum, to_tensor=to_tensor_all, scalar_op=lambda x: x)
        self.mean = adapter.classical_from_torch.reduce(torch.mean, to_tensor=to_tensor_all, scalar_op=lambda x: x)
        self.var = adapter.classical_from_torch.reduce(torch.var, to_tensor=to_tensor_all, scalar_op=lambda x: x)
        self.std = adapter.classical_from_torch.reduce(torch.std, to_tensor=to_tensor_all, scalar_op=lambda x: x)
        self.prod = adapter.classical_from_torch.reduce(torch.prod, to_tensor=to_tensor_all, no_axis_tuple=True, scalar_op=lambda x: x)
        self.count_nonzero = adapter.classical_from_torch.reduce(torch.count_nonzero, to_tensor=to_tensor_all)
        self.any = adapter.classical_from_torch.reduce(torch.any, to_tensor=to_tensor_all, scalar_op=lambda x: x)
        self.all = adapter.classical_from_torch.reduce(torch.all, to_tensor=to_tensor_all, scalar_op=lambda x: x)
        self.max = adapter.classical_from_torch.reduce(torch.amax, to_tensor=to_tensor_all, scalar_op=lambda x: x)
        self.min = adapter.classical_from_torch.reduce(torch.amin, to_tensor=to_tensor_all, scalar_op=lambda x: x)
        self.logsumexp = adapter.classical_from_torch.reduce(torch.logsumexp, to_tensor=to_tensor_all, no_none_axis=True)
        self.argmax = adapter.classical_from_torch.reduce(torch.argmax, to_tensor=to_tensor_all)
        self.argmin = adapter.classical_from_torch.reduce(torch.argmin, to_tensor=to_tensor_all)

        self.sort = adapter.classical_from_torch.sort(torch.sort, to_tensor=to_tensor_all)
        self.argsort = adapter.classical_from_torch.sort(torch.argsort, to_tensor=to_tensor_all)
        self.roll = adapter.classical_from_numpy.roll(lambda x, *, shift, axis: torch.roll(x, shifts=shift, dims=axis), to_tensor=to_tensor_all)
        self.flip = adapter.classical_from_numpy.preserve_shape(lambda x, *, axis: torch.flip(x, dims=axis), to_tensor=to_tensor_all, no_none_axis=True)
        self.softmax = adapter.classical_from_torch.softmax(
            torch.nn.functional.softmax, adapter.classical_from_classical.softmax(self), to_tensor=to_tensor_all
        )
        self.log_softmax = adapter.classical_from_torch.softmax(
            torch.nn.functional.log_softmax, adapter.classical_from_classical.log_softmax(self), to_tensor=to_tensor_all
        )

        def to_tensor_get_at(x, y):
            x = _to_tensor(asarray_with_device, forward=[torch.Tensor], convert=["numpy", "scalar"])(x)
            y = _to_tensor(asarray_with_device, forward=[torch.Tensor, "scalar"], convert=["numpy"])(y)
            return x, y

        self.get_at = adapter.classical_from_numpy.get_at(torch.Tensor.__getitem__, torch.take, to_tensor=to_tensor_get_at)
        self.set_at = adapter.classical_from_numpy.update_at(
            lambda x, indices, updates: torch.index_put_(x, indices, updates, accumulate=False), to_tensor=to_tensor_all, broadcast=self.broadcast_to
        )
        self.add_at = adapter.classical_from_numpy.update_at(
            lambda x, indices, updates: torch.index_put_(x, indices, updates, accumulate=True), to_tensor=to_tensor_all, broadcast=self.broadcast_to
        )
        self.subtract_at = adapter.classical_from_numpy.update_at(
            lambda x, indices, updates: torch.index_put_(x, indices, torch.neg(updates), accumulate=True), to_tensor=to_tensor_all, broadcast=self.broadcast_to
        )

        self.arange = adapter.classical_from_numpy.arange(lambda x, dtype: torch.arange(x, dtype=dtype, device=get_device()), to_dtype=_to_dtype)
        self.split = adapter.classical_from_numpy.split(torch.split, to_tensor=to_tensor_all, argname_axis="dim", cumulative=False)
        self.concatenate = adapter.classical_from_numpy.concatenate(lambda xs, *, axis=0: torch.cat(xs, dim=axis), to_tensor=to_tensor_all)
        self.dot = adapter.classical_from_numpy.dot(torch.dot, to_tensor=to_tensor_all)
        self.matmul = adapter.classical_from_numpy.matmul(torch.matmul, to_tensor=to_tensor_all)

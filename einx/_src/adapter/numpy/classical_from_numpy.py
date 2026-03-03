import numpy as _np
import einx._src.tracer as tracer
import einx._src.adapter as adapter
from einx._src.util.functools import use_name_of
from .._util import _associative_binary_to_nary, _axis_to_axisint, _axis_to_axistuple, _to_tensor
from einx._src.frontend.errors import OperationNotSupportedError


def reshape(op, to_tensor=None):
    def reshape(x, shape):
        shape = tuple(shape)
        if tuple(x.shape) == shape:
            return x
        else:
            (x,) = to_tensor(x)
            return op(x, shape)

    return reshape


def transpose(op, to_tensor):
    def transpose(x, perm):
        perm = tuple(perm)
        if perm == tuple(range(len(perm))):
            return x
        else:
            (x,) = to_tensor(x)
            return op(x, perm)

    return transpose


def broadcast_to(op, to_tensor):
    def broadcast_to(x, shape):
        shape = tuple(shape)
        if tuple(x.shape) == shape:
            return x
        else:
            (x,) = to_tensor(x)
            return op(x, shape)

    return broadcast_to


def diagonal(diagonal, transpose, to_tensor, argname_axis1="axis1", argname_axis2="axis2", axis_always_last=False):
    if axis_always_last:
        diagonal0 = diagonal

        def diagonal(x, axis1, axis2):
            # Transpose axis1 and axis2 to the last two axes
            axis1, axis2 = sorted([axis1, axis2])
            perm = [i for i in range(x.ndim) if i != axis1 and i != axis2] + [axis1, axis2]
            x = transpose(x, perm)

            # Call diagonal (always on the last two axes)
            x = diagonal0(x)

            return x

    def inner(x, axes_in, axis_out):
        (x,) = to_tensor(x)

        def canon_axis(axis):
            if axis < 0:
                axis += x.ndim
            if axis < 0 or axis >= x.ndim:
                raise ValueError(f"Invalid axis {axis} for array of dimension {x.ndim}")
            return axis

        axes_in = sorted([canon_axis(axis) for axis in axes_in])
        axis_out = canon_axis(axis_out)

        # Call diagonal for every pair of in-axes, start from highest axes to keep axis indices valid.
        while len(axes_in) > 1:
            axes_in_now = axes_in[-2:]
            kwargs = {argname_axis1: axes_in_now[0], argname_axis2: axes_in_now[1]}
            x = diagonal(x, **kwargs)
            axes_in = tuple(axes_in[:-2]) + (x.ndim - 1,)
        axis_in = axes_in[0]

        # Only one in-axis remains. Move it to the out-axis.
        perm = []
        for i in range(x.ndim):
            if i == axis_out:
                perm.append(axis_in)
            elif i == axis_in:
                perm.append(axis_out)
            else:
                perm.append(i)
        x = transpose(x, perm)

        return x

    inner.__name__ = "diagonal"
    return inner


def elementwise(op, to_tensor=None):
    @use_name_of(op)
    def inner(*xs):
        xs = to_tensor(*xs)
        return op(*xs)

    return inner


def reduce(op, to_tensor, no_none_axis=False, no_axis_tuple=False, scalar_op=None, argname_axis="axis", argname_keepdims="keepdims"):
    @use_name_of(op)
    def inner(x, *, axis=None, keepdims=None):
        (x,) = to_tensor(x)

        kwargs = {}

        if axis is not None:
            if isinstance(axis, list | _np.ndarray):
                axis = tuple(axis)
            if isinstance(axis, tuple) and len(axis) == 1:
                axis = axis[0]
            kwargs["axis"] = axis
        elif no_none_axis:
            kwargs["axis"] = tuple(range(x.ndim))

        if keepdims is not None:
            kwargs["keepdims"] = keepdims

        if scalar_op is not None and (x.ndim == 0 or ("axis" in kwargs and kwargs["axis"] == ())):
            return scalar_op(x)

        if no_axis_tuple and "axis" in kwargs and isinstance(kwargs["axis"], tuple):
            if len(kwargs["axis"]) == 1:
                kwargs["axis"] = kwargs["axis"][0]
            elif set(kwargs["axis"]) == set(range(x.ndim)):
                kwargs["axis"] = None
            else:
                raise ValueError(f"This backend uses {op.__name__} which allows reduction only over (1) all axes or (2) a single axis.")

        if "axis" in kwargs:
            kwargs[argname_axis] = kwargs.pop("axis")
        if "keepdims" in kwargs:
            kwargs[argname_keepdims] = kwargs.pop("keepdims")

        return op(x, **kwargs)

    return inner


def sort(op, to_tensor, argname_axis="axis"):
    @use_name_of(op)
    def inner(x, axis=None, **kwargs):
        (x,) = to_tensor(x)
        if axis is None:
            if x.ndim == 1:
                axis = 0
            else:
                raise ValueError("When 'axis' is not specified, 'x' must be a 1D array.")
        kwargs = {**kwargs}
        kwargs[argname_axis] = _axis_to_axisint(axis)
        return op(x, **kwargs)

    return inner


def roll(op, to_tensor, argname_axis="axis", argname_shift="shift"):
    def roll(x, *, shift, axis=None):
        (x,) = to_tensor(x)
        if axis is None:
            axis = tuple(range(x.ndim))
        axis = _axis_to_axistuple(axis)
        shift = _axis_to_axistuple(shift)
        if len(shift) != len(axis):
            if len(shift) == 1:
                shift = shift * len(axis)
            else:
                raise ValueError(f"When 'shift' has length != 1, it must have the same length as 'axis'. Got lengths {len(shift)} and {len(axis)}.")

        kwargs = {}
        kwargs[argname_axis] = axis
        kwargs[argname_shift] = shift
        return op(x, **kwargs)

    return roll


def preserve_shape(op, to_tensor, no_none_axis=False):
    @use_name_of(op)
    def inner(x, **kwargs):
        (x,) = to_tensor(x)
        if "axis" in kwargs and isinstance(kwargs["axis"], list | _np.ndarray):
            kwargs["axis"] = tuple(kwargs["axis"])
        if no_none_axis and "axis" not in kwargs:
            kwargs["axis"] = tuple(range(x.ndim))
        return op(x, **kwargs)

    return inner


def matmul(op, to_tensor):
    def matmul(x, y):
        x, y = to_tensor(x, y)
        return op(x, y)

    return matmul


def get_at(getitem, take, to_tensor, reshape=None):
    def get_at(x, indices, *, axis=None):
        if axis is None:
            # Multi-dimensional indexing
            if not isinstance(indices, tuple):
                raise ValueError(f"Expected indices to be a tuple, but got {type(indices)}")
            if len(indices) != x.ndim:
                raise ValueError(f"Expected indices to have the same number of elements as x.ndim, but got {len(indices)} and {x.ndim}")
            x, *indices = to_tensor(x, *indices)
            indices = tuple(indices)
            indices_shapes = {i.shape for i in indices}
            if len(indices_shapes) != 1:
                raise ValueError(f"Expected all indices to have the same shape, but got {indices_shapes}")
            return getitem(x, indices)
        else:
            # Single-dimensional indexing
            x, indices = to_tensor(x, indices)
            if not hasattr(indices, "shape") or indices.ndim == 0:
                if axis < 0:
                    axis += x.ndim
                if axis < 0 or axis >= x.ndim:
                    raise ValueError(f"Invalid axis {axis} for array of dimension {x.ndim}")
                return getitem(x, (slice(None),) * axis + (indices,) + (slice(None),) * (x.ndim - axis - 1))
            elif x.ndim == 1 and axis == 0:
                if reshape is not None:
                    shape = indices.shape
                    indices = reshape(indices, (int(_np.prod(shape)),))
                x = take(x, indices)
                if reshape is not None:
                    x = reshape(x, shape)
                return x
            else:
                raise NotImplementedError("Don't know how to express this operation")

    return get_at


def update_at(op, to_tensor, broadcast=None, reshape=None):
    @use_name_of(op)
    def inner(x, indices, updates):
        x, indices, updates = to_tensor(x, indices, updates)
        if x.ndim != 1:
            raise ValueError(f"Expected 1D array, but got {x.ndim}D")
        if indices.ndim != updates.ndim:
            raise ValueError(f"Expected indices and updates to have the same number of dimensions, but got {indices.ndim}D and {updates.ndim}D")

        if broadcast is not None:
            shape = tuple(int(i) for i in _np.maximum(_np.asarray(indices.shape), _np.asarray(updates.shape)))
            indices = broadcast(indices, shape)
            updates = broadcast(updates, shape)

        if reshape is not None:
            l = int(_np.prod(indices.shape))
            indices = reshape(indices, (l, 1))
            updates = reshape(updates, (l,))

        return op(x, indices, updates)

    return inner


def arange(op, to_dtype=lambda x: x):
    def arange(n, dtype="int32"):
        if not isinstance(n, int | _np.integer):
            raise ValueError(f"Expected an integer for n, but got {type(n)}")
        return op(n, dtype=to_dtype(dtype))

    return arange


def split(op, to_tensor, cumulative=True, argname_axis="axis"):
    def split(x, indices, axis=0):
        (x,) = to_tensor(x)

        if axis < 0:
            axis += x.ndim
        if axis < 0 or axis >= x.ndim:
            raise ValueError(f"Invalid axis {axis} for array of dimension {x.ndim}")

        if not cumulative:
            indices = (0,) + tuple(indices) + (x.shape[axis],)
            indices = _np.diff(indices)
            indices = [int(s) for s in indices]

        kwargs = {}
        kwargs[argname_axis] = axis
        return op(x, indices, **kwargs)

    return split


def concatenate(op, to_tensor, argname_axis="axis"):
    def concatenate(xs, axis=0):
        xs = to_tensor(*xs)
        if axis < 0:
            axis += xs[0].ndim
        if axis < 0 or axis >= xs[0].ndim:
            raise ValueError(f"Invalid axis {axis} for arrays of dimension {xs[0].ndim}")
        kwargs = {}
        kwargs[argname_axis] = axis
        return op(xs, **kwargs)

    return concatenate


def dot(dot, to_tensor):
    def inner(*tensors):
        if len(tensors) != 2:
            raise OperationNotSupportedError("dot operation with this backend does not support more than two argument tensors.")
        x, y = to_tensor(*tensors)
        if x.ndim != 1 or y.ndim != 1:
            raise OperationNotSupportedError(f"dot operation with this backend only supports a single contraction axis, but found {x.ndim} contraction axes")
        if x.shape != y.shape:
            raise ValueError(f"Expected x and y to have the same shape, but got {x.shape} and {y.shape}")
        return dot(x, y)

    inner.__name__ = "dot"
    return inner


class ops:
    def __init__(self, np):
        def to_tensor_forward_all(*args):
            to_tensor_one = _to_tensor(np.asarray, forward=["numpy", "scalar"], convert=[])
            return [to_tensor_one(arg) for arg in args]

        self.reshape = adapter.classical_from_numpy.reshape(np.reshape, to_tensor=to_tensor_forward_all)
        self.transpose = adapter.classical_from_numpy.transpose(np.transpose, to_tensor=to_tensor_forward_all)
        self.broadcast_to = adapter.classical_from_numpy.broadcast_to(np.broadcast_to, to_tensor=to_tensor_forward_all)
        self.diagonal = adapter.classical_from_numpy.diagonal(np.diagonal, self.transpose, to_tensor=to_tensor_forward_all)
        self.stop_gradient = lambda x: x

        self.add = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(np.add), to_tensor=to_tensor_forward_all)
        self.subtract = adapter.classical_from_numpy.elementwise(np.subtract, to_tensor=to_tensor_forward_all)
        self.multiply = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(np.multiply), to_tensor=to_tensor_forward_all)
        self.true_divide = adapter.classical_from_numpy.elementwise(np.true_divide, to_tensor=to_tensor_forward_all)
        self.floor_divide = adapter.classical_from_numpy.elementwise(np.floor_divide, to_tensor=to_tensor_forward_all)
        self.divide = adapter.classical_from_numpy.elementwise(np.divide, to_tensor=to_tensor_forward_all)
        self.logaddexp = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(np.logaddexp), to_tensor=to_tensor_forward_all)
        self.logical_and = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(np.logical_and), to_tensor=to_tensor_forward_all)
        self.logical_or = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(np.logical_or), to_tensor=to_tensor_forward_all)
        self.where = adapter.classical_from_numpy.elementwise(np.where, to_tensor=to_tensor_forward_all)
        self.maximum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(np.maximum), to_tensor=to_tensor_forward_all)
        self.minimum = adapter.classical_from_numpy.elementwise(_associative_binary_to_nary(np.minimum), to_tensor=to_tensor_forward_all)
        self.less = adapter.classical_from_numpy.elementwise(np.less, to_tensor=to_tensor_forward_all)
        self.less_equal = adapter.classical_from_numpy.elementwise(np.less_equal, to_tensor=to_tensor_forward_all)
        self.greater = adapter.classical_from_numpy.elementwise(np.greater, to_tensor=to_tensor_forward_all)
        self.greater_equal = adapter.classical_from_numpy.elementwise(np.greater_equal, to_tensor=to_tensor_forward_all)
        self.equal = adapter.classical_from_numpy.elementwise(np.equal, to_tensor=to_tensor_forward_all)
        self.not_equal = adapter.classical_from_numpy.elementwise(np.not_equal, to_tensor=to_tensor_forward_all)
        self.exp = adapter.classical_from_numpy.elementwise(np.exp, to_tensor=to_tensor_forward_all)
        self.log = adapter.classical_from_numpy.elementwise(np.log, to_tensor=to_tensor_forward_all)
        self.negative = adapter.classical_from_numpy.elementwise(np.negative, to_tensor=to_tensor_forward_all)
        self.divmod = adapter.classical_from_numpy.elementwise(np.divmod, to_tensor=to_tensor_forward_all)

        self.sum = adapter.classical_from_numpy.reduce(np.sum, to_tensor=to_tensor_forward_all)
        self.mean = adapter.classical_from_numpy.reduce(np.mean, to_tensor=to_tensor_forward_all)
        self.var = adapter.classical_from_numpy.reduce(np.var, to_tensor=to_tensor_forward_all)
        self.std = adapter.classical_from_numpy.reduce(np.std, to_tensor=to_tensor_forward_all)
        self.prod = adapter.classical_from_numpy.reduce(np.prod, to_tensor=to_tensor_forward_all)
        self.count_nonzero = adapter.classical_from_numpy.reduce(np.count_nonzero, to_tensor=to_tensor_forward_all)
        self.any = adapter.classical_from_numpy.reduce(np.any, to_tensor=to_tensor_forward_all)
        self.all = adapter.classical_from_numpy.reduce(np.all, to_tensor=to_tensor_forward_all)
        self.max = adapter.classical_from_numpy.reduce(np.max, to_tensor=to_tensor_forward_all)
        self.min = adapter.classical_from_numpy.reduce(np.min, to_tensor=to_tensor_forward_all)
        self.logsumexp = adapter.classical_from_classical.logsumexp(self)
        self.argmax = adapter.classical_from_numpy.reduce(np.argmax, to_tensor=to_tensor_forward_all)
        self.argmin = adapter.classical_from_numpy.reduce(np.argmin, to_tensor=to_tensor_forward_all)

        self.sort = adapter.classical_from_numpy.sort(np.sort, to_tensor=to_tensor_forward_all)
        self.argsort = adapter.classical_from_numpy.sort(np.argsort, to_tensor=to_tensor_forward_all)
        self.roll = adapter.classical_from_numpy.roll(np.roll, to_tensor=to_tensor_forward_all)
        self.flip = adapter.classical_from_numpy.preserve_shape(np.flip, to_tensor=to_tensor_forward_all)
        self.softmax = adapter.classical_from_classical.softmax(self)
        self.log_softmax = adapter.classical_from_classical.log_softmax(self)

        def to_tensor_index(x, *args):
            x = _to_tensor(np.asarray, forward=["numpy"], convert=["scalar"])(x)
            args = [_to_tensor(np.asarray, forward=["numpy", "scalar"], convert=[])(a) for a in args]
            return x, *args

        self.get_at = adapter.classical_from_numpy.get_at(np.ndarray.__getitem__, np.take, to_tensor=to_tensor_index)
        self.set_at = adapter.classical_from_numpy.update_at(np.put, to_tensor=to_tensor_index)
        self.add_at = adapter.classical_from_numpy.update_at(np.add.at, to_tensor=to_tensor_index, broadcast=self.broadcast_to)
        self.subtract_at = adapter.classical_from_numpy.update_at(np.subtract.at, to_tensor=to_tensor_index, broadcast=self.broadcast_to)

        self.arange = adapter.classical_from_numpy.arange(np.arange)
        self.split = adapter.classical_from_numpy.split(np.split, to_tensor=to_tensor_forward_all)
        self.concatenate = adapter.classical_from_numpy.concatenate(np.concatenate, to_tensor=to_tensor_forward_all)
        self.dot = adapter.classical_from_numpy.dot(np.dot, to_tensor=to_tensor_forward_all)
        self.matmul = adapter.classical_from_numpy.matmul(np.matmul, to_tensor=to_tensor_forward_all)

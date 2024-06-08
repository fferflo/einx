import einx
import functools
import threading
import numpy as np
from einx.tracer.tensor import op


def associative_binary_to_nary(binary_op):
    @einx.trace
    def nary_op(*args):
        x = args[0]
        for y in args[1:]:
            x = binary_op(x, y)
        return x

    return nary_op


def vmap_forloop(op, in_axes, out_axes, backend):
    if not isinstance(in_axes, (tuple, list)) or not isinstance(out_axes, (tuple, list)):
        raise ValueError("in_axes and out_axes must be tuples or lists of integers")

    def stack(xs, axis):
        return einx.jit(lambda *xs, backend: backend.stack(xs, axis=axis))(*xs)

    def inner(*args):
        if len(args) != len(in_axes):
            raise ValueError(f"Expected {len(in_axes)} arguments, got {len(args)}")
        value = {arg.shape[axis] for arg, axis in zip(args, in_axes) if axis is not None}
        if len(value) != 1:
            raise ValueError(
                f"Expected all arguments to have same size along vmap axis, got {value}"
            )
        value = value.pop()
        xs_stacks = [[]] * len(out_axes)
        for i in range(value):
            xs = op(*[
                arg[(slice(None),) * axis + (i,)] if axis is not None else arg
                for arg, axis in zip(args, in_axes)
            ])
            if len(xs) != len(out_axes):
                raise ValueError(
                    f"Expected {len(out_axes)} arguments from vmapped function, got {len(xs)}"
                )
            for xs_stack, x in zip(xs_stacks, xs):
                xs_stack.append(x)
        xs = tuple(
            stack(xs_stack, axis=out_axis) for out_axis, xs_stack in zip(out_axes, xs_stacks)
        )
        return xs

    inner.__name__ = f"vmap({op.__name__ if '__name__' in dir(op) else str(op)}, "
    f"in_axes={in_axes}, out_axes={out_axes})"

    return inner


_thread_local = threading.local()


def _get_backend_stack():
    if not hasattr(_thread_local, "backend_stack"):
        _thread_local.backend_stack = []
    return _thread_local.backend_stack


def get_default():
    if len(_get_backend_stack()) > 0:
        return _get_backend_stack()[-1]
    else:
        return None


class Backend:
    function_name = None
    decorators = []

    def __enter__(backend):
        _get_backend_stack().append(backend)
        return backend

    def __exit__(backend, *args):
        assert _get_backend_stack()[-1] is backend
        _get_backend_stack().pop()

    @staticmethod
    def _decorate_construct_graph(f):
        return f

    @classmethod
    @einx.trace
    def all_to_tensor(backend, tensors, convert_scalars=False):
        def to_tensor(tensor):
            if isinstance(tensor, einx.tracer.TensorRequiringConversion) or (
                convert_scalars and einx.tracer.is_scalar(tensor)
            ):
                tensor = backend.to_tensor(tensor, tensor.shape)
            return tensor

        return [to_tensor(tensor) for tensor in tensors]

    @classmethod
    @einx.trace
    def stack(backend, tensors, axis=0):
        s = (slice(None),) * axis + (None,)
        return backend.concatenate([tensor[s] for tensor in tensors], axis=axis)

    @classmethod
    @einx.trace
    def mod(backend, x, y):
        return backend.subtract(x, backend.multiply(backend.floor_divide(x, y), y))

    @classmethod
    @einx.trace
    def logsumexp(backend, x, axis=None):
        if isinstance(axis, int):
            axis = (axis,)
        x_max_keepdims = backend.max(x, axis=axis, keepdims=True)
        x_max_keepdims = backend.stop_gradient(x_max_keepdims)
        x_max_dropdims = backend.reshape(
            x_max_keepdims,
            tuple(s for i, s in enumerate(x_max_keepdims.shape) if i not in axis),
        )
        return (
            backend.log(backend.sum(backend.exp(x - x_max_keepdims), axis=axis, keepdims=False))
            + x_max_dropdims
        )

    @classmethod
    @einx.trace
    def std(backend, x, axis=None, keepdims=False):
        return backend.sqrt(backend.var(x, axis=axis, keepdims=keepdims))

    @classmethod
    @einx.trace
    def prod(backend, tensor, axis=None):
        tensor = backend.log(tensor)
        tensor = backend.sum(tensor, axis=axis)
        tensor = backend.exp(tensor)
        return tensor

    @classmethod
    @einx.trace
    def any(backend, tensor, axis=None):
        return backend.count_nonzero(tensor, axis=axis) > 0

    @classmethod
    @einx.trace
    def all(backend, tensor, axis=None):
        if axis is None:
            total_num = np.prod(tensor.shape)
        elif isinstance(axis, int):
            total_num = tensor.shape[axis]
        else:
            total_num = np.prod([tensor.shape[i] for i in axis])
        return backend.count_nonzero(tensor, axis=axis) == total_num

    @classmethod
    @einx.trace
    def softmax(backend, x, axis=None):
        x_max = backend.max(x, axis=axis, keepdims=True)
        x_max = backend.stop_gradient(x_max)
        x = x - x_max
        return backend.exp(x) / backend.sum(backend.exp(x), axis=axis, keepdims=True)

    @classmethod
    @einx.trace
    def log_softmax(backend, x, axis=None):
        x_max = backend.max(x, axis=axis, keepdims=True)
        x_max = backend.stop_gradient(x_max)
        x = x - x_max
        return x - backend.log(backend.sum(backend.exp(x), axis=axis, keepdims=True))

    @classmethod
    @einx.trace
    def flip(backend, tensor, axis):
        if isinstance(axis, int):
            axis = (axis,)
        for axis in axis:
            c = (slice(None),) * axis + (slice(None, None, -1),)
            tensor = tensor[c]
        return tensor

    @classmethod
    @einx.trace
    def roll(backend, tensor, shift, axis):
        if isinstance(axis, int):
            axis = (axis,)
        if isinstance(shift, int):
            shift = (shift,)
        if len(axis) != len(shift):
            raise ValueError(f"Got {len(shift)} shifts, expected {len(axis)}")
        for shift, axis in zip(shift, axis):
            indices = backend.arange(tensor.shape[axis])
            indices = backend.mod(indices - shift, tensor.shape[axis])
            c = (slice(None),) * axis + (indices,)
            tensor = tensor[c]
        return tensor

    @classmethod
    @einx.trace
    def rsqrt(backend, x):
        return 1.0 / backend.sqrt(x)

    @classmethod
    @einx.trace
    def vmap(backend, op, in_axes, out_axes):
        return einx.tracer.import_("einx").backend.vmap_forloop(
            backend, op, in_axes=in_axes, out_axes=out_axes
        )

    stop_gradient = op.keep_shape(einx.trace(lambda x: x))


class ErrorBackend:
    def __init__(self, message):
        self.message = message

    def __getattr__(self, name):
        raise RuntimeError(self.message)

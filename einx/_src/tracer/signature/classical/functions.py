import einx._src.tracer as tracer
import numpy as np
from functools import partial
from einx._src.util.functools import use_name_of
import einx._src.util.pytree as pytree


def _get_shape(x):
    if isinstance(x, tracer.signature.classical.Tensor | tracer.signature.classical.ConvertibleTensor):
        return tuple(x.shape)
    elif isinstance(x, int | float | bool | np.integer | np.floating | np.bool_):
        return ()
    else:
        raise ValueError(f"Object of type {type(x)} is not a tensor or scalar")


def preserve_shape(op, num_outputs=1):
    @use_name_of(op)
    def inner(x, **kwargs):
        shape = _get_shape(x)
        x = op(x, **kwargs)

        if num_outputs == 1:
            output = partial(tracer.signature.classical.Tensor, shape=shape)
        else:
            assert num_outputs > 1
            output = lambda origin: tuple(tracer.signature.classical.Tensor(origin, shape=shape) for _ in range(num_outputs))

        return tracer.cast(x, output)

    return inner


def set_shape(op):
    @use_name_of(op)
    def inner(x, shape):
        x = op(x, shape)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape))
        return x

    return inner


def transpose(op):
    @use_name_of(op)
    def inner(x, axes):
        shape_in = _get_shape(x)
        shape = tuple(shape_in[i] for i in axes)
        x = op(x, axes)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape))
        return x

    return inner


def diagonal(op, argname_axis1="axis1", argname_axis2="axis2", axis_always_last=False):
    @use_name_of(op)
    def inner(x, **kwargs):
        if axis_always_last:
            axis1 = x.ndim - 2
            axis2 = x.ndim - 1
        else:
            if argname_axis1 not in kwargs:
                raise ValueError(f"Missing required argument '{argname_axis1}' in diagonal")
            if argname_axis2 not in kwargs:
                raise ValueError(f"Missing required argument '{argname_axis2}' in diagonal")
            axis1 = kwargs.pop(argname_axis1)
            axis2 = kwargs.pop(argname_axis2)

        shape_in = list(_get_shape(x))
        if axis1 < 0:
            axis1 += len(shape_in)
        if axis2 < 0:
            axis2 += len(shape_in)
        if axis1 < 0 or axis1 >= len(shape_in) or axis2 < 0 or axis2 >= len(shape_in):
            raise ValueError(f"Invalid axis indices for diagonal: {axis1}, {axis2}")

        if shape_in[axis1] != shape_in[axis2]:
            raise ValueError(f"Cannot take diagonal over axes of different length: {shape_in[axis1]} and {shape_in[axis2]}")
        l = shape_in[axis1]

        shape_out = list(shape_in)
        del shape_out[max(axis1, axis2)]
        del shape_out[min(axis1, axis2)]
        shape_out.append(l)
        shape_out = tuple(shape_out)

        if not axis_always_last:
            kwargs = {argname_axis1: axis1, argname_axis2: axis2} | kwargs
        x = op(x, **kwargs)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape_out))

        return x

    return inner


def reduce(op, argname_axis="axis", argname_keepdims="keepdims"):
    @use_name_of(op)
    def inner(x, **kwargs):
        shape = list(_get_shape(x))
        if argname_axis not in kwargs or kwargs[argname_axis] is None:
            axes = list(range(x.ndim))
        elif isinstance(kwargs[argname_axis], int):
            axes = [kwargs[argname_axis]]
        else:
            axes = kwargs[argname_axis]

        if argname_keepdims in kwargs and kwargs[argname_keepdims]:
            for a in axes:
                shape[a] = 1
        else:
            for a in sorted(axes, reverse=True):
                del shape[a]
        shape = tuple(shape)
        x = op(x, **kwargs)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape))
        return x

    return inner


def elementwise(op, num_outputs=1):
    @use_name_of(op)
    def inner(*xs):
        shape = None
        for a in xs:
            if hasattr(a, "shape"):
                if shape is None:
                    shape = a.shape
                else:
                    shape2 = a.shape
                    while len(shape) < len(shape2):
                        shape = (1,) + shape
                    while len(shape2) < len(shape):
                        shape2 = (1,) + shape2
                    shape = np.maximum(shape, shape2)
        if shape is None:
            raise ValueError("elementwise operation requires at least one tensor as argument")

        x = op(*xs)

        if num_outputs == 1:
            output = partial(tracer.signature.classical.Tensor, shape=shape)
        else:
            assert num_outputs > 1
            output = lambda origin: tuple(tracer.signature.classical.Tensor(origin, shape=shape) for _ in range(num_outputs))

        return tracer.cast(x, output)

    return inner


def matmul(op):
    def matmul(x, y):
        if x.ndim != y.ndim:
            raise ValueError(f"matmul requires input tensors to have the same number of dimensions, got {x.ndim} and {y.ndim}")
        if x.ndim < 2 or y.ndim < 2:
            raise ValueError(f"matmul requires input tensors to have at least 2 dimensions, got {x.ndim} and {y.ndim}")
        shape = tuple(np.maximum(x.shape[:-2], y.shape[:-2])) + (x.shape[-2], y.shape[-1])

        x = op(x, y)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape))
        return x

    return matmul


def take(op, argname_axis="axis"):
    @use_name_of(op)
    def inner(x, indices, **kwargs):
        if argname_axis in kwargs:
            axis = kwargs[argname_axis]
        else:
            axis = None
        if axis is None:
            shape = indices.shape
        else:
            axis2 = axis
            if axis2 < 0:
                axis2 += x.ndim
            if axis2 < 0 or axis2 >= x.ndim:
                raise ValueError(f"axis {axis} out of bounds for array of dimension {x.ndim}")
            if hasattr(indices, "shape"):
                indices_shape = tuple(indices.shape)
            else:
                indices_shape = ()

            shape = tuple(x.shape[:axis2]) + indices_shape + tuple(x.shape[axis2 + 1 :])

        x = op(x, indices, **kwargs)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape))
        return x

    return inner


def inplace(op):
    if not isinstance(op, tracer.Tracer):
        raise ValueError("op must be Tracer")

    @use_name_of(op)
    def inner(x, *args, **kwargs):
        return tracer.signature.python.call_inplace(x, op, [x, *args], kwargs)

    return inner


def getitem():
    def getitem(tensor, key):
        if isinstance(key, tuple):
            # Tuple of indices are given -> one for each dimension of tensor
            if len(key) != tensor.ndim:
                raise ValueError(f"Number of indices ({len(key)}) must match the rank of the tensor ({tensor.ndim})")

            in_shape = list(tensor.shape)
            shape = []
            for k in key:
                if isinstance(k, np.integer | int) or (
                    isinstance(k, tracer.signature.classical.Tensor | tracer.signature.classical.ConvertibleTensor) and k.ndim == 0
                ):
                    in_shape = in_shape[1:]
                elif k == slice(None) or k == slice(None, None, -1):
                    shape.append(in_shape[0])
                    in_shape = in_shape[1:]
                elif k is None:
                    shape.append(1)
                else:
                    raise NotImplementedError(f"Key type {type(k)} not supported")
            shape = tuple(shape) + tuple(in_shape)
        else:
            # Single index given -> only supported for 1D tensors
            if tensor.ndim != 1:
                raise ValueError(f"Single index only supported for 1D tensors, got {tensor.ndim}D tensor")
            shape = tuple(key.shape)

        x = tracer.signature.python.getitem(tensor, key)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape))
        return x

    return getitem


def setitem(op=None):
    if op is None:
        op = tracer.signature.python.setitem

    def setitem(x, *args, **kwargs):
        tracer_type = x._tracer_type
        x = op(x, *args, **kwargs)
        x = tracer.cast(x, tracer_type)
        return x

    return setitem


def arange(op):
    @use_name_of(op)
    def inner(n, *args, **kwargs):
        shape = (n,)
        x = op(n, *args, **kwargs)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape))
        return x

    return inner


def concatenate(op, argname_axis="axis"):
    @use_name_of(op)
    def inner(xs, **kwargs):
        if argname_axis in kwargs:
            axis = kwargs[argname_axis]
        else:
            axis = 0
        if axis < 0:
            axis += xs[0].ndim
        if axis < 0 or axis >= xs[0].ndim:
            raise ValueError(f"axis {axis} out of bounds for array of dimension {xs[0].ndim}")

        shape = list(xs[0].shape)
        shape[axis] = sum(x.shape[axis] for x in xs)
        shape = tuple(shape)

        x = op(xs, **kwargs)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape))
        return x

    return inner


def split(op, cumulative, argname_axis="axis"):
    @use_name_of(op)
    def inner(x, arg1, **kwargs):
        if argname_axis in kwargs:
            axis = kwargs[argname_axis]
        else:
            axis = 0
        if axis < 0:
            axis += x.ndim
        if axis < 0 or axis >= x.ndim:
            raise ValueError(f"axis {axis} out of bounds for array of dimension {x.ndim}")

        if isinstance(arg1, int):
            section_length = x.shape[axis] // arg1
            if section_length * arg1 != x.shape[0]:
                raise ValueError(f"array split does not result in an equal division: {x.shape[axis]} % {arg1} != 0")
            lengths = [section_length] * arg1
        else:
            if cumulative:
                indices = [0] + list(arg1) + [x.shape[axis]]
                lengths = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
            else:
                lengths = arg1

        shapes = []
        for length in lengths:
            shape = list(x.shape)
            shape[axis] = int(length)
            shapes.append(tuple(shape))

        xs = op(x, arg1, **kwargs)
        xs = tracer.cast(xs, lambda origin: [tracer.signature.classical.Tensor(origin, shape=shape) for shape in shapes])
        return xs

    return inner


def dot(op):
    @use_name_of(op)
    def inner(x, y):
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError(f"dot only supports 1D tensors, got {x.ndim}D and {y.ndim}D")

        x = op(x, y)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=()))
        return x

    return inner


def tensordot(op, argname_axes="axes"):
    @use_name_of(op)
    def inner(a, b, **kwargs):
        if argname_axes in kwargs:
            axes = kwargs[argname_axes]
        else:
            raise ValueError(f"Missing required argument '{argname_axes}' in tensordot")

        if isinstance(axes, int):
            axes_a = list(range(a.ndim - axes, a.ndim))
            axes_b = list(range(axes))
        elif isinstance(axes, tuple | list) and len(axes) == 2:
            axes_a, axes_b = axes
            if isinstance(axes_a, int):
                axes_a = [axes_a]
            if isinstance(axes_b, int):
                axes_b = [axes_b]
        else:
            raise ValueError(f"Invalid axes argument for tensordot: {axes}")
        if len(axes_a) != len(axes_b):
            raise ValueError(f"Axes lengths do not match: {len(axes_a)} != {len(axes_b)}")

        def canonicalize_axis(axis, ndim):
            if axis < 0:
                axis += ndim
            if axis < 0 or axis >= ndim:
                raise ValueError("Invalid axis index in tensordot")
            return axis

        axes_a = [canonicalize_axis(axis, a.ndim) for axis in axes_a]
        axes_b = [canonicalize_axis(axis, b.ndim) for axis in axes_b]

        a_remain = [i for i in range(a.ndim) if i not in axes_a]
        b_remain = [i for i in range(b.ndim) if i not in axes_b]
        shape = tuple([a.shape[i] for i in a_remain] + [b.shape[i] for i in b_remain])

        x = op(a, b, **kwargs)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape))
        return x

    return inner


def einsum(op):
    @use_name_of(op)
    def inner(subscripts, *tensors):
        exprs = subscripts.split("->")[0].split(",")
        if len(exprs) != len(tensors):
            raise ValueError(f"Expected {len(exprs)} tensors, got {len(tensors)}")
        values = {}
        for i, (expr, tensor) in enumerate(zip(exprs, tensors, strict=False)):
            expr = expr.strip().replace(" ", "")
            if len(expr) != len(tensor.shape):
                raise ValueError(f"Expected {len(expr)} axes, got {len(tensor.shape)} for {i}-th (zero-based) input tensor")
            for axis, value in zip(expr, tensor.shape, strict=False):
                if axis in values:
                    if values[axis] != value:
                        raise ValueError(f"Got conflicting values for axis {axis}: {values[axis]} and {value}")
                else:
                    values[axis] = value
        expr_out = subscripts.split("->")[-1].strip().replace(" ", "")
        shape_out = tuple(values[axis] for axis in expr_out)

        x = op(subscripts, *tensors)
        x = tracer.cast(x, partial(tracer.signature.classical.Tensor, shape=shape_out))
        return x

    return inner


def _to_shapes_inner(axes, shapes_outer):
    shapes_inner = []
    n = None
    for axis, shape_outer in zip(axes, shapes_outer, strict=False):
        shape_inner = list(shape_outer)
        if axis is not None:
            if axis < 0:
                axis += len(shape_outer)
            if axis < 0 or axis >= len(shape_outer):
                raise ValueError("Invalid axis index in vmap")
            if n is None:
                n = shape_inner[axis]
            elif n != shape_inner[axis]:
                raise ValueError("Inconsistent axis sizes in vmap")
            del shape_inner[axis]
        shapes_inner.append(shape_inner)
    return shapes_inner, n


def _to_shapes_outer(axes, shapes_inner, n):
    shapes_outer = []
    for axis, shape_inner in zip(axes, shapes_inner, strict=False):
        shape_outer = list(shape_inner)
        if axis is not None:
            if axis < 0:
                axis += len(shape_inner) + 1
            if axis < 0 or axis >= len(shape_inner) + 1:
                raise ValueError("Invalid axis index in vmap")
            shape_outer.insert(axis, n)
        shapes_outer.append(shape_outer)
    return shapes_outer


def vmap(original_vmap):
    @use_name_of(original_vmap)
    def vmap_with_shapes(op, in_axes=0, out_axes=0):
        if isinstance(in_axes, int | np.integer):
            in_axes = (in_axes,)
        if isinstance(out_axes, int | np.integer):
            out_axes = (out_axes,)
        if not isinstance(in_axes, tuple):
            raise ValueError(f"Expected in_axes to be a tuple or int, got {type(in_axes)}")
        if not isinstance(out_axes, tuple):
            raise ValueError(f"Expected out_axes to be a tuple or int, got {type(out_axes)}")

        def vmapped_op_with_shapes(*tensors):
            if len(tensors) != len(in_axes):
                raise ValueError(f"Expected {len(in_axes)} arguments in vmapped function, got {len(tensors)}")

            # Get vmapped input shapes
            in_shapes_outer = [t.shape for t in tensors]
            in_shapes_inner, vmapped_axis_len = _to_shapes_inner(in_axes, in_shapes_outer)

            # Create inner graph
            in_tracers_inner = [tracer.signature.classical.Tensor(None, shape) for shape in in_shapes_inner]
            out_tracers_inner = op(*in_tracers_inner)
            if not isinstance(out_tracers_inner, tracer.signature.classical.Tensor) and not (
                isinstance(out_tracers_inner, tuple) and all(isinstance(t, tracer.signature.classical.Tensor) for t in out_tracers_inner)
            ):
                raise ValueError(f"Expected vmapped function to return a tensor or tuple of tensor, got {pytree.map(type, out_tracers_inner)}")
            graph = tracer.Graph(in_tracers_inner, out_tracers_inner)

            # Get vmapped output shapes
            if isinstance(out_tracers_inner, tracer.Tracer):
                out_tracers_inner = [out_tracers_inner]
            out_shapes_inner = [t.shape for t in out_tracers_inner]
            out_shapes_outer = _to_shapes_outer(out_axes, out_shapes_inner, vmapped_axis_len)

            # Run vmapped function (return value is a simple tracer, since original_vmap is a simple tracer)
            tensors = original_vmap(graph, in_axes=in_axes if len(in_axes) > 1 else in_axes[0], out_axes=out_axes if len(out_axes) > 1 else out_axes[0])(
                *tensors
            )

            # Cast return values to the expected tensor types
            if len(out_axes) == 1:
                return tracer.cast(tensors, partial(tracer.signature.classical.Tensor, shape=out_shapes_outer[0]))
            else:
                return tracer.cast(tensors, lambda origin: tuple(tracer.signature.classical.Tensor(origin, shape=shape) for shape in out_shapes_outer))

        return vmapped_op_with_shapes

    return vmap_with_shapes


class at:
    def __init__(self, x, indices):
        self._x = x
        self._indices = indices
        self.set = _at_updater(self, "set")
        self.add = _at_updater(self, "add")
        self.subtract = _at_updater(self, "subtract")


def _at_updater(self, op):
    @use_name_of(op)
    def inner(updates):
        x = tracer.signature.python.getattr(tracer.signature.python.getattr(self._x, "at")[self._indices], op)(updates)
        x = tracer.cast(x, self._x._tracer_type)
        return x

    return inner

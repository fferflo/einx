import einx._src.namedtensor.stage3 as stage3
import numpy as np
import functools
import einx._src.tracer as tracer
import einx._src.util.pytree as pytree
from einx._src.util.functools import use_name_of
from einx._src.frontend.errors import OperationNotSupportedError


def _axis_to_axisint(axis, name="axis"):
    if isinstance(axis, int | np.integer):
        return axis
    elif isinstance(axis, list | tuple | np.ndarray):
        if len(axis) != 1:
            raise ValueError(f"Expected {name} to be a single integer or a list/tuple/array of length 1, but got {axis}")
        return axis[0]
    else:
        raise ValueError(f"Expected {name} to be an integer or a list/tuple/array, but got {type(axis)}")


def _axis_to_axistuple(axis, name="axis"):
    if isinstance(axis, int | np.integer):
        return (axis,)
    elif isinstance(axis, list | tuple | np.ndarray):
        return tuple(axis)
    else:
        raise ValueError(f"Expected {name} to be an integer or a list/tuple/array, but got {type(axis)}")


def _associative_binary_to_nary(binary_op):
    def nary_op(*args):
        x = args[0]
        for y in args[1:]:
            x = binary_op(x, y)
        return x

    return nary_op


def _unsqueeze(classical, tensor, axis):
    if axis < 0:
        axis += tensor.ndim + 1
    if axis < 0 or axis > tensor.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for tensor with {tensor.ndim} dimensions.")

    new_shape = list(tensor.shape)
    new_shape.insert(axis, 1)
    return classical.reshape(tensor, new_shape)


def _stack(classical, tensors, axis):
    if axis < 0:
        axis += len(tensors[0].shape) + 1
    if axis < 0 or axis > len(tensors[0].shape):
        raise ValueError(f"Axis {axis} is out of bounds for tensors with {len(tensors[0].shape)} dimensions.")

    tensors = [_unsqueeze(classical, tensor, axis) for tensor in tensors]
    return classical.concatenate(tensors, axis=axis)


def _squeeze_transpose_broadcast(classical, expr_in, tensor, expr_out, broadcast_to_unitary=False):
    # Squeeze axes if necessary
    squeezable_in_axes = [a.name for a in expr_in.nodes() if isinstance(a, stage3.Axis) and a.value == 1]
    out_axes = [a.name for a in expr_out.nodes() if isinstance(a, stage3.Axis)]
    squeeze_axes = set(squeezable_in_axes) - set(out_axes)
    if len(squeeze_axes) > 0:
        expr_in = stage3.remove(expr_in, lambda a: isinstance(a, stage3.Axis) and a.name in squeeze_axes, keep_children=False)
        tensor = classical.reshape(tensor, expr_in.shape)

    # Transpose axes if necessary
    def _to_axis_ids(expr):
        counts = {}
        axes = []
        for a in expr.nodes():
            if isinstance(a, stage3.Axis):
                c = counts.get(a.name, 0)
                axes.append((a.name, c))
                counts[a.name] = c + 1
        return axes

    in_axes = _to_axis_ids(expr_in)
    out_axes = _to_axis_ids(expr_out)
    out_axes_intersect = [a for a in out_axes if a in in_axes]
    if set(out_axes_intersect) != set(in_axes):
        invalid_axes = set(in_axes) - set(out_axes_intersect)
        invalid_axes = {name for name, count in invalid_axes}
        if len(invalid_axes) == 1:
            invalid_axes = f"axis {invalid_axes.pop()} does"
        else:
            invalid_axes = f"axes {', '.join(invalid_axes)} do"
        raise ValueError(f"The input {invalid_axes} not appear in the corresponding output expression.")

    perm = [in_axes.index(out_axis) for out_axis in out_axes_intersect]
    tensor = classical.transpose(tensor, tuple(perm))

    # Expand and broadcast missing output dimensions if necessary
    in_axes = [a.name for a in expr_in.nodes() if isinstance(a, stage3.Axis)]
    out_axes = [a.name for a in expr_out.nodes() if isinstance(a, stage3.Axis)]
    out_axes_broadcast = [a for a in out_axes if a not in in_axes]
    if len(out_axes_broadcast) > 0:
        pre_broadcast_shape = tuple(1 if a.name in out_axes_broadcast else a.value for a in expr_out.nodes() if isinstance(a, stage3.Axis))
        tensor = classical.reshape(tensor, pre_broadcast_shape)
        if not broadcast_to_unitary:
            tensor = classical.broadcast_to(tensor, expr_out.shape)

    if broadcast_to_unitary:
        expr_out = stage3.List.create([(axis if axis.name in in_axes else stage3.Axis.new_unnamed(1)) for axis in expr_out])
    return expr_out, tensor


def _squeeze_shape(shape):
    return tuple(s for s in shape if s != 1)


def _squeeze_unsqueeze(classical, tensor, shape):
    if _squeeze_shape(tensor.shape) != _squeeze_shape(shape):
        raise ValueError(f"Expected tensor with shape {tensor.shape} to have the same squeezed shape as {shape}")
    return classical.reshape(tensor, shape)


def _unravel(classical, tensor, ravel_shape, axis):
    # 1D does not need to be unravelled
    if len(ravel_shape) == 1:
        if axis is None:
            return tensor
        else:
            return _unsqueeze(classical, tensor, axis)

    out_indices = [None] * len(ravel_shape)
    for i, s in reversed(list(enumerate(ravel_shape))):
        tensor, out_indices[i] = classical.divmod(tensor, s)

    out_indices = _stack(classical, out_indices, axis=axis)

    return out_indices


def _to_ord_str(i):
    if i == 0:
        return "1st"
    elif i == 1:
        return "2nd"
    elif i == 2:
        return "3rd"
    else:
        return f"{i + 1}th"


def _ensure_output(op, expected_out_shapes, expected_type=None, allow_squeeze_unsqueeze=False, classical=None):
    if allow_squeeze_unsqueeze:
        if classical is None:
            raise ValueError("classical must be provided if allow_squeeze_unsqueeze=True")
    expected_type2 = expected_type

    @use_name_of(op)
    def inner(*args, **kwargs):
        expected_type = expected_type2
        if callable(expected_type) and not isinstance(expected_type, type) and not isinstance(expected_type, tracer.Tracer):
            expected_type = expected_type()

        tensors_out = op(*args, **kwargs)
        if isinstance(tensors_out, tracer.Tracer):
            # Create list of tracers
            if len(expected_out_shapes) == 1:
                tensors_out = [tensors_out]
            else:
                tensors_out = tracer.signature.python.assert_(
                    tensors_out,
                    tracer.signature.python.builtins.isinstance(tensors_out, tracer.signature.python.builtins.tuple),
                    f"Expected the adapted function to return a tuple of length {len(expected_out_shapes)}",
                )
                tensors_out = tracer.signature.python.assert_(
                    tensors_out,
                    tracer.signature.python.equal(tracer.signature.python.builtins.len(tensors_out), len(expected_out_shapes)),
                    f"Expected the adapted function to return a tuple of length {len(expected_out_shapes)}",
                )
                tensors_out = tracer.cast(tensors_out, lambda origin: [tracer.signature.python.Value(origin) for _ in range(len(expected_out_shapes))])
        elif isinstance(tensors_out, tuple) and all(isinstance(t, tracer.Tracer) for t in tensors_out):
            # Return value already is a tuple of tracers
            if len(tensors_out) != len(expected_out_shapes):
                raise ValueError(f"Expected the adapted function to return a tuple of length {len(expected_out_shapes)}, but got length {len(tensors_out)}")
        else:
            raise ValueError(f"Expected the adapted function to return a tracer or a tuple of tracers, but got {pytree.map(type, tensors_out)}")

        tensors_out2 = []
        for i, (tensor, expected_out_shape) in enumerate(zip(tensors_out, expected_out_shapes, strict=False)):
            if isinstance(tensor, tracer.signature.classical.Tensor | tracer.signature.classical.ConvertibleTensor):
                # Return type is a tensor -> ensure that the static shape is correct
                t = _squeeze_shape if allow_squeeze_unsqueeze else tuple
                if t(tensor.shape) != t(expected_out_shape):
                    raise ValueError(
                        f"Expected {_to_ord_str(i)} return value of the adapted function to be a tensor with shape {expected_out_shape}, "
                        f"but got shape {tensor.shape}"
                    )
                if allow_squeeze_unsqueeze:
                    tensor = _squeeze_unsqueeze(classical, tensor, expected_out_shape)
            else:
                # Return type is a general tracer object -> ensure that it has the correct type and that the shape is correct at runtime.
                # Then cast to expected shape
                if expected_type is not None:
                    tensor = tracer.signature.python.assert_(
                        tensor,
                        tracer.signature.python.builtins.isinstance(tensor, expected_type),
                        f"Expected {_to_ord_str(i)} return value of the adapted function to be a tensor",
                    )
                if allow_squeeze_unsqueeze:
                    raise ValueError(
                        "allow_squeeze_unsqueeze=True cannot currently be used if return type of the adapted function is a general tracer object, not a tensor."
                    )
                tensor = tracer.signature.python.assert_(
                    tensor,
                    tracer.signature.python.equal(tracer.signature.python.builtins.tuple(tensor.shape), expected_out_shape),
                    f"Expected {_to_ord_str(i)} return value of the adapted function to be a tensor with shape {expected_out_shape}",
                )
                tensor = tracer.cast(tensor, lambda origin: tracer.signature.classical.Tensor(origin, shape=expected_out_shape))
            tensors_out2.append(tensor)
        tensors_out = tensors_out2

        if len(tensors_out) == 1:
            return tensors_out[0]
        else:
            return tuple(tensors_out)

    return inner


def _make_concrete_types(concrete_types):
    concrete_types2 = []
    for t in concrete_types:
        if isinstance(t, str):
            if t == "scalar":
                concrete_types2.extend([int, float, bool, np.integer, np.floating, np.bool_])
            elif t == "numpy":
                concrete_types2.append(np.ndarray)
            else:
                raise ValueError(f"Unknown concrete type string: {t}")
        elif isinstance(t, tracer.Tracer):
            pass
        else:
            assert isinstance(t, type)
            concrete_types2.append(t)
    return tuple(concrete_types2)


def _has_type(x, types):
    types = _make_concrete_types(types)
    return isinstance(x, types) or (isinstance(x, tracer.signature.classical.ConvertibleTensor) and issubclass(x.concrete.type, types))


def _is_scalar(x):
    return isinstance(x, int | float | bool | np.integer | np.floating | np.bool_)


def _get_shape(x):
    if _is_scalar(x):
        return ()
    elif isinstance(x, np.ndarray):
        return x.shape
    elif isinstance(x, tracer.signature.classical.Tensor | tracer.signature.classical.ConvertibleTensor):
        return x.shape
    else:
        raise ValueError(f"Cannot determine shape of object with type {type(x)}.")


def _to_tensor(to_tensor, forward, convert):
    _to_tensor = to_tensor
    forward = _make_concrete_types(forward)
    convert = _make_concrete_types(convert)

    def to_tensor(x):
        if isinstance(x, (*forward, tracer.signature.classical.Tensor)):
            return x
        elif isinstance(x, tracer.signature.classical.ConvertibleTensor):
            if issubclass(x.concrete.type, forward):
                return x
            elif issubclass(x.concrete.type, convert):
                shape = x.shape
                x = _to_tensor(x)
                x = tracer.cast(x, lambda origin: tracer.signature.classical.Tensor(origin, shape=shape))
                return x
            else:
                raise ValueError(f"An object of type {x.concrete.type} cannot be used as a tensor")
        elif isinstance(x, (*convert,)):
            shape = _get_shape(x)
            x = _to_tensor(x)
            x = tracer.cast(x, lambda origin: tracer.signature.classical.Tensor(origin, shape=shape))
            return x
        else:
            raise ValueError(f"Expected a tensor, but got {type(x)}")

    return to_tensor


def _unsupported_op(name, backend, message=None):
    def op(*args, **kwargs):
        message2 = f"{name} operation is not supported by the {backend} backend."
        if message is not None:
            message2 += " " + message
        raise OperationNotSupportedError(message2)

    op.__name__ = name
    return op

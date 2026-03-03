import einx._src.namedtensor.stage3 as stage3
import einx._src.tracer as tracer
import functools
import einx._src.util.pytree as pytree
from einx._src.namedtensor import NamedTensor
from einx._src.util.functools import use_name_of
from .._util import _to_tensor


def _to_ord_str(i):
    if i == 0:
        return "1st"
    elif i == 1:
        return "2nd"
    elif i == 2:
        return "3rd"
    else:
        return f"{i + 1}th"


def expr_to_ftdims(expr, axisname_to_ftdim, runtime_axisname_to_ftdims=None):
    if isinstance(expr, stage3.List):
        return tuple(ftdim for child in expr.children for ftdim in expr_to_ftdims(child, axisname_to_ftdim, runtime_axisname_to_ftdims))
    elif isinstance(expr, stage3.Axis):
        if expr.name in axisname_to_ftdim:
            return (axisname_to_ftdim[expr.name],)
        else:
            assert runtime_axisname_to_ftdims is not None
            return (runtime_axisname_to_ftdims[expr.name],)
    elif isinstance(expr, stage3.ConcatenatedAxis):
        raise ValueError("functorchdim does not support axis concatenation.")
    elif isinstance(expr, stage3.FlattenedAxis):
        return (expr_to_ftdims(expr.inner, axisname_to_ftdim, runtime_axisname_to_ftdims),)
    elif isinstance(expr, stage3.Brackets):
        return expr_to_ftdims(expr.inner, axisname_to_ftdim, runtime_axisname_to_ftdims)
    else:
        raise ValueError(f"Unexpected expression type: {type(expr)}")


def op(op, torch, functorchdim, get_device):
    @use_name_of(op)
    def inner(*tensors, out, **kwargs):
        if not isinstance(out, list | tuple):
            out = [out]

        to_tensor = _to_tensor(lambda x: torch.asarray(x, device=get_device()), forward=[torch.Tensor], convert=["numpy", "scalar"])
        tensors = [NamedTensor(to_tensor(tensor.value), tensor.expr) for tensor in tensors]

        # Convert classical tensors to functorchdim tensors
        axisname_to_value = {axis.name: axis.value for tensor in tensors for axis in tensor.expr.nodes() if isinstance(axis, stage3.Axis)}
        axisname_to_ftdim = {name: functorchdim.Dim(name, value) for name, value in axisname_to_value.items()}

        fttensors = []
        for tensor in tensors:
            axisnames = [axis.name for axis in tensor.expr.nodes() if isinstance(axis, stage3.Axis)]
            ftdims = expr_to_ftdims(tensor.expr, axisname_to_ftdim)
            shape = {axisname: axisname_to_value[axisname] for axisname in axisnames}

            fttensor = tracer.signature.python.getitem(tensor.value, ftdims)
            fttensor = tracer.cast(fttensor, lambda origin: tracer.signature.functorchdim.Tensor(origin, shape=shape))
            fttensors.append(fttensor)

        # Call the operation with functorchdim tensors
        axes = {axis.name for tensor in tensors for axis in tensor.expr.nodes() if isinstance(axis, stage3.Axis) if stage3.is_in_brackets(axis)}
        axes = {name: axisname_to_ftdim[name] for name in axes}
        axes = tuple(
            tuple(axes[axis.name] for axis in tensor.expr.nodes() if isinstance(axis, stage3.Axis) if stage3.is_in_brackets(axis)) for tensor in tensors
        )
        fttensors = op(*fttensors, axes=axes, **kwargs)

        # Cast return value back to classical tensors
        if isinstance(fttensors, tracer.Tracer):
            # Create list of tracers
            if len(out) == 1:
                fttensors = [fttensors]
            else:
                fttensors = tracer.signature.python.assert_(
                    fttensors,
                    tracer.signature.python.builtins.isinstance(fttensors, tracer.signature.python.builtins.tuple),
                    f"Expected the adapted function to return a tuple of length {len(out)}",
                )
                fttensors = tracer.signature.python.assert_(
                    fttensors,
                    tracer.signature.python.equal(tracer.signature.python.builtins.len(fttensors), len(out)),
                    f"Expected the adapted function to return a tuple of length {len(out)}",
                )
                fttensors = tracer.cast(fttensors, lambda origin: [tracer.signature.python.Value(origin) for _ in range(len(out))])
        elif isinstance(fttensors, tuple) and all(isinstance(t, tracer.signature.Tracer) for t in fttensors):
            # Return value already is a tuple of tracers
            if len(fttensors) != len(out):
                raise ValueError(f"Expected the adapted function to return a tuple of length {len(out)}, but got length {len(fttensors)}")
        else:
            raise ValueError(f"Expected the adapted function to return a tracer or a tuple of tracers, but got {pytree.map(type, fttensors)}")

        tensors = []
        for i, (fttensor, expr) in enumerate(zip(fttensors, out, strict=False)):
            axisnames = [axis.name for axis in expr.nodes() if isinstance(axis, stage3.Axis)]
            expected_shape = {axisname: axisname_to_value[axisname] for axisname in axisnames}

            if isinstance(fttensor, tracer.signature.functorchdim.Tensor):
                # Return type is a tensor -> ensure that the static shape is correct
                if t(fttensor.shape) != t(expected_shape):
                    raise ValueError(
                        f"Expected {_to_ord_str(i)} return value of the adapted function to be a tensor with shape {expected_shape}, "
                        f"but got shape {fttensor.shape}"
                    )
            else:
                # Return type is a general tracer object -> ensure that it has the correct type and that the shape is correct at runtime.
                # Then cast to expected shape
                fttensor = tracer.signature.python.assert_(
                    fttensor,
                    tracer.signature.python.builtins.isinstance(fttensor, functorchdim.Tensor),
                    f"Expected {_to_ord_str(i)} return value of the adapted function to be a tensor",
                )
                runtime_shape = tracer.signature.python.builtins.dict(
                    tracer.signature.python.builtins.map(
                        tracer.signature.python.function(lambda dim: (tracer.signature.python.builtins.repr(dim), dim.size)), fttensor.dims
                    )
                )
                fttensor = tracer.signature.python.assert_(
                    fttensor,
                    tracer.signature.python.equal(runtime_shape, expected_shape),
                    f"Expected {_to_ord_str(i)} return value of the adapted function to be a tensor with shape {expected_shape}",
                )
                fttensor = tracer.cast(fttensor, lambda origin: tracer.signature.functorchdim.Tensor(origin, shape=expected_shape))

            runtime_axisname_to_ftdims = tracer.signature.python.builtins.dict(
                tracer.signature.python.builtins.map(tracer.signature.python.function(lambda dim: (tracer.signature.python.builtins.repr, dim)), fttensor.dims)
            )
            ftdims = expr_to_ftdims(expr, axisname_to_ftdim, runtime_axisname_to_ftdims)  # Prefer statically available ftdims

            tensor = fttensor.order(*ftdims)
            tensor = tracer.cast(tensor, lambda origin: tracer.signature.classical.Tensor(origin, shape=expr.shape))
            tensors.append(NamedTensor(tensor, expr))

        if len(tensors) == 1:
            return tensors[0]
        else:
            return tuple(tensors)

    return inner

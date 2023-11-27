import einx
from . import util
import numpy as np
from functools import partial

_op_names = ["sum", "mean", "var", "std", "prod", "count_nonzero", "any", "all", "max", "min"]
_all = all # Is overwritten below
_any = any

@einx.lru_cache(trace=lambda k: k[0] in [1, "tensors_in"])
def reduce_stage3(exprs_in, tensors_in, exprs_out, op, backend=None):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    if op is None:
        raise TypeError("op cannot be None")
    if isinstance(op, str):
        op = getattr(backend, op)
    else:
        op = partial(backend.reduce, op=op)
    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}")
    if _any(isinstance(expr, einx.expr.stage3.Marker) for root in exprs_out for expr in root.all()):
        raise ValueError(f"Marker '{expr}' in output expression is not allowed")

    # Call tensor factories
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend) for tensor, expr in zip(tensors_in, exprs_in)]

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend)
    exprs_out_flat = util.flatten(exprs_out)
    assert _all(einx.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert _all(einx.expr.stage3.is_flat(expr) for expr in exprs_out_flat)
    if len(exprs_in) != len(exprs_out_flat):
        raise ValueError("Got different number of input and output expressions (after flattening)")

    # Reduce input dimensions
    exprs_in2 = []
    tensors_in2 = []
    any_reduced = False
    for expr_in, tensor_in in zip(exprs_in, tensors_in):
        # Find reduced axes
        reduced_axis_indices = tuple(i for i, axis in enumerate(expr_in) if einx.expr.stage3.is_marked(axis))
        any_reduced = any_reduced or len(reduced_axis_indices) > 0
        if len(reduced_axis_indices) > 0:
            # Apply reduction
            tensor_in = op(tensor_in, axis=reduced_axis_indices)
            expr_in = einx.expr.stage3.remove(expr_in, lambda expr: isinstance(expr, einx.expr.stage3.Marker))
        exprs_in2.append(expr_in)
        tensors_in2.append(tensor_in)
    exprs_in = exprs_in2
    tensors_in = tensors_in2
    if not any_reduced:
        raise ValueError("No (non-trivial) axes are reduced")

    # Order inputs to align with output expressions
    indices = util.assignment(exprs_in, exprs_out_flat)
    exprs_in = [exprs_in[i] for i in indices]
    tensors_in = [tensors_in[i] for i in indices]

    # Transpose and broadcast missing output dimensions
    tensors = [util.transpose_broadcast(expr_in, tensor, expr_out) for expr_in, tensor, expr_out in zip(exprs_in, tensors_in, exprs_out_flat)]

    # Unflatten output expressions
    tensors = util.unflatten(exprs_out_flat, tensors, exprs_out, backend)

    return tensors, exprs_out

def parse(description, *tensors_shapes, keepdims=None, cse=True, **parameters):
    description, parameters = einx.expr.util._clean_description_and_parameters(description, parameters)

    if "->" in description:
        if not keepdims is None:
            raise ValueError("keepdims cannot be given when using '->'")
        description = description.split("->")
        if len(description) != 2:
            raise ValueError("Operation cannot contain more than one '->'")

        exprs_in, exprs_out = description
        exprs_in = exprs_in.split(",")
        exprs_out = exprs_out.split(",")
        exprs = exprs_in + exprs_out
        if len(exprs_in) != len(tensors_shapes):
            raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensors_shapes)}")

        exprs = einx.expr.solve(
                [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensors_shapes)] \
              + [einx.expr.Condition(expr=expr_out, depth=0) for expr_out in exprs_out] \
              + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
            cse=cse,
        )[:len(exprs_in) + len(exprs_out)]
        exprs_in, exprs_out = exprs[:len(exprs_in)], exprs[len(exprs_in):]

        # If no axes are marked for reduction in exprs_in, mark all axes that don't appear in exprs_out
        if not _any(einx.expr.stage3.is_marked(axis) for expr_in in exprs_in for axis in expr_in.all()):
            axes_names_out = set(axis.name for expr in exprs_out for axis in expr.all() if isinstance(axis, einx.expr.stage3.Axis))
            exprs_in = [einx.expr.stage3.mark(expr, lambda expr: isinstance(expr, einx.expr.stage3.Axis) and expr.name not in axes_names_out) for expr in exprs_in]

    else:
        exprs_in = description.split(",")
        if len(exprs_in) != 1:
            raise ValueError("Operation with implicit output shape cannot contain more than one input expression")
        if len(exprs_in) != len(tensors_shapes):
            raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_shapes)}")

        exprs_in = einx.expr.solve(
                [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensors_shapes)] \
              + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
            cse=cse,
        )[:len(exprs_in)]

        if not _any(isinstance(expr, einx.expr.stage3.Marker) for root in exprs_in for expr in root.all()):
            raise ValueError("No axes were marked for reduction")

        # Determine output expressions by removing markers from input expressions
        def replace(expr):
            if isinstance(expr, einx.expr.stage3.Marker):
                if keepdims:
                    return [einx.expr.stage3.Axis(None, 1)]
                else:
                    return []
        exprs_out = [einx.expr.stage3.replace(expr_in, replace) for expr_in in exprs_in]

    return exprs_in, exprs_out

@einx.lru_cache(trace=lambda k: isinstance(k[0], int) and k[0] >= 1)
def reduce_stage0(description, *tensors, op, keepdims=None, backend=None, cse=True, **parameters):
    exprs_in, exprs_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], keepdims=keepdims, cse=cse, **parameters)
    tensors, exprs_out = reduce_stage3(exprs_in, tensors, exprs_out, op=op, backend=backend)
    return tensors[0] if len(exprs_out) == 1 else tensors

def reduce(arg0, *args, **kwargs):
    if isinstance(arg0, str) or (isinstance(arg0, tuple) and isinstance(arg0[0], str)):
        return reduce_stage0(arg0, *args, **kwargs)
    else:
        return reduce_stage3(arg0, *args, **kwargs)
reduce._op_names = _op_names
reduce.parse = parse


def _make(name):
    def func(*args, **kwargs):
        return reduce(*args, op=name, **kwargs)
    func.__name__ = name
    globals()[name] = func

for name in ["sum", "mean", "var", "std", "prod", "count_nonzero", "any", "all", "max", "min"]:
    _make(name)
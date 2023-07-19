import einx
from .tensor import Tensor
import numpy as np

@einx.lru_cache
def _make_op(exprs_in, expr_out):
    if len(exprs_in) < 1:
        raise ValueError("Must have at least one input tensor")
    exprs = list(exprs_in) + [expr_out]
    isolated_axes = einx.expr.get_isolated_axes(exprs)

    ops = []

    # Reshape nested input to flat input
    shapes = [tuple(einx.expr.get_flattened_shape(expr)) for expr in exprs]

    # Squeeze input dimensions
    shapes2 = [s for s in shapes]
    for i in range(len(exprs_in)):
        if len(isolated_axes[i]) > 0:
            if any(v.value > 1 for v in isolated_axes[i]):
                raise ValueError(f"Non-trivial isolated input axes {[str(v) for v in isolated_axes[i]]} are not allowed")
            shapes2[i] = tuple(v.value for v in einx.expr.get_flattened_axes(exprs_in[i]) if not v in isolated_axes[i])
            assert len(shapes2[i]) != len(shapes[i])
    shapes = shapes2
    shapes2 = [(shape if shape != expr_in.shape else None) for shape, expr_in in zip(shapes, exprs_in)]
    ops.append(lambda xs, backend, shapes=shapes2: [(backend.reshape(x, shape) if not shape is None else x) for x, shape in zip(xs, shapes)])

    # Apply einsum
    einsum_variables = {}
    def get_einsum_variable(key):
        if key in einsum_variables:
            return einsum_variables[key]
        else:
            v = chr(ord("a") + len(einsum_variables))
            if ord(v) > ord("z"):
                raise ValueError(f"Only supports up to {ord('z') - ord('a') + 1} input tensors")
            einsum_variables[key] = v
            return v

    einsum_str = ""
    for i in range(len(exprs)):
        for variable in [v for v in einx.expr.get_flattened_axes(exprs[i]) if not v in isolated_axes[i]]:
            einsum_str += get_einsum_variable(variable) + " "
        if i < len(exprs) - 2:
            einsum_str += ", "
        elif i == len(exprs) - 2:
            einsum_str += " -> "

    ops.append(lambda xs, backend, einsum_str=einsum_str: backend.einsum(einsum_str, *xs))

    # Expand and broadcast missing output dimensions
    if len(isolated_axes[-1]) > 0:
        shape = tuple(1 if v in isolated_axes[-1] else v.value for v in einx.expr.get_flattened_axes(expr_out))
        ops.append(lambda x, backend, shape=shape: backend.reshape(x, shape))
        shape2 = tuple(einx.expr.get_flattened_shape(expr_out))
        if shape != shape2:
            ops.append(lambda x, backend, shape=shape2: backend.broadcast_to(x, shape))
            shape = shape2
    else:
        shape = tuple(v.value for v in einx.expr.get_flattened_axes(expr_out))

    # Reshape flat output to nested output
    if shape != expr_out.shape:
        ops.append(lambda x, backend, shape=expr_out.shape: backend.reshape(x, shape))

    def tensor_op(*x, backend, ops=ops):
        for op in ops:
            x = op(x, backend)
        return x
    return tensor_op

def dot(tensors_in, expr_out, backend=None):
    if backend is None:
        backend = einx.backend.get([t.value for t in tensors_in])
    op = _make_op([t.expr for t in tensors_in], expr_out)
    value_out = op(*[t.value for t in tensors_in], backend=backend)
    return Tensor(value_out, expr_out, backend=backend)
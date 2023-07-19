import einx
import numpy as np
from .tensor import Tensor
from functools import partial

@einx.lru_cache
def _make_op(expr_in, expr_out):
    isolated_axes = einx.expr.get_isolated_axes([expr_in, expr_out])
    if len(isolated_axes[0]) == 0:
        raise ValueError("No axes are reduced")

    ops = []

    # Reshape nested input to flat input
    shape = tuple(einx.expr.get_flattened_shape(expr_in))
    if tuple(expr_in.shape) != shape:
        ops.append(lambda x, backend, op, shape=shape: backend.reshape(x, shape))

    # Reduce input dimensions
    axis = tuple(j for j, v in enumerate(einx.expr.get_flattened_axes(expr_in)) if v in isolated_axes[0])
    assert len(axis) > 0
    ops.append(lambda x, backend, op, axis=axis: op(x, axis))
    shape = tuple(s for j, s in enumerate(shape) if j not in axis)

    # Transpose to flat output
    in_variables = [v for v in einx.expr.get_flattened_axes(expr_in) if not v in isolated_axes[0]]
    out_variables = [v for v in einx.expr.get_flattened_axes(expr_out) if not v in isolated_axes[1]]
    assert len(in_variables) == len(shape)
    assert len(out_variables) == len(shape)
    perm = [in_variables.index(out_variable) for out_variable in out_variables]
    if perm != list(range(len(perm))):
        shape = [shape[i] for i in perm]
        ops.append(lambda x, backend, op, perm=perm: backend.transpose(x, perm))

    # Expand and broadcast missing output dimensions
    if len(isolated_axes[1]) > 0:
        shape = tuple(1 if v in isolated_axes[1] else v.value for v in einx.expr.get_flattened_axes(expr_out))
        ops.append(lambda x, backend, op, shape=shape: backend.reshape(x, shape))
        broadcast_shape = tuple(einx.expr.get_flattened_shape(expr_out))
        if np.any(shape != broadcast_shape):
            ops.append(lambda x, backend, op, shape=broadcast_shape: backend.broadcast_to(x, shape))
        shape = broadcast_shape

    # Reshape flat output to nested output
    if shape != expr_out.shape:
        ops.append(lambda x, backend, op, shape=expr_out.shape: backend.reshape(x, shape))

    def tensor_op(x, op, backend, ops=ops):
        for op_step in ops:
            x = op_step(x, backend, op)
        return x
    return tensor_op

def reduce(tensor_in, expr_out, op, backend=None):
    if op is None:
        raise ValueError("op cannot be None")
    if isinstance(op, str):
        op = vars(tensor_in.backend)[op]
    if backend is None:
        backend = tensor_in.backend
    tensor_op = _make_op(tensor_in.expr, expr_out)
    value_out = tensor_op(tensor_in.value, op=op, backend=backend)
    return Tensor(value_out, expr_out, backend=backend)

def _make(name):
    def func(*args, **kwargs):
        return reduce(*args, op=name, **kwargs)
    func.__name__ = name
    globals()[name] = func

for name in ["sum", "mean", "var", "std", "prod", "count_nonzero", "any", "all", "max", "min"]:
    _make(name)
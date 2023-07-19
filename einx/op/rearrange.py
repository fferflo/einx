import einx
import numpy as np
from .tensor import Tensor

@einx.lru_cache
def _make_op(expr_in, expr_out):
    isolated_axes = einx.expr.get_isolated_axes([expr_in, expr_out])

    ops = []

    # Reshape nested input to flat input
    shape = tuple(einx.expr.get_flattened_shape(expr_in))

    # Squeeze input dimensions
    if len(isolated_axes[0]) > 0:
        if any(v.value > 1 for v in isolated_axes[0]):
            raise ValueError(f"Non-trivial isolated input axes {[str(v) for v in isolated_axes[0]]} are not allowed")
        shape2 = tuple(v.value for v in einx.expr.get_flattened_axes(expr_in) if not v in isolated_axes[0])
        assert len(shape2) != len(shape)
        shape = shape2
    if shape != expr_in.shape:
        ops.append(lambda x, backend, shape=shape: backend.reshape(x, shape))

    # Transpose to flat output
    in_variables = [v for v in einx.expr.get_flattened_axes(expr_in) if not v in isolated_axes[0]]
    out_variables = [v for v in einx.expr.get_flattened_axes(expr_out) if not v in isolated_axes[1]]
    assert len(in_variables) == len(shape)
    assert len(out_variables) == len(shape)
    perm = [in_variables.index(out_variable) for out_variable in out_variables]
    if perm != list(range(len(perm))):
        shape = [shape[i] for i in perm]
        ops.append(lambda x, backend, perm=perm: backend.transpose(x, perm))

    # Expand and broadcast missing output dimensions
    if len(isolated_axes[1]) > 0:
        shape = tuple(1 if v in isolated_axes[1] else v.value for v in einx.expr.get_flattened_axes(expr_out))
        ops.append(lambda x, backend, shape=shape: backend.reshape(x, shape))
        broadcast_shape = tuple(einx.expr.get_flattened_shape(expr_out))
        if np.any(shape != broadcast_shape):
            ops.append(lambda x, backend, shape=broadcast_shape: backend.broadcast_to(x, shape))
        shape = broadcast_shape

    # Reshape flat output to nested output
    if expr_out.shape != shape:
        ops.append(lambda x, backend, shape=expr_out.shape: backend.reshape(x, shape))

    def op(x, backend, ops=ops):
        for op in ops:
            x = op(x, backend)
        return x
    return op

def rearrange(tensor_in, expr_out, backend=None):
    if backend is None:
        backend = tensor_in.backend
    op = _make_op(tensor_in.expr, expr_out)
    value_out = op(tensor_in.value, backend=backend)
    return Tensor(value_out, expr_out, backend=backend)
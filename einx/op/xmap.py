import einx
from einx.expr import stage1, stage2, stage3, solve, Condition
from .tensor import Tensor
import numpy as np

@einx.lru_cache
def _make_op(exprs_in, expr_out, op, xmap):
    if len(exprs_in) < 1:
        raise ValueError("Must have at least one input tensor")
    exprs = list(exprs_in) + [expr_out]

    ops = []

    # Reshape nested input to flat input
    shapes = [tuple(einx.expr.get_flattened_shape(expr)) for expr in exprs]

    # Squeeze input dimensions
    shapes2 = [(shape if shape != expr_in.shape else None) for shape, expr_in in zip(shapes, exprs_in)]
    ops.append(lambda xs, backend, shapes=shapes2: [(backend.reshape(x, shape) if not shape is None else x) for x, shape in zip(xs, shapes)])

    # Apply vmap to op
    def is_not_vmapped(expr):
        return (isinstance(expr, stage3.Group) and expr.front == "[") or (not expr.parent is None and is_not_vmapped(expr.parent))
    vmapped_variables = []
    for expr in exprs:
        for v in einx.expr.get_flattened_axes(expr):
            if not is_not_vmapped(v) and not v in vmapped_variables:
                vmapped_variables.append(v)
    flattened_axes = [list(einx.expr.get_flattened_axes(expr)) for expr in exprs]
    exprs = [stage3.prune_group(expr, lambda n: n.front == "[") for expr in exprs]
    exprs_in = exprs[:-1]
    expr_out = exprs[-1]

    for v in flattened_axes[-1]:
        if v in vmapped_variables and not any(v in expr for expr in flattened_axes[:-1]):
            raise ValueError(f"Only non-vmapped axes in output expression should be marked with []-brackets, but found {v}")

    xmaps = []
    for v in vmapped_variables:
        dims = [axes.index(v) if v in axes else None for axes in flattened_axes]
        in_axes = tuple(dims[:-1])
        out_axes = dims[-1]
        if out_axes is None:
            raise ValueError("All vmapped axes must appear in the output expression")
        xmaps.append((in_axes, out_axes)) # TODO: add support for multiple outputs (can only be tuples?)
        for axes in flattened_axes:
            if v in axes:
                axes.remove(v)
    for in_axes, out_axes in reversed(xmaps):
        op = xmap(op, in_axes=in_axes, out_axes=out_axes)
    ops.append(lambda xs, backend, op=op: op(*xs))

    # Check that output of vmap is as expected
    shape = tuple(v.value for v in einx.expr.get_flattened_axes(expr_out))
    def check(x, backend, shape=shape):
        if shape != x.shape:
            raise ValueError(f"Expected vmapped output shape {shape} but got {x.shape}")
        return x
    ops.append(check)

    # Reshape flat output to nested output
    if shape != expr_out.shape:
        ops.append(lambda x, backend, shape=expr_out.shape: backend.reshape(x, shape))

    def tensor_op(*x, backend, ops=ops):
        for op in ops:
            x = op(x, backend)
        return x
    return tensor_op

def xmap(tensors_in, expr_out, op, xmap, backend=None):
    if backend is None:
        backend = einx.backend.get([t.value for t in tensors_in])
    if isinstance(op, str):
        op = vars(backend)[op]
    if isinstance(xmap, str):
        xmap = vars(backend)[xmap]
    tensor_op = _make_op([t.expr for t in tensors_in], expr_out, op=op, xmap=xmap)
    value_out = tensor_op(*[t.value for t in tensors_in], backend=backend)
    return Tensor(value_out, expr_out, backend=backend)
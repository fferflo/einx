import einx._src.namedtensor.stage3 as stage3
from functools import partial
from einx._src.namedtensor import NamedTensor
import einx._src.tracer as tracer
import functools
from ._util import _ensure_output
from einx._src.util.functools import use_name_of


def op(op, vmap, expected_type=None, allow_squeeze_unsqueeze=False, classical=None):
    op_in = op

    @use_name_of(op)
    def inner(*tensors, out, **kwargs):
        if isinstance(out, list | tuple):
            exprs_out = out
        else:
            exprs_out = [out]

        exprs_in = [t.expr for t in tensors]
        tensors = [t.value for t in tensors]

        # TODO: In what order should vmap be called over multiple axes?
        vectorized_axisnames = []
        for expr in list(exprs_in) + list(exprs_out):
            for axis in expr.nodes():
                if isinstance(axis, stage3.Axis) and not stage3.is_in_brackets(axis) and axis.name not in vectorized_axisnames:
                    vectorized_axisnames.append(axis.name)

        def get_pos_in_expr(axisname, expr):
            for i, node in enumerate(expr):
                if isinstance(node, stage3.Axis) and node.name == axisname:
                    return i
            return None

        def get_pos_in_exprs(axisname, exprs):
            axis = tuple(get_pos_in_expr(axisname, expr) for expr in exprs)
            if len(axis) == 1:
                return axis[0]
            else:
                return axis

        def remove_axis(expr, axisname):
            return stage3.remove(expr, pred=lambda x: isinstance(x, stage3.Axis) and x.name == axisname)

        # Determine arguments to the vmap calls
        vmaps = []
        exprs_in_left = list(exprs_in)
        exprs_out_left = list(exprs_out)
        expected_out_shapes_inner = [expr.shape for expr in exprs_out_left]
        for axis in vectorized_axisnames:
            in_axes = get_pos_in_exprs(axis, exprs_in_left)
            out_axes = get_pos_in_exprs(axis, exprs_out_left)

            exprs_in_left = [remove_axis(expr, axis) for expr in exprs_in_left]
            exprs_out_left = [remove_axis(expr, axis) for expr in exprs_out_left]

            expected_out_shapes_inner = [expr.shape for expr in exprs_out_left]

            vmaps.append((in_axes, out_axes))

        # Create vmapped function
        op = op_in
        op = partial(op, **kwargs)
        op = _ensure_output(op, expected_out_shapes_inner, expected_type=expected_type, allow_squeeze_unsqueeze=allow_squeeze_unsqueeze, classical=classical)
        for in_axes, out_axes in reversed(vmaps):
            op = vmap(op, in_axes=in_axes, out_axes=out_axes)

        tensors = op(*tensors)

        if len(exprs_out) == 1:
            return NamedTensor(tensors, exprs_out[0])
        else:
            return tuple(NamedTensor(tensor, expr) for tensor, expr in zip(tensors, exprs_out, strict=False))

    return inner

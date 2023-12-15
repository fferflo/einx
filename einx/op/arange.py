import einx
from functools import partial
from . import util
import numpy as np

@einx.lru_cache(trace=lambda k: False)
def arange_stage3(expr_in, expr_out, backend, dtype="int32"):
    if isinstance(backend, str):
        backend = einx.backend.get(backend)
    for expr in expr_in.all():
        if isinstance(expr, einx.expr.stage3.Marker):
            raise ValueError("Marker in input expression not allowed")

    marked_axes = [expr for expr in expr_out.all() if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr)]
    if len(marked_axes) > 1:
        raise ValueError(f"Expected at most one marked axis, got {len(marked_axes)}")
    ndim = marked_axes[0].value if len(marked_axes) == 1 else 1

    expr_in = util.flatten([expr_in])[0]
    expr_out_flat = util.flatten([expr_out])[0]

    def replace(expr):
        if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr):
            expr = einx.expr.stage3.Concatenation([einx.expr.stage3.Axis(None, 1) for _ in range(ndim)])
            expr = einx.expr.stage3.Composition(expr)
            return expr
    expr_out_flat_withconcat = einx.expr.stage3.replace(expr_out_flat, replace)
    expr_out_flat_withconcat = einx.expr.stage3.demark(expr_out_flat_withconcat)

    (tensor,), _ = einx.rearrange(
        [axis.__deepcopy__() for axis in expr_in],
        [backend.arange(axis.value, dtype=dtype) for axis in expr_in],
        [expr_out_flat_withconcat]
    )

    # Unflatten output expressions
    tensor, = util.unflatten([expr_out_flat], [tensor,], [expr_out], backend)

    return tensor, einx.expr.stage3.demark(expr_out)

@einx.lru_cache
def parse(description, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    description = description.split("->")
    if len(description) > 2:
        raise ValueError("Operation string must contain at most one '->'")
    if "," in description[0]:
            raise ValueError("Only a single input expression is allowed")

    if len(description) == 2:
        if "," in description[1]:
            raise ValueError("Only a single output expression is allowed")
        expr_in = einx.expr.stage1.parse(description[0])
        expr_out = einx.expr.stage1.parse(description[1])
    else:
        expr_out = einx.expr.stage1.parse(description[0])
        expr_in = einx.expr.stage1.get_unmarked(expr_out)

    marked_expr_out = einx.expr.stage1.Composition(einx.expr.stage1.get_marked(expr_out))
    def after_stage2(exprs1, exprs2):
        expr_out = exprs1[1]
        out_axes = [expr for expr in expr_out.all() if isinstance(expr, (einx.expr.stage2.NamedAxis, einx.expr.stage2.UnnamedAxis))]
        marked_out_axes = [expr for expr in out_axes if einx.expr.stage2.is_marked(expr)]
        if len(marked_out_axes) > 1:
            raise ValueError(f"Expected at most one marked axis, got {len(marked_out_axes)}")
        ndim = len(out_axes) - len(marked_out_axes)
        return [einx.expr.Equation(marked_expr_out, np.asarray([ndim]))]

    expr_in, expr_out = einx.expr.solve(
            [einx.expr.Equation(expr_in)] \
          + [einx.expr.Equation(expr_out)] \
          + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
        cse=cse,
        after_stage2=after_stage2,
    )[:2]

    return expr_in, expr_out

@einx.lru_cache(trace=lambda k: False)
def arange_stage0(description, backend, dtype="int32", cse=True, **parameters):
    expr_in, expr_out = parse(description, cse=cse, **parameters)
    tensor, expr_out = arange_stage3(expr_in, expr_out, backend=backend, dtype=dtype)
    return tensor

def arange(arg0, *args, **kwargs):
    """Updates and/ or returns values from an array at the given coordinates. Specializes :func:`einx.vmap`.

    The `description` argument specifies the input and output expressions and must meet one of the following formats:

    1. ``tensor, update, coordinates -> output``
       when modifying values in the tensor.
    2. ``tensor, coordinates -> output``
       when only returning values from the tensor.

    Brackets in the tensor and update expressions mark the axes that will be arangeed. Brackets in the coordinates expression mark the single coordinate axis. All other
    axes are considered batch axes.

    Args:
        description: Description string in Einstein notation (see above).
        tensor: The tensor that values will be updates/ gathered from.
        coordinates: The tensor that contains the coordinates of the values to be gathered.
        update: The tensor that contains the update values, or None. Defaults to None.
        op: The update/gather function. If `op` is a string, retrieves the attribute of `backend` with the same name.
        backend: Backend to use for all operations. If None, determines the backend from the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults to True.
        graph: Whether to return the graph representation of the operation instead of computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the update/ gather operation if `graph=False`, otherwise the graph representation of the operation.

    Examples:
        Get values from a batch of images (different indices per image):

        >>> tensor = np.random.uniform(size=(4, 128, 128, 3))
        >>> coordinates = np.random.uniform(size=(4, 100, 2))
        >>> einx.get_at("b [h w] c, b p [2] -> b p c", tensor, coordinates).shape
        (4, 100, 3)

        Set values in a batch of images (same indices per image):

        >>> tensor = np.random.uniform(size=(4, 128, 128, 3))
        >>> coordinates = np.random.uniform(size=(100, 2))
        >>> updates = np.random.uniform(size=(100, 3))
        >>> einx.set_at("b [h w] c, p [2], p c -> b [h w] c", tensor, coordinates, updates).shape
        (4, 128, 128, 3)
    """
    if isinstance(arg0, str):
        return arange_stage0(arg0, *args, **kwargs)
    else:
        return arange_stage3(arg0, *args, **kwargs)
arange.parse = parse

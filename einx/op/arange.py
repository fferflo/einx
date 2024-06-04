import einx
from functools import partial
from . import util
import numpy as np
from typing import Union
import numpy.typing as npt


@einx.jit(
    trace=lambda t, c: lambda exprs_in, expr_out, backend=None, dtype="int32": c(
        exprs_in, expr_out, dtype=dtype
    )
)
def arange_stage3(expr_in, expr_out, backend, dtype="int32"):
    if isinstance(backend, str):
        backend = einx.backend.get(backend)
    for expr in expr_in.all():
        if isinstance(expr, einx.expr.stage3.Marker):
            raise ValueError("Marker in input expression not allowed")
    for root in [expr_in, expr_out]:
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")

    marked_axes = [
        expr
        for expr in expr_out.all()
        if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr)
    ]
    if len(marked_axes) > 1:
        raise ValueError(f"Expected at most one marked axis, got {len(marked_axes)}")
    ndim = marked_axes[0].value if len(marked_axes) == 1 else 1

    expr_in = util.flatten([expr_in])[0]
    expr_out_flat = util.flatten([expr_out])[0]

    def replace(expr):
        if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr):
            expr = einx.expr.stage3.Concatenation([
                einx.expr.stage3.Axis(None, 1) for _ in range(ndim)
            ])
            expr = einx.expr.stage3.Composition(expr)
            return expr

    expr_out_flat_withconcat = einx.expr.stage3.replace(expr_out_flat, replace)
    expr_out_flat_withconcat = einx.expr.stage3.demark(expr_out_flat_withconcat)

    (tensor,), _ = einx.rearrange_stage3(
        [axis.__deepcopy__() for axis in expr_in],
        [backend.arange(axis.value, dtype=dtype) for axis in expr_in],
        [expr_out_flat_withconcat],
        backend=backend,
    )

    # Unflatten output expressions
    (tensor,) = util.unflatten(
        [expr_out_flat],
        [
            tensor,
        ],
        [expr_out],
        backend=backend,
    )

    return tensor, einx.expr.stage3.demark(expr_out)


@einx.lru_cache
def parse(description, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(
        description, parameters
    )

    op = einx.expr.stage1.parse_op(description)

    # Implicitly determine input expression
    if len(op) == 1:
        op = einx.expr.stage1.Op(
            [
                einx.expr.stage1.Args([einx.expr.stage1.get_unmarked(op[0][0])]),
                op[0],
            ],
        )

    if len(op[0]) != 1:
        raise ValueError(f"Expected 1 input expression, but got {len(op[0])}")
    if len(op[1]) != 1:
        raise ValueError(f"Expected 1 output expression, but got {len(op[1])}")

    marked_expr_out = einx.expr.stage1.Composition(einx.expr.stage1.get_marked(op[1][0]))

    def after_stage2(exprs1, exprs2):
        expr_out = exprs1[1]
        out_axes = [
            expr
            for expr in expr_out.all()
            if isinstance(expr, (einx.expr.stage2.NamedAxis, einx.expr.stage2.UnnamedAxis))
        ]
        marked_out_axes = [expr for expr in out_axes if einx.expr.stage2.is_marked(expr)]
        if len(marked_out_axes) > 1:
            raise ValueError(f"Expected at most one marked axis, got {len(marked_out_axes)}")
        ndim = len(out_axes) - len(marked_out_axes)
        return [einx.expr.Equation(marked_expr_out, np.asarray([ndim]))]

    expr_in, expr_out = einx.expr.solve(
        [einx.expr.Equation(op[0][0])]
        + [einx.expr.Equation(op[1][0])]
        + [
            einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None)
            for k, v in parameters.items()
        ],
        cse=cse,
        after_stage2=after_stage2,
    )[:2]

    return expr_in, expr_out


@einx.traceback_util.filter
@einx.jit(trace=lambda t, c: lambda description, backend=None, **kwargs: c(description, **kwargs))
def arange(
    description: str,
    *,
    backend: Union[einx.Backend, str],
    dtype: str = "int32",
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """n-dimensional ``arange`` operation.

    *This function might be removed in a future version.*

    Runs ``arange`` for every axis in ``input``, and stacks the results along the single
    marked axis in ``output``. Always uses ``start=0`` and ``step=1``.

    The `description` argument must meet one of the following formats:

    1. ``input -> output``
        Runs ``backend.arange`` for every axis in ``input``, and stacks the results along the
        marked axis in ``output``. The values are stacked in the order that the axes appear
        in ``input``.

    2. ``output``
        Implicitly determines the input expression by removing the marked axis from ``output``.

        Example: ``a b [2]`` resolves to ``a b -> a b [2]``

    Args:
        description: Description string in Einstein notation (see above).
        backend: Backend to use for all operations.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.
        graph: Whether to return the graph representation of the operation instead of computing
            the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the n-dimensional arange operation if `graph=False`, otherwise the graph
        representation of the operation.

    Examples:
        Arange two-dimensional coordinates:

        >>> tensor = einx.arange("a b [2]", a=5, b=6, backend="numpy")
        >>> tensor.shape
        (5, 6, 2)
        >>> tensor[2, 3]
        array([2, 3], dtype=int32)

        Arange two-dimensional coordinates with inverted coordinates (`Cartesian ordering
        <https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html>`_:
        First axis of tensor corresponds to second coordinate along stacked axis and vice versa.):

        >>> tensor = einx.arange("a b -> b a [2]", a=5, b=6, backend="numpy")
        >>> tensor.shape
        (6, 5, 2)
        >>> tensor[2, 3]
        array([3, 2], dtype=int32)

        Arange one-dimensional coordinates:

        >>> einx.arange("a", a=5, backend="numpy").shape
        (5,)
    """
    expr_in, expr_out = parse(description, cse=cse, **parameters)
    tensor, expr_out = arange_stage3(expr_in, expr_out, backend=backend, dtype=dtype)
    return tensor


arange.parse = parse

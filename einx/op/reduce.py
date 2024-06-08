import einx
from . import util
import numpy as np
from functools import partial
from typing import Callable, Union
import numpy.typing as npt

_any = any  # Is overwritten below


@einx.jit(
    trace=lambda t, c: lambda expr_in, tensor_in, expr_out, op, backend=None: c(
        expr_in, t(tensor_in), expr_out, op=op
    )
)
def reduce_stage3(expr_in, tensor_in, expr_out, op, backend=None):
    for root in [expr_in, expr_out]:
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    tensors_out, exprs_out = einx.vmap_with_axis_stage3(
        [expr_in], [tensor_in], [expr_out], op, backend=backend
    )
    return tensors_out[0], exprs_out[0]


@einx.lru_cache
def parse(description, tensor_shape, keepdims=None, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(
        description, parameters
    )

    op = einx.expr.stage1.parse_op(description)

    if len(op) == 1:
        expr_in = einx.expr.solve(
            [einx.expr.Equation(op[0][0], tensor_shape)]
            + [
                einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None)
                for k, v in parameters.items()
            ],
            cse=cse,
            cse_in_markers=True,
        )[0]

        if not _any(isinstance(expr, einx.expr.stage3.Marker) for expr in expr_in.all()):
            raise ValueError("No axes are marked for reduction")

        # Determine output expressions by removing markers from input expressions
        def replace(expr):
            if isinstance(expr, einx.expr.stage3.Marker):
                if keepdims:
                    return [einx.expr.stage3.Axis(None, 1)]
                else:
                    return []

        expr_out = einx.expr.stage3.replace(expr_in, replace)

    else:
        if keepdims is not None:
            raise ValueError("keepdims cannot be given when using '->'")

        if len(op[0]) != 1:
            raise ValueError(f"Expected 1 input expression, but got {len(op[0])}")
        if len(op[1]) != 1:
            raise ValueError(f"Expected 1 output expression, but got {len(op[1])}")

        expr_in, expr_out = einx.expr.solve(
            [einx.expr.Equation(op[0][0], tensor_shape)]
            + [einx.expr.Equation(op[1][0])]
            + [
                einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None)
                for k, v in parameters.items()
            ],
            cse=cse,
            cse_in_markers=True,
        )[:2]

        # If no axes are marked for reduction in expr_in, mark all axes that
        # don't appear in expr_out
        if not _any(einx.expr.stage3.is_marked(expr) for expr in expr_in.all()):
            axes_names_out = {
                axis.name for axis in expr_out.all() if isinstance(axis, einx.expr.stage3.Axis)
            }
            expr_in = einx.expr.stage3.mark(
                expr_in,
                lambda expr: isinstance(expr, einx.expr.stage3.Axis)
                and expr.name not in axes_names_out,
            )

    return expr_in, expr_out


@einx.traceback_util.filter
@einx.jit(
    trace=lambda t, c: lambda description, tensor, backend=None, **kwargs: c(
        description, t(tensor), **kwargs
    )
)
def reduce(
    description: str,
    tensor: einx.Tensor,
    op: Union[Callable, str],
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Applies a reduction operation on the given tensors.

    The operation reduces all marked axes in the input to a single scalar. It supports
    the following shorthand notation:

    * When no brackets are found, brackets are placed implicitly around all axes that do not
      appear in the output.

      Example: ``a b c -> a c`` resolves to ``a [b] c -> a c``.

    * When no output is given, it is determined implicitly by removing marked subexpressions
      from the input.

      Example: ``a [b] c`` resolves to ``a [b] c -> a c``.

    Args:
        description: Description string for the operation in einx notation.
        tensor: Input tensor or tensor factory matching the description string.
        op: Backend reduction operation. Is called with ``op(tensor, axis=...)``. If ``op`` is
            a string, retrieves the attribute of ``backend`` with the same name.
        keepdims: Whether to replace marked expressions with 1s instead of dropping them. Must
            be None when ``description`` already contains an output expression. Defaults to None.
        backend: Backend to use for all operations. If None, determines the backend from the
            input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.
        graph: Whether to return the graph representation of the operation instead of
            computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the reduction operation if ``graph=False``, otherwise the graph
        representation of the operation.

    Examples:
        Compute mean along rows of a matrix:

        >>> x = np.random.uniform(size=(16, 20))
        >>> einx.mean("a b -> b", x).shape
        (20,)
        >>> einx.mean("[a] b -> b", x).shape
        (20,)
        >>> einx.mean("[a] b", x).shape
        (20,)

        Compute sum along rows of a matrix and broadcast to the original shape:

        >>> x = np.random.uniform(size=(16, 20))
        >>> einx.sum("[a] b -> a b", x).shape
        (16, 20,)

        Sum pooling with kernel size 2:

        >>> x = np.random.uniform(size=(4, 16, 16, 3))
        >>> einx.sum("b (s [s2])... c", x, s2=2).shape
        (4, 8, 8, 3)

        Compute variance per channel over an image:

        >>> x = np.random.uniform(size=(256, 256, 3))
        >>> einx.var("[...] c", x).shape
        (3,)
    """
    expr_in, expr_out = parse(
        description, einx.tracer.get_shape(tensor), keepdims=keepdims, cse=cse, **parameters
    )
    tensor, expr_out = reduce_stage3(expr_in, tensor, expr_out, op=op, backend=backend)
    return tensor


reduce.parse = parse


@einx.traceback_util.filter
def sum(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="sum"``"""
    return reduce(
        description, tensor, op="sum", keepdims=keepdims, backend=backend, cse=cse, **parameters
    )


def sum_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="sum", **kwargs)


@einx.traceback_util.filter
def mean(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="mean"``"""
    return reduce(
        description, tensor, op="mean", keepdims=keepdims, backend=backend, cse=cse, **parameters
    )


def mean_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="mean", **kwargs)


@einx.traceback_util.filter
def var(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="var"``"""
    return reduce(
        description, tensor, op="var", keepdims=keepdims, backend=backend, cse=cse, **parameters
    )


def var_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="var", **kwargs)


@einx.traceback_util.filter
def std(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="std"``"""
    return reduce(
        description, tensor, op="std", keepdims=keepdims, backend=backend, cse=cse, **parameters
    )


def std_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="std", **kwargs)


@einx.traceback_util.filter
def prod(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="prod"``"""
    return reduce(
        description, tensor, op="prod", keepdims=keepdims, backend=backend, cse=cse, **parameters
    )


def prod_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="prod", **kwargs)


@einx.traceback_util.filter
def count_nonzero(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="count_nonzero"``"""
    return reduce(
        description,
        tensor,
        op="count_nonzero",
        keepdims=keepdims,
        backend=backend,
        cse=cse,
        **parameters,
    )


def count_nonzero_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="count_nonzero", **kwargs)


@einx.traceback_util.filter
def any(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="any"``"""
    return reduce(
        description, tensor, op="any", keepdims=keepdims, backend=backend, cse=cse, **parameters
    )


def any_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="any", **kwargs)


@einx.traceback_util.filter
def all(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="all"``"""
    return reduce(
        description, tensor, op="all", keepdims=keepdims, backend=backend, cse=cse, **parameters
    )


def all_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="all", **kwargs)


@einx.traceback_util.filter
def max(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="max"``"""
    return reduce(description, tensor, op="max", keepdims=keepdims, backend=backend, **parameters)


def max_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="max", **kwargs)


@einx.traceback_util.filter
def min(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="min"``"""
    return reduce(description, tensor, op="min", keepdims=keepdims, backend=backend, **parameters)


def min_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="min", **kwargs)


@einx.traceback_util.filter
def logsumexp(
    description: str,
    tensor: einx.Tensor,
    keepdims: Union[bool, None] = None,
    backend: Union[einx.Backend, str, None] = None,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.reduce` with ``op="logsumexp"``"""
    return reduce(
        description, tensor, op="logsumexp", keepdims=keepdims, backend=backend, **parameters
    )


def logsumexp_stage3(*args, **kwargs):
    return reduce_stage3(*args, op="logsumexp", **kwargs)

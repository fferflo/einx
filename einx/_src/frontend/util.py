from einx._src.adapter.einx_from_namedtensor import _parse_op
from einx._src.adapter.einx_from_namedtensor import Invocation
from einx._src.adapter.einx_from_namedtensor import solve as _solve2
from einx._src.frontend.errors import SyntaxError
from einx._src.namedtensor import ExpressionIndicator
import einx._src.namedtensor.stage3 as stage3
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from collections.abc import Mapping
from .api import _is_scalar
from .types import Tensor
import einx._src.tracer as tracer
import types


def _exprs_to_axes(exprs):
    values = defaultdict(list)
    for root in exprs:
        for expr in root.nodes():
            if isinstance(expr, stage3.Axis):
                tokens = expr.name.split(".")
                values[tokens[0]].append((tuple(int(t) for t in tokens[1:]), expr.value))

    values2 = {}
    for name, xs in values.items():
        shape = np.amax([coord for coord, value in xs], axis=0) + 1
        value = np.zeros(shape, dtype="int32")
        for coord, v in xs:
            value[coord] = v
        if value.shape == ():
            value = int(value)
        values2[name] = value

    return values2


def _solve(description, tensor_shapes, parameters, reraise, cse):
    invocation = Invocation(
        description,
        name="operation",
        tensors=[
            tracer.signature.classical.Tensor(None, shape=shape)
            if shape is not None
            else tracer.signature.classical.ConvertibleTensor(None, shape=None, concrete=types.SimpleNamespace(type=None))
            for shape in tensor_shapes
        ],
        kwargs={},
    )
    if "->" in description:
        indicator = ExpressionIndicator(description)
        raise SyntaxError(description, pos=indicator.get_pos_for_literal("->"), message="The expression must not contain a '->' operator.\n%EXPR%")
    try:
        exprs_in, exprs_out = _parse_op(f"{description} ->", el_op=None, invocation=invocation, allow_concat=True)
        exprs_in, exprs_out = _solve2(exprs_in, exprs_out, tensor_shapes, invocation, parameters, cse_concat=True, cse=cse)
    except:
        if reraise:
            raise
        else:
            return None
    return exprs_in


def _get_shape(tensor):
    if tensor is None:
        return None
    try:
        return tuple(int(x) for x in tensor.shape)
    except:
        pass
    if _is_scalar(tensor):
        return ()
    elif callable(tensor):
        return None
    else:
        raise ValueError(f"Found {type(tensor)} which is not a valid tensor argument.")


def solve_shapes(description: str, *tensors: Tensor, **parameters: npt.ArrayLike) -> tuple[tuple[int, ...], ...]:
    """Solve for the shapes of the einx expressions under the given constraints.

    Args:
        description: Comma-separated list of tensor expressions in einx notation.
        *tensors: Tensors matching the description string. Accepts ``None`` for unknown
            shapes.
        **parameters: Additional parameters that specify dimension sizes, e.g. ``a=4``.

    Returns:
        A tuple of shapes corresponding to the input tensors.

    Example:
        >>> x = np.random.rand(3, 4)
        >>> einx.solve_shapes("a b, c b a", x, None, c=3)
        ((3, 4), (5, 4, 3))
    """
    exprs = _solve(description, [_get_shape(tensor) for tensor in tensors], parameters, reraise=True, cse=True)
    return tuple(expr.shape for expr in exprs)


def solve_axes(description: str, *tensors: Tensor, **parameters: npt.ArrayLike) -> Mapping[str, npt.ArrayLike]:
    """Solve for the length of all axes in an expression under the given constraints.

    Args:
        description: Comma-separated list of tensor expressions in einx notation.
        *tensors: Tensors matching the description string. Accepts ``None`` for unknown
            shapes.
        **parameters: Additional parameters that specify dimension sizes, e.g. ``a=4``.

    Returns:
        A mapping from axis name to their lengths. If an axis is used with an ellipsis,
        the lengths are given as a list of integers.

    Example:
        >>> x = np.random.rand(3, 4)
        >>> einx.solve_axes("a b, c b a", x, None, c=3)
        {'a': 3, 'b': 4, 'c': 3}
        >>> einx.solve_axes("a..., c a...", x, None, c=3)
        {'a': array([3, 4], dtype=int32), 'c': 3}
    """
    exprs = _solve(description, [_get_shape(tensor) for tensor in tensors], parameters, reraise=True, cse=False)
    return _exprs_to_axes(exprs)


def solve(description: str, *tensors: Tensor, **parameters: npt.ArrayLike) -> Mapping[str, npt.ArrayLike]:
    """This function is an alias for :func:`einx.solve_axes`."""
    return solve_axes(description, *tensors, **parameters)


def matches(description: str, *tensors: Tensor, **parameters: npt.ArrayLike) -> bool:
    """Returns whether the given tensors match the einx expression description under the
    given constraints.

    Args:
        description: Comma-separated list of tensor expressions in einx notation.
        *tensors: Tensors matching the description string. Accepts ``None`` for unknown
            shapes.
        **parameters: Additional parameters that specify dimension sizes, e.g. ``a=4``.

    Returns:
        True if the tensors and constraints match the description, False otherwise.

    Example:
        >>> x = np.random.rand(3, 4)
        >>> einx.matches("a b", x)
        True
        >>> einx.matches("a b c", x)
        False
    """
    try:
        solve_shapes(description, *tensors, **parameters)
        return True
    except:
        return False


def check(description: str, *tensors: Tensor, **parameters: npt.ArrayLike) -> None:
    warnings.warn("einx.check is deprecated and will be removed in a future release. Please call einx.id instead.", DeprecationWarning, stacklevel=2)
    _solve(description, [_get_shape(tensor) for tensor in tensors], parameters, reraise=True, cse=True)

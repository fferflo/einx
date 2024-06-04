import einx
import numpy as np
from collections import defaultdict
from typing import Mapping, Optional
import numpy.typing as npt


@einx.lru_cache
def _solve(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(
        description, parameters
    )

    exprs = einx.expr.stage1.parse_args(description)
    if len(exprs) != len(tensor_shapes):
        raise ValueError(f"Expected {len(exprs)} tensors, got {len(tensor_shapes)}")

    try:
        exprs = einx.expr.solve(
            [
                einx.expr.Equation(expr, tensor_shape)
                for expr, tensor_shape in zip(exprs, tensor_shapes)
            ]
            + [
                einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None)
                for k, v in parameters.items()
            ],
            cse=cse,
        )
    except (
        einx.expr.stage2.SolveDepthException,
        einx.expr.stage2.SolveExpansionException,
        einx.expr.stage3.SolveValueException,
    ):
        return None

    values = defaultdict(list)
    for root in exprs:
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Axis):
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


def solve(
    description: str, *tensors: einx.Tensor, cse: bool = False, **parameters: npt.ArrayLike
) -> Optional[Mapping[str, npt.ArrayLike]]:
    """Solve for the axis values of the given expressions and tensors.

    Args:
        description: Description string for the tensors in einx notation.
        tensors: Input tensors or tensor factories matching the description string.
        cse: Whether to apply common subexpression elimination to the expressions.
            Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        A mapping from axis name to axis value, or ``None`` if no solution was found.

    Examples:
        >>> x = np.zeros((10, 5))
        >>> einx.solve("a b", x)
        {'a': 10, 'b': 5}
    """
    return _solve(
        description, *[einx.tracer.get_shape(tensor) for tensor in tensors], cse=cse, **parameters
    )


def matches(
    description: str, *tensors: einx.Tensor, cse: bool = True, **parameters: npt.ArrayLike
) -> bool:
    """Check whether the given expressions and tensors match.

    Args:
        description: Description string for the tensors in einx notation.
        tensors: Input tensors or tensor factories matching the description string.
        cse: Whether to apply common subexpression elimination to the expressions.
            Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        True if the expressions and tensors match, False otherwise.

    Examples:
        >>> x = np.zeros((10, 5))
        >>> einx.matches("a b", x)
        True
        >>> einx.matches("a b c", x)
        False
    """
    return solve(description, *tensors, cse=cse, **parameters) is not None


@einx.traceback_util.filter
def check(
    description: str, *tensors: einx.Tensor, cse: bool = True, **parameters: npt.ArrayLike
) -> None:
    """Check whether the given expressions and tensors match and raise an exception if they don't.

    Args:
        description: Description string for the tensors in einx notation.
        tensors: Input tensors or tensor factories matching the description string.
        cse: Whether to apply common subexpression elimination to the expressions.
            Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    description, parameters = einx.op.util._clean_description_and_parameters(
        description, parameters
    )

    exprs = einx.expr.stage1.parse_args(description)
    if len(exprs) != len(tensors):
        raise ValueError(f"Expected {len(exprs)} tensors, got {len(tensors)}")

    tensor_shapes = [einx.tracer.get_shape(tensor) for tensor in tensors]
    einx.expr.solve(
        [einx.expr.Equation(expr, tensor_shape) for expr, tensor_shape in zip(exprs, tensor_shapes)]
        + [
            einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None)
            for k, v in parameters.items()
        ],
        cse=cse,
    )  # Raises an exception if no solution is found

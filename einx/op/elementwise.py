import einx
from . import util
from functools import partial
import numpy as np
from typing import Callable, Union
import numpy.typing as npt


@einx.jit(
    trace=lambda t, c: lambda exprs_in, tensors_in, expr_out, op, backend=None: c(
        exprs_in, [t(x) for x in tensors_in], expr_out, op
    )
)
def elementwise_stage3(exprs_in, tensors_in, expr_out, op, backend=None):
    for root in list(exprs_in) + [expr_out]:
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")

    assert not any(einx.expr.stage3.is_marked(expr) for root in exprs_in for expr in root.all())
    assert not any(einx.expr.stage3.is_marked(expr) for expr in expr_out.all())

    # Call tensor factories
    def get_name(s):
        if s == "add":
            return "bias"
        elif s == "multiply":
            return "scale"
        else:
            return s

    tensors_in = [
        einx.tracer.call_factory(
            tensor,
            expr.shape,
            backend,
            name=get_name(util._op_to_str(op)),
            init=util._op_to_str(op),
        )
        for tensor, expr in zip(tensors_in, exprs_in)
    ]
    tensors_in = backend.all_to_tensor(tensors_in)

    tensors_out, exprs_out = einx.vmap_with_axis_stage3(
        exprs_in, tensors_in, [expr_out], op, backend=backend
    )
    assert len(tensors_out) == 1 and len(exprs_out) == 1
    return tensors_out[0], exprs_out[0]


@einx.lru_cache
def parse(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(
        description, parameters
    )

    op = einx.expr.stage1.parse_op(description)

    # Add second input expression from marked subexpressions
    if len(op[0]) == 1 and len(tensor_shapes) == 2:
        op = einx.expr.stage1.Op(
            [
                einx.expr.stage1.Args([
                    einx.expr.stage1.demark(op[0][0]),
                    einx.expr.stage1.get_marked(op[0][0]),
                ]),
            ]
            + list(op[1:])
        )

    # Implicitly determine output expression
    if len(op) == 1:
        # Use one of the input expression if contains the axis names of
        # all others and if this choice is unique
        input_args = op[0]
        in_axis_names = [
            {expr.name for expr in root.all() if isinstance(expr, einx.expr.stage1.NamedAxis)}
            for root in input_args
        ]

        valid_parents = set()
        for i, parent in enumerate(in_axis_names):
            for j, child in enumerate(in_axis_names):
                if i != j and not child.issubset(parent):
                    break
            else:
                # Found valid parent
                valid_parents.add(input_args[i])

        if len(valid_parents) != 1:
            raise ValueError(f"Could not implicitly determine output expression for op '{op}'")
        expr_out = next(iter(valid_parents)).__deepcopy__()
        op = einx.expr.stage1.Op([op[0], einx.expr.stage1.Args([expr_out])])

    if len(op[0]) != len(tensor_shapes):
        raise ValueError(f"Expected {len(op[0])} input tensors, but got {len(tensor_shapes)}")
    if len(op[1]) != 1:
        raise ValueError(f"Expected 1 output expression, but got {len(op[1])}")

    exprs = einx.expr.solve(
        [
            einx.expr.Equation(expr_in, tensor_shape)
            for expr_in, tensor_shape in zip(op[0], tensor_shapes)
        ]
        + [
            einx.expr.Equation(
                op[1][0],
            )
        ]
        + [
            einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None)
            for k, v in parameters.items()
        ],
        cse=cse,
        cse_concat=False,
    )[: len(op[0]) + 1]
    exprs_in, expr_out = exprs[:-1], exprs[-1]

    return exprs_in, expr_out


@einx.traceback_util.filter
@einx.jit(
    trace=lambda t, c: lambda description, *tensors, backend=None, **kwargs: c(
        description, *[t(x) for x in tensors], **kwargs
    )
)
def elementwise(
    description: str,
    *tensors: einx.Tensor,
    op: Callable,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Applies an element-by-element operation over the given tensors.

    It supports the following shorthand notation:

    * The output is determined implicitly if one of the input expressions contains the named axes
      of all other inputs and if this choice is unique.

      | Example: ``a b, a`` expands to ``a b, a -> a b``.
      | Example: ``b a, b, a`` expands to ``b a, b, a -> b a``.
      | Example: ``a b, b a`` raises an exception.
      | Example: ``a b, a b`` expands to ``a b, a b -> a b``.

    * Bracket notation can be used when passing two input tensors to indicate that the second
      input is a subexpression of the first.

      Example: ``a [b]`` expands to ``a b, b``.

    Args:
        description: Description string for the operation in einx notation.
        tensors: Input tensors or tensor factories matching the description string.
        op: Backend elemebt-by-element operation. Must accept the same number of tensors
            as specified in the description string and comply with numpy broadcasting rules.
            If ``op`` is a string, retrieves the attribute of ``backend`` with the same name.
        backend: Backend to use for all operations. If None, determines the backend from
            the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.
        graph: Whether to return the graph representation of the operation instead of
            computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the elementwise operation if ``graph=False``, otherwise the graph
        representation of the operation.

    Examples:
        Compute a sum of two vectors:

        >>> a, b = np.random.uniform(size=(10,)), np.random.uniform(size=(10,))
        >>> einx.elementwise("a, a -> a", a, b, op=np.add).shape
        (10,)

        Add a vector on all columns of a matrix:

        >>> a, b = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))
        >>> einx.add("a b, a -> a b", a, b).shape
        (10, 10,)

        Subtract a vector from all rows of a matrix:

        >>> a, b = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))
        >>> einx.subtract("a b, b -> a b", a, b).shape
        (10, 10,)

        Select from one of two choices according to a boolean mask:

        >>> x, mask = (
        ...     np.random.uniform(size=(10, 10)),
        ...     np.random.uniform(size=(10,)),
        ... )
        >>> einx.where("a, a b, -> a b", mask, x, 0).shape
        (10, 10,)

        Add a bias onto all channels of a tensor:

        >>> x, w = (
        ...     np.random.uniform(size=(4, 16, 16, 64)),
        ...     np.random.uniform(size=(64,)),
        ... )
        >>> einx.add("b... [c]", x, w).shape
        (4, 16, 16, 64)
    """
    exprs_in, expr_out = parse(
        description, *[einx.tracer.get_shape(tensor) for tensor in tensors], cse=cse, **parameters
    )
    tensor, expr_out = elementwise_stage3(exprs_in, tensors, expr_out, op=op, backend=backend)
    return tensor


elementwise.parse = parse


@einx.traceback_util.filter
def add(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="add"``"""
    return elementwise(description, *tensors, op="add", backend=backend, cse=cse, **parameters)


def add_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="add", **kwargs)


@einx.traceback_util.filter
def subtract(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="subtract"``"""
    return elementwise(description, *tensors, op="subtract", backend=backend, cse=cse, **parameters)


def subtract_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="subtract", **kwargs)


@einx.traceback_util.filter
def multiply(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="multiply"``"""
    return elementwise(description, *tensors, op="multiply", backend=backend, cse=cse, **parameters)


def multiply_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="multiply", **kwargs)


@einx.traceback_util.filter
def true_divide(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="true_divide"``"""
    return elementwise(
        description, *tensors, op="true_divide", backend=backend, cse=cse, **parameters
    )


def true_divide_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="true_divide", **kwargs)


@einx.traceback_util.filter
def floor_divide(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="floor_divide"``"""
    return elementwise(
        description, *tensors, op="floor_divide", backend=backend, cse=cse, **parameters
    )


def floor_divide_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="floor_divide", **kwargs)


@einx.traceback_util.filter
def divide(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="divide"``"""
    return elementwise(description, *tensors, op="divide", backend=backend, cse=cse, **parameters)


def divide_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="divide", **kwargs)


@einx.traceback_util.filter
def logical_and(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="logical_and"``"""
    return elementwise(
        description, *tensors, op="logical_and", backend=backend, cse=cse, **parameters
    )


def logical_and_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="logical_and", **kwargs)


@einx.traceback_util.filter
def logical_or(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="logical_or"``"""
    return elementwise(
        description, *tensors, op="logical_or", backend=backend, cse=cse, **parameters
    )


def logical_or_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="logical_or", **kwargs)


@einx.traceback_util.filter
def where(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="where"``"""
    return elementwise(description, *tensors, op="where", backend=backend, cse=cse, **parameters)


def where_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="where", **kwargs)


@einx.traceback_util.filter
def less(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="less"``"""
    return elementwise(description, *tensors, op="less", backend=backend, cse=cse, **parameters)


def less_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="less", **kwargs)


@einx.traceback_util.filter
def less_equal(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="less_equal"``"""
    return elementwise(
        description, *tensors, op="less_equal", backend=backend, cse=cse, **parameters
    )


def less_equal_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="less_equal", **kwargs)


@einx.traceback_util.filter
def greater(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="greater"``"""
    return elementwise(description, *tensors, op="greater", backend=backend, cse=cse, **parameters)


def greater_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="greater", **kwargs)


@einx.traceback_util.filter
def greater_equal(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="greater_equal"``"""
    return elementwise(
        description, *tensors, op="greater_equal", backend=backend, cse=cse, **parameters
    )


def greater_equal_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="greater_equal", **kwargs)


@einx.traceback_util.filter
def equal(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="equal"``"""
    return elementwise(description, *tensors, op="equal", backend=backend, cse=cse, **parameters)


def equal_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="equal", **kwargs)


@einx.traceback_util.filter
def not_equal(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="not_equal"``"""
    return elementwise(
        description, *tensors, op="not_equal", backend=backend, cse=cse, **parameters
    )


def not_equal_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="not_equal", **kwargs)


@einx.traceback_util.filter
def maximum(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="maximum"``"""
    return elementwise(description, *tensors, op="maximum", backend=backend, cse=cse, **parameters)


def maximum_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="maximum", **kwargs)


@einx.traceback_util.filter
def minimum(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="minimum"``"""
    return elementwise(description, *tensors, op="minimum", backend=backend, cse=cse, **parameters)


def minimum_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="minimum", **kwargs)

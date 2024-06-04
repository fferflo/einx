import einx
from . import util
import numpy as np
from typing import Union, Tuple
import numpy.typing as npt


@einx.jit(
    trace=lambda t, c: lambda exprs_in, tensors_in, exprs_out, backend=None: c(
        exprs_in, [t(x) for x in tensors_in], exprs_out
    )
)
def rearrange_stage3(exprs_in, tensors_in, exprs_out, backend=None):
    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}")
    if any(
        isinstance(expr, einx.expr.stage3.Marker)
        for root in list(exprs_in) + list(exprs_out)
        for expr in root.all()
    ):
        raise ValueError(f"Marker '{expr}' is not allowed")

    # Call tensor factories
    tensors_in = [
        einx.tracer.call_factory(tensor, expr.shape, backend, name="embedding", init="rearrange")
        for tensor, expr in zip(tensors_in, exprs_in)
    ]
    tensors_in = backend.all_to_tensor(tensors_in, convert_scalars=True)

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend=backend)
    exprs_out_flat = util.flatten(exprs_out)
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_out_flat)
    if len(exprs_in) != len(exprs_out_flat):
        raise ValueError(
            f"Got different number of input ({len(exprs_in)}) and output expressions "
            f"({len(exprs_out_flat)}) (after flattening)"
        )  # TODO:

    # Order inputs to align with output expressions
    indices = util.assignment(exprs_in, exprs_out_flat)
    exprs_in = [exprs_in[i] for i in indices]
    tensors_in = [tensors_in[i] for i in indices]

    # Transpose and broadcast missing output dimensions
    tensors = [
        util.transpose_broadcast(expr_in, tensor, expr_out, backend=backend)[0]
        for expr_in, tensor, expr_out in zip(exprs_in, tensors_in, exprs_out_flat)
    ]

    # Unflatten output expressions
    tensors = util.unflatten(exprs_out_flat, tensors, exprs_out, backend=backend)

    return tensors, exprs_out


@einx.lru_cache
def parse(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(
        description, parameters
    )

    op = einx.expr.stage1.parse_op(description)

    if len(op[0]) != len(tensor_shapes):
        raise ValueError(f"Expected {len(op[0])} input tensors, but got {len(tensor_shapes)}")

    exprs = einx.expr.solve(
        [
            einx.expr.Equation(expr_in, tensor_shape)
            for expr_in, tensor_shape in zip(op[0], tensor_shapes)
        ]
        + [einx.expr.Equation(expr_out) for expr_out in op[1]]
        + [
            einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None)
            for k, v in parameters.items()
        ],
        cse=cse,
    )[: len(op[0]) + len(op[1])]
    exprs_in, exprs_out = exprs[: len(op[0])], exprs[len(op[0]) :]

    return exprs_in, exprs_out


@einx.traceback_util.filter
@einx.jit(
    trace=lambda t, c: lambda description, *tensors, backend=None, **kwargs: c(
        description, *[t(x) for x in tensors], **kwargs
    )
)
def rearrange(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> Union[einx.Tensor, Tuple[einx.Tensor, ...]]:
    """Rearranges the input tensors to match the output expressions.

    Args:
        description: Description string for the operation in einx notation. Must not contain
            brackets.
        tensors: Input tensors or tensor factories matching the description string.
        backend: Backend to use for all operations. If None, determines the backend from
            the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.
        graph: Whether to return the graph representation of the operation instead of
            computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the rearrange operation if ``graph=False``, otherwise the graph
        representation of the operation.

    Examples:
        Transpose the row and column axes of a batch of images:

        >>> x = np.random.uniform(size=(4, 64, 48, 3))
        >>> einx.rearrange("b h w c -> b w h c", x).shape
        (4, 48, 64, 3,)

        Insert new axis (repeats elements along the new axis):

        >>> x = np.random.uniform(size=(10, 10))
        >>> einx.rearrange("a b -> a c b", x, c=100).shape
        (10, 100, 10,)

        Concatenate two tensors along the first axis:

        >>> a, b = (
        ...     np.random.uniform(size=(10, 10)),
        ...     np.random.uniform(size=(20, 10)),
        ... )
        >>> einx.rearrange("a b, c b -> (a + c) b", a, b).shape
        (30, 10,)

        Split a tensor:

        >>> x = np.random.uniform(size=(10, 2))
        >>> a, b = einx.rearrange("a (1 + 1) -> a, a", x)
        >>> a.shape, b.shape
        ((10,), (10,))

        Swap the first and last third of a tensor along a given axis:

        >>> x = np.arange(6)
        >>> einx.rearrange("(b + c + d) -> (d + c + b)", x, b=2, c=2)
        array([4, 5, 2, 3, 0, 1])
    """
    exprs_in, exprs_out = parse(
        description, *[einx.tracer.get_shape(tensor) for tensor in tensors], cse=cse, **parameters
    )
    tensors, exprs_out = rearrange_stage3(exprs_in, tensors, exprs_out, backend=backend)
    return tensors[0] if len(exprs_out) == 1 else tensors


rearrange.parse = parse

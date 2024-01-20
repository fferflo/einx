import einx
from . import util
import numpy as np
from typing import Union, Tuple
import numpy.typing as npt

@einx.lru_cache(trace=lambda t, c: lambda exprs_in, tensors_in, exprs_out, backend=None: c(exprs_in, [t(x) for x in tensors_in], exprs_out))
def rearrange_stage3(exprs_in, tensors_in, exprs_out, backend=None):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    elif isinstance(backend, str):
        backend = einx.backend.get(backend)
    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}")
    if any(isinstance(expr, einx.expr.stage3.Marker) for root in list(exprs_in) + list(exprs_out) for expr in root.all()):
        raise ValueError(f"Marker '{expr}' is not allowed")

    # Call tensor factories
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend, name="embedding", init="rearrange") for tensor, expr in zip(tensors_in, exprs_in)]

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend)
    exprs_out_flat = util.flatten(exprs_out)
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_out_flat)
    if len(exprs_in) != len(exprs_out_flat):
        raise ValueError(f"Got different number of input ({len(exprs_in)}) and output expressions ({len(exprs_out_flat)}) (after flattening)") # TODO:

    # Order inputs to align with output expressions
    indices = util.assignment(exprs_in, exprs_out_flat)
    exprs_in = [exprs_in[i] for i in indices]
    tensors_in = [tensors_in[i] for i in indices]

    # Transpose and broadcast missing output dimensions
    tensors = [util.transpose_broadcast(expr_in, tensor, expr_out)[0] for expr_in, tensor, expr_out in zip(exprs_in, tensors_in, exprs_out_flat)]

    # Unflatten output expressions
    tensors = util.unflatten(exprs_out_flat, tensors, exprs_out, backend)

    return tensors, exprs_out

@einx.lru_cache
def parse(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    description = description.split("->")
    if len(description) != 2:
        raise ValueError("Operation must contain exactly one '->'")
    exprs_in, exprs_out = description
    exprs_in = exprs_in.split(",")
    exprs_out = exprs_out.split(",")
    exprs = exprs_in + exprs_out
    if len(exprs_in) != len(tensor_shapes):
        raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensor_shapes)}")

    exprs = einx.expr.solve(
            [einx.expr.Equation(expr_in, tensor_shape) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
          + [einx.expr.Equation(expr_out) for expr_out in exprs_out] \
          + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
        cse=cse,
    )[:len(exprs_in) + len(exprs_out)]
    exprs_in, exprs_out = exprs[:len(exprs_in)], exprs[len(exprs_in):]

    return exprs_in, exprs_out

@einx.lru_cache(trace=lambda t, c: lambda description, *tensors, backend=None, **kwargs: c(description, *[t(x) for x in tensors], **kwargs))
def rearrange(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> Union[einx.Tensor, Tuple[einx.Tensor, ...]]:
    """Rearranges the input tensors to match the output expressions.

    See :doc:`How does einx handle input and output tensors? </faq/flatten>`.

    The `description` argument specifies the input and output expressions. It must meet the following format:
    ``input1, input2, ... -> output1, output2, ...``

    Args:
        description: Description string in Einstein notation (see above).
        tensors: Input tensors or tensor factories matching the description string.
        backend: Backend to use for all operations. If None, determines the backend from the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults to True.
        graph: Whether to return the graph representation of the operation instead of computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the elementwise operation if `graph=False`, otherwise the graph representation of the operation.

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

        >>> a, b = np.random.uniform(size=(10, 10)), np.random.uniform(size=(20, 10))
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
    exprs_in, exprs_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensors, exprs_out = rearrange_stage3(exprs_in, tensors, exprs_out, backend=backend)
    return tensors[0] if len(exprs_out) == 1 else tensors
rearrange.parse = parse

from einx.expr import stage1, stage2, stage3, solve, Condition
import einx
from . import util
import numpy as np

@einx.lru_cache
def _parse(description, tensor_shape, **kwargs):
    conditions = kwargs.pop("conditions", [])
    output_shape = kwargs.pop("output_shape", None)
    output_ndims = kwargs.pop("output_ndims", None)
    cse = kwargs.pop("cse", True)
    parameters = kwargs

    if isinstance(description, tuple):
        if len(description) != 2:
            raise ValueError("Expected tuple of length 2")
        for k in parameters:
            if k in description[1]:
                raise ValueError("Parameter '{k}' is given twice")
        parameters.update(description[1])
        description = description[0]
    if not isinstance(description, str):
        raise ValueError("First argument must be an operation string")

    description = description.split("->")
    if len(description) != 2:
        raise ValueError("Operation must contain exactly one '->'")
    expr_in, expr_out = description
    if "," in expr_in or "," in expr_out:
        raise ValueError("Expected single input and output description")

    exprs = solve(
           [Condition(expr=expr_in, value=tensor_shape, depth=0)] \
         + [Condition(expr=expr_out, value=output_shape, shape=(output_ndims,) if not output_ndims is None else None, depth=0)] \
         + [Condition(expr=k, value=[v]) for k, v in parameters.items()] \
         + list(conditions)
    )[:2]
    for expr in exprs:
        for expr in expr.traverse():
            if isinstance(expr, stage3.Group) and not expr.front in ["", "("]:
                raise ValueError(f"Found marker group {expr} which is not allowed")

    if cse:
        exprs = einx.expr.cse.mark_common_subexpressions(exprs)
    expr_in, expr_out = exprs

    return expr_in, expr_out

def rearrange(description, tensor, return_named=False, **kwargs):
    backend = einx.backend.get([tensor])
    tensor = tensor if util.is_tensor_factory(tensor) else backend.to_tensor(tensor)

    expr_in, expr_out = _parse(description, util.get_shape(tensor), **kwargs)

    tensor_in = einx.op.Tensor(tensor, expr_in, backend=backend)

    tensor_out = einx.op.rearrange(tensor_in, expr_out, backend=backend)

    return tensor_out if return_named else tensor_out.value
rearrange.parse = _parse
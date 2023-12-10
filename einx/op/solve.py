import einx
import numpy as np

@einx.lru_cache
def _solve(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    exprs = description.split(",")
    if len(exprs) != len(tensor_shapes):
        raise ValueError(f"Expected {len(exprs)} tensors, got {len(tensor_shapes)}")

    try:
        exprs = einx.expr.solve(
              [einx.expr.Condition(expr=expr, value=tensor_shape, depth=0) for expr, tensor_shape in zip(exprs, tensor_shapes)] \
            + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
            cse=cse,
        )
    except (einx.expr.stage2.SolveDepthException, einx.expr.stage2.SolveExpansionException, einx.expr.stage3.SolveValueException):
        return None

    values = {expr.name: expr.value for root in exprs for expr in root.all() if isinstance(expr, einx.expr.stage3.Axis)}

    return values

def solve(description, *tensors, cse=True, **parameters):
    return _solve(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)

def matches(description, *tensors, cse=True, **parameters):
    return not solve(description, *tensors, cse=cse, **parameters) is None

def check(description, *tensors, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    exprs = description.split(",")
    if len(exprs) != len(tensors):
        raise ValueError(f"Expected {len(exprs)} tensors, got {len(tensors)}")

    tensor_shapes = [einx.param.get_shape(tensor) for tensor in tensors]
    einx.expr.solve(
            [einx.expr.Condition(expr=expr, value=tensor_shape, depth=0) for expr, tensor_shape in zip(exprs, tensor_shapes)] \
        + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
        cse=cse,
    ) # Raises an exception if no solution is found
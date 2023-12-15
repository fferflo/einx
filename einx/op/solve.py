import einx
import numpy as np
from collections import defaultdict

@einx.lru_cache
def _solve(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    exprs = description.split(",")
    if len(exprs) != len(tensor_shapes):
        raise ValueError(f"Expected {len(exprs)} tensors, got {len(tensor_shapes)}")

    try:
        exprs = einx.expr.solve(
              [einx.expr.Equation(expr, tensor_shape) for expr, tensor_shape in zip(exprs, tensor_shapes)] \
            + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
            cse=cse,
        )
    except (einx.expr.stage2.SolveDepthException, einx.expr.stage2.SolveExpansionException, einx.expr.stage3.SolveValueException):
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
        values2[name] = value

    return values2

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
          [einx.expr.Equation(expr, tensor_shape) for expr, tensor_shape in zip(exprs, tensor_shapes)] \
        + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
        cse=cse,
    ) # Raises an exception if no solution is found
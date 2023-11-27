import einx
from . import util
import numpy as np

@einx.lru_cache(trace=lambda k: k[0] in [1, "tensors_in"])
def rearrange_stage3(exprs_in, tensors_in, exprs_out, backend=None):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}")
    if any(isinstance(expr, einx.expr.stage3.Marker) for root in list(exprs_in) + list(exprs_out) for expr in root.all()):
        raise ValueError(f"Marker '{expr}' is not allowed")

    # Call tensor factories
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend) for tensor, expr in zip(tensors_in, exprs_in)]

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend)
    exprs_out_flat = util.flatten(exprs_out)
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_out_flat)
    if len(exprs_in) != len(exprs_out_flat):
        raise ValueError("Got different number of input and output expressions (after flattening)") # TODO:

    # Order inputs to align with output expressions
    indices = util.assignment(exprs_in, exprs_out_flat)
    exprs_in = [exprs_in[i] for i in indices]
    tensors_in = [tensors_in[i] for i in indices]

    # Transpose and broadcast missing output dimensions
    tensors = [util.transpose_broadcast(expr_in, tensor, expr_out) for expr_in, tensor, expr_out in zip(exprs_in, tensors_in, exprs_out_flat)]

    # Unflatten output expressions
    tensors = util.unflatten(exprs_out_flat, tensors, exprs_out, backend)

    return tensors, exprs_out

def parse(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.expr.util._clean_description_and_parameters(description, parameters)

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
            [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
          + [einx.expr.Condition(expr=expr_out, depth=0) for expr_out in exprs_out] \
          + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
        cse=cse,
    )[:len(exprs_in) + len(exprs_out)]
    exprs_in, exprs_out = exprs[:len(exprs_in)], exprs[len(exprs_in):]

    return exprs_in, exprs_out

@einx.lru_cache(trace=lambda k: isinstance(k[0], int) and k[0] >= 1)
def rearrange_stage0(description, *tensors, backend=None, cse=True, **parameters):
    exprs_in, exprs_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensors, exprs_out = rearrange_stage3(exprs_in, tensors, exprs_out, backend=backend)
    return tensors[0] if len(exprs_out) == 1 else tensors

def rearrange(arg0, *args, **kwargs):
    if isinstance(arg0, str) or (isinstance(arg0, tuple) and isinstance(arg0[0], str)):
        return rearrange_stage0(arg0, *args, **kwargs)
    else:
        return rearrange_stage3(arg0, *args, **kwargs)
rearrange.parse = parse
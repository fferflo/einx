import einx
from . import util
import numpy as np

@einx.lru_cache(trace=lambda k: k[0] in [1, "tensors_in"])
def dot_stage3(exprs_in, tensors_in, expr_out, backend=None):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    if any(isinstance(expr, einx.expr.stage3.Concatenation) for expr in expr_out.all()):
        raise ValueError("Output expression cannot contain concatenations")
    for root in list(exprs_in) + [expr_out]:
        if any(isinstance(expr, einx.expr.stage3.Marker) for expr in root.all()):
            raise ValueError(f"Marker is not allowed, but got {root}")

    # TODO: implicitly determine second input expr if it is not given?

    # Call tensor factories
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend) for tensor, expr in zip(tensors_in, exprs_in)]

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend)
    expr_out_flat = util.flatten([expr_out])[0]
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert einx.expr.stage3.is_flat(expr_out_flat)

    # Apply einsum
    einsum_variables = {}
    def get_einsum_variable(key):
        if key in einsum_variables:
            return einsum_variables[key]
        else:
            v = chr(ord("a") + len(einsum_variables))
            if ord(v) > ord("z"):
                raise ValueError(f"Only supports up to {ord('z') - ord('a') + 1} input tensors")
            einsum_variables[key] = v
            return v
    def to_einsum(axes):
        return " ".join(get_einsum_variable(a.name) for a in axes)

    input_axes = set(a.name for expr in exprs_in for a in einx.expr.stage3.get_axes(expr))

    einsum_str = ", ".join(to_einsum(einx.expr.stage3.get_axes(expr)) for expr in exprs_in) \
               + " -> " + to_einsum([a for a in einx.expr.stage3.get_axes(expr_out_flat) if a.name in input_axes])

    tensor = backend.einsum(einsum_str, *tensors_in)

    # Transpose and broadcast missing output dimensions
    tensor = util.transpose_broadcast(expr_out_flat, tensor, expr_out)

    # Unflatten output expression
    tensor = backend.reshape(tensor, expr_out.shape)

    return tensor, expr_out

def parse(description, *tensor_shapes, cse=True, **parameters):
    if isinstance(description, tuple):
        if len(description) != 2:
            raise ValueError("Expected tuple of length 2")
        for k in parameters:
            if k in description[1]:
                raise ValueError(f"Parameter '{k}' is given twice")
        parameters.update(description[1])
        description = description[0]
    if not isinstance(description, str):
        raise ValueError("First argument must be an operation string")

    description = description.split("->")
    if len(description) == 1:
        # "input -> output" using [|]-choice
        expr = description[0]
        if "," in expr:
            raise ValueError("Only a single input expression is allowed when output expression is not given")
        if len(tensor_shapes) != 2:
            raise ValueError(f"Expected 2 input tensors, got {len(tensor_shapes)}")

        expr = einx.expr.stage1.parse(expr)
        expr_in1 = str(einx.expr.stage1.choose(expr, 0, num=2))
        expr_out = str(einx.expr.stage1.choose(expr, 1, num=2))

        exprs_in = [expr_in1]
    else:
        # "inputs... -> output"
        if len(description) > 2:
            raise ValueError("Operation can contain at most one '->'")
        exprs_in, expr_out = description
        exprs_in = exprs_in.split(",")

    if len(exprs_in) == 1 and len(tensor_shapes) == 2:
        # input1 -> output, determine input2 implicitly
        expr_in1 = einx.expr.stage1.parse(exprs_in[0])
        expr_out = einx.expr.stage1.parse(expr_out)

        for root in [expr_in1, expr_out]:
            for expr in root.all():
                if isinstance(expr, einx.expr.stage1.UnnamedAxis) and expr.value != 1 and einx.expr.stage1.is_marked(expr):
                    raise ValueError(f"Cannot mark unnamed non-trivial axes, but found {expr}")

        # Get ordered list of axes for second input
        names = []
        for root in [expr_in1, expr_out]:
            for expr in root.all():
                if isinstance(expr, einx.expr.stage1.NamedAxis) and einx.expr.stage1.is_marked(expr):
                    name = expr.name
                    for _ in range(expr.depth):
                        name = name + einx.expr.stage1._ellipsis
                    if not name in names:
                        names.append(name)
        expr_in2 = " ".join(names)

        expr_in1 = str(einx.expr.stage1.demark(expr_in1))
        expr_out = str(einx.expr.stage1.demark(expr_out))
        exprs_in = [expr_in1, expr_in2]

    if len(exprs_in) != len(tensor_shapes):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensor_shapes)}")

    exprs = einx.expr.solve(
            [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
          + [einx.expr.Condition(expr=expr_out, depth=0)] \
          + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
        cse=cse,
        cse_concat=False,
    )[:len(exprs_in) + 1]
    exprs_in, expr_out = exprs[:-1], exprs[-1]

    return exprs_in, expr_out

@einx.lru_cache(trace=lambda k: isinstance(k[0], int) and k[0] >= 1)
def dot_stage0(description, *tensors, backend=None, cse=True, **parameters):
    exprs_in, expr_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensor, expr = dot_stage3(exprs_in, tensors, expr_out, backend=backend)
    return tensor

def dot(arg0, *args, **kwargs):
    if isinstance(arg0, str) or (isinstance(arg0, tuple) and isinstance(arg0[0], str)):
        return dot_stage0(arg0, *args, **kwargs)
    else:
        return dot_stage3(arg0, *args, **kwargs)
dot.parse = parse
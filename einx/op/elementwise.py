import einx
from . import util
from functools import partial

_op_names = ["add", "subtract", "multiply", "true_divide", "floor_divide", "divide", "logical_and", "logical_or", "where", "less", "less_equal", "greater", "greater_equal", "equal", "not_equal", "maximum", "minimum"]

@einx.lru_cache(trace=lambda k: k[0] in [1, "tensors_in"])
def elementwise_stage3(exprs_in, tensors_in, expr_out, backend=None, op=None):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    if op is None:
        raise TypeError("op cannot be None")
    if isinstance(op, str):
        op = getattr(backend, op)
    else:
        op = partial(backend.elementwise, op=op)

    # Implicitly determine output expression
    if expr_out is None:
        # Check if one input expression is parent of all others
        children_str = [str(einx.expr.stage3.remove_unnamed_trivial_axes(expr)) for expr in exprs_in]
        for i, parent in enumerate(exprs_in):
            parent_str = str(parent)
            for j, child_str in enumerate(children_str):
                if i != j and not child_str in parent_str:
                    break
            else:
                # Found valid parent
                expr_out = parent.__deepcopy__()
                break
        else:
            raise ValueError(f"Could not implicitly determine output expression for input expressions {[str(expr) for expr in exprs_in]}")

    if any(isinstance(expr, einx.expr.stage3.Marker) for root in list(exprs_in) + [expr_out] for expr in root.all()):
        raise ValueError(f"Marker '{expr}' is not allowed")
    if any(isinstance(expr, einx.expr.stage3.Concatenation) for expr in expr_out.all()):
        raise ValueError("Output expression cannot contain concatenation")

    # Call tensor factories
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend) for tensor, expr in zip(tensors_in, exprs_in)]

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend)
    expr_out_flat = util.flatten([expr_out])[0]
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert einx.expr.stage3.is_flat(expr_out_flat)

    # Transpose and insert trivial axes
    tensors = [util.transpose_broadcast(expr_in, tensor, expr_out_flat, broadcast=False) for expr_in, tensor in zip(exprs_in, tensors_in)]

    # Apply elementwise operation
    tensor = op(*tensors)
    if tensor.shape != expr_out_flat.shape:
        tensor = backend.broadcast_to(tensor, expr_out_flat.shape)

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

    if "->" in description:
        # Description: Inputs and output
        description = description.split("->")
        if len(description) != 2:
            raise ValueError("Operation cannot contain more than one '->'")
        exprs_in, expr_out = description
        exprs_in = exprs_in.split(",")
        if len(exprs_in) != len(tensor_shapes):
            raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensor_shapes)}")

        exprs = einx.expr.solve(
              [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
            + [einx.expr.Condition(expr=expr_out, depth=0)] \
            + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
            cse=cse,
            cse_concat=False,
        )[:len(exprs_in) + 1]
        exprs_in, expr_out = exprs[:-1], exprs[-1]
    else:
        # Description: Only inputs
        exprs_in = description.split(",")

        if "[" in description:
            # Expression contains markers -> add second input expression from marked subexpressions
            if len(exprs_in) != 1:
                raise ValueError(f"Expected 1 input expression when using markers, got {len(exprs_in)}")
            if len(tensor_shapes) != 2:
                raise ValueError(f"Expected 2 input tensors when using markers, got {len(tensor_shapes)}")

            expr_in1 = einx.expr.solve(
                [einx.expr.Condition(expr=exprs_in[0], value=tensor_shapes[0], depth=0)] \
              + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
                cse=cse,
                cse_concat=False,
            )[0]

            expr_in2 = einx.expr.stage3.get_marked(expr_in1)
            if not tensor_shapes[1] is None and expr_in2.shape != tensor_shapes[1]:
                raise einx.expr.stage3.SolveError(f"Failed to solve axis values. Expected shape {expr_in2.shape} for second input tensor, got {tensor_shapes[1]}")
            expr_in1 = einx.expr.stage3.demark(expr_in1)
            exprs_in = [expr_in1, expr_in2]
        else:
            if len(exprs_in) != len(tensor_shapes):
                raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensor_shapes)}")

            exprs_in = einx.expr.solve(
                [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
              + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
                cse=cse,
                cse_concat=False,
            )[:len(exprs_in)]

        expr_out = None

    return exprs_in, expr_out

@einx.lru_cache(trace=lambda k: isinstance(k[0], int) and k[0] >= 1)
def elementwise_stage0(description, *tensors, op, backend=None, cse=True, **parameters):
    exprs_in, expr_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensor, expr_out = elementwise_stage3(exprs_in, tensors, expr_out, op=op, backend=backend)
    return tensor

def elementwise(arg0, *args, **kwargs):
    if isinstance(arg0, str) or (isinstance(arg0, tuple) and isinstance(arg0[0], str)):
        return elementwise_stage0(arg0, *args, **kwargs)
    else:
        return elementwise_stage3(arg0, *args, **kwargs)
elementwise._op_names = _op_names
elementwise.parse = parse


def _make(name):
    def func(*args, **kwargs):
        return elementwise(*args, op=name, **kwargs)
    func.__name__ = name
    globals()[name] = func

for name in _op_names:
    _make(name)

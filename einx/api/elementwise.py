from einx.expr import stage1, stage2, stage3, solve, Condition
import einx
from . import util
from functools import partial

_op_names = ["add", "subtract", "multiply", "true_divide", "floor_divide", "divide", "logical_and", "logical_or", "where", "less", "less_equal", "greater", "greater_equal", "equal", "not_equal", "maximum", "minimum"]

@einx.lru_cache
def _parse(description, *tensor_shapes, conditions=[], output_shape=None, output_ndims=None, cse=True, **parameters):
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

    if "->" in description:
        # Description: Inputs and output
        description = description.split("->")
        if len(description) != 2:
            raise ValueError("Operation cannot contain more than one '->'")
        exprs_in, expr_out = description
        exprs_in = exprs_in.split(",")
        if len(exprs_in) != len(tensor_shapes):
            raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensor_shapes)}")
    else:
        # Description: Only inputs
        exprs_in = description.split(",")

        expr_in1 = stage1.parse(exprs_in[0])
        if len(exprs_in) == 1 and any(isinstance(expr, stage1.Group) and expr.front == "[" for expr in expr_in1.traverse()):
            # Description: Single input with [] group
            def any_parent_is_marker(node):
                if isinstance(node, stage1.Group) and node.front == "[":
                    return True
                elif node.parent is None:
                    return False
                else:
                    return any_parent_is_marker(node.parent)
            expr_in2 = expr_in1
            expr_in2 = stage1.remove(expr_in2, lambda n: not any_parent_is_marker(n))
            expr_in2 = stage1.prune_group(expr_in2, lambda n: n.front == "[")

            expr_in1 = stage1.prune_group(expr_in1, lambda n: n.front == "[")

            exprs_in = [str(expr_in1), str(expr_in2)]

        if len(exprs_in) != len(tensor_shapes):
            raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensor_shapes)}")

        expr_out = None

    # Drop unnecessary parameters
    exprs = [stage1.parse(expr) if not expr_out is None else None for expr in exprs_in + [expr_out]]
    def is_necessary_parameter(k):
        for expr in exprs:
            if not expr is None and any(var.name == k for var in expr.variables):
                return True
        return False
    parameters = {k: v for k, v in parameters.items() if is_necessary_parameter(k)}

    exprs = solve(
           [Condition(expr=expr, value=tensor_shape, depth=0) for expr, tensor_shape in zip(exprs_in, tensor_shapes)] \
         + ([Condition(expr=expr_out, value=output_shape, shape=(output_ndims,) if not output_ndims is None else None, depth=0)] if not expr_out is None else []) \
         + [Condition(expr=k, value=[v]) for k, v in parameters.items()] \
         + list(conditions)
    )[:len(exprs_in) + (1 if not expr_out is None else 0)]
    for expr in exprs:
        for expr in expr.traverse():
            if isinstance(expr, stage3.Group) and not expr.front in ["", "("]:
                raise ValueError(f"Found marker group {expr} which is not allowed")

    if cse:
        exprs = einx.expr.cse.mark_common_subexpressions(exprs)
    if expr_out is None:
        exprs_in = exprs
    else:
        exprs_in, expr_out = exprs[:-1], exprs[-1]

    return exprs_in, expr_out

def elementwise(description, *tensors, op, conditions=[], output_shape=None, output_ndims=None, return_named=False, cse=True, **parameters):
    backend = einx.backend.get(tensors)
    if isinstance(op, str):
        op = vars(backend)[op]
    tensors = [t if util.is_tensor_factory(t) else backend.to_tensor(t) for t in tensors]

    exprs_in, expr_out = _parse(description, *[util.get_shape(t) for t in tensors], conditions=conditions, output_shape=output_shape, output_ndims=output_ndims, cse=cse, **parameters)

    tensors_in = [einx.op.Tensor(tensor, expr, backend=backend) for tensor, expr in zip(tensors, exprs_in)]

    tensor_out = einx.op.elementwise(tensors_in, expr_out, op=op, backend=backend)

    return tensor_out if return_named else tensor_out.value
elementwise.parse = _parse
elementwise.op_names = _op_names

def _make(name):
    def func(*args, **kwargs):
        return elementwise(*args, op=name, **kwargs)
    func.__name__ = name
    func.parse = partial(_parse, op=name)
    globals()[name] = func

for name in _op_names:
    _make(name)

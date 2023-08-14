from einx.expr import stage1, stage2, stage3, solve, Condition
import einx
from . import util

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

    description = description.split("->")
    if len(description) == 1:
        # Singleton input/output with []-brackets
        expr = description[0]
        if "," in expr:
            raise ValueError("A singleton expression for einx.dot cannot contain ','")
        expr = stage1.parse(expr)
        if not any(isinstance(expr, stage1.Choice) and expr.separator == "|" for expr in expr.traverse()):
            raise ValueError("Must contain | separator when only one expression is given")

        expr_in = expr
        expr_in = stage1.prune_group(expr_in, lambda n: n.front == "[" and len(n.children) == 1 and isinstance(n.children[0], stage1.Choice) and n.children[0].separator == "|")
        expr_in = stage1.make_choice(expr_in, lambda n: n.separator == "|", 0, 2)

        expr_out = expr
        expr_out = stage1.prune_group(expr_out, lambda n: n.front == "[" and len(n.children) == 1 and isinstance(n.children[0], stage1.Choice) and n.children[0].separator == "|")
        expr_out = stage1.make_choice(expr_out, lambda n: n.separator == "|", 1, 2)

        exprs_in = [expr_in]
    else:
        if len(description) > 2:
            raise ValueError("Operation can contain at most one '->'")
        exprs_in, expr_out = description
        exprs_in = exprs_in.split(",")

    if len(exprs_in) == 1:
        # input1 -> output, determine input2 implicitly
        expr_in1 = stage1.parse(exprs_in[0])
        expr_out = stage1.parse(expr_out)

        vars_in1 = set(v for v in expr_in1.traverse() if isinstance(v, stage1.Variable))
        vars_out = set(v for v in expr_out.traverse() if isinstance(v, stage1.Variable))

        def any_parent_is_marker(node):
            if isinstance(node, stage1.Group) and node.front == "[":
                return True
            elif node.parent is None:
                return False
            else:
                return any_parent_is_marker(node.parent)
        all_batch_vars = set(v for v in vars_in1 if any_parent_is_marker(v))
        reduced_vars = vars_in1.difference(vars_out)
        right_batch_vars = vars_out.difference(vars_in1)
        vars_in2 = reduced_vars.union(right_batch_vars).union(all_batch_vars)

        invalid = all_batch_vars.intersection(reduced_vars)
        if len(invalid) > 0:
            raise ValueError(f"Batch variables must appear in output: {invalid}")

        expr_in1 = stage1.prune_group(expr_in1, lambda n: n.front == "[")
        expr_in2_1 = stage1.remove(expr_in1, lambda n: isinstance(n, stage1.Variable) and n not in vars_in2)
        expr_in2_2 = stage1.remove(expr_out, lambda n: isinstance(n, stage1.Variable) and (n not in right_batch_vars))
        expr_in2 = stage1.concatenate([expr_in2_1, expr_in2_2])

        exprs_in = [expr_in1, expr_in2]
        expr_out = str(expr_out)

    if len(exprs_in) != len(tensor_shapes):
        raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensor_shapes)}")
    exprs = exprs_in + [expr_out]

    # Drop unnecessary parameters
    exprs = [stage1.parse(expr) for expr in exprs]
    def is_necessary_parameter(k):
        for expr in exprs:
            if any(var.name == k for var in expr.variables):
                return True
        return False
    parameters = {k: v for k, v in parameters.items() if is_necessary_parameter(k)}

    exprs = solve(
           [Condition(expr=expr, value=tensor_shape, depth=0) for expr, tensor_shape in zip(exprs_in, tensor_shapes)] \
         + [Condition(expr=expr_out, value=output_shape, shape=(output_ndims,) if not output_ndims is None else None, depth=0)] \
         + [Condition(expr=k, value=[v]) for k, v in parameters.items()] \
         + list(conditions)
    )[:len(exprs)]
    for expr in exprs:
        for expr in expr.traverse():
            if isinstance(expr, stage3.Group) and not expr.front in ["", "("]:
                raise ValueError(f"Found marker group {expr} which is not allowed")

    if cse:
        exprs = einx.expr.cse.mark_common_subexpressions(exprs)
    exprs_in, expr_out = exprs[:-1], exprs[-1]

    return exprs_in, expr_out

def dot(description, *tensors, conditions=[], output_shape=None, output_ndims=None, return_named=False, cse=True, **parameters):
    backend = einx.backend.get(tensors)
    tensors = [t if util.is_tensor_factory(t) else backend.to_tensor(t) for t in tensors]

    exprs_in, expr_out = _parse(description, *[util.get_shape(t) for t in tensors], conditions=conditions, output_shape=output_shape, output_ndims=output_ndims, cse=cse, **parameters)

    tensors_in = [einx.op.Tensor(tensor, expr, backend=backend) for tensor, expr in zip(tensors, exprs_in)]

    tensor_out = einx.op.dot(tensors_in, expr_out, backend=backend)

    return tensor_out if return_named else tensor_out.value
dot.parse = _parse
from einx.expr import stage1, stage2, stage3, solve, Condition
import einx
from functools import partial
from . import util

_op_names = ["vmap", "pmap"]

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

    if len(description) > 2:
        raise ValueError("Operation can contain at most one '->'")
    exprs_in, expr_out = description
    exprs_in = exprs_in.split(",")

    if len(exprs_in) != len(tensor_shapes):
        raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensor_shapes)}")

    exprs = exprs_in + [expr_out]

    # Drop unnecessary parameters
    exprs = solve(
           [Condition(expr=expr, value=tensor_shape, depth=0) for expr, tensor_shape in zip(exprs_in, tensor_shapes)] \
         + [Condition(expr=expr_out, value=output_shape, shape=(output_ndims,) if not output_ndims is None else None, depth=0)] \
         + [Condition(expr=k, value=[v]) for k, v in parameters.items()] \
         + list(conditions)
    )[:len(exprs)]
    for expr in exprs:
        for expr in expr.traverse():
            if isinstance(expr, stage3.Group) and not expr.front in ["", "(", "["]:
                raise ValueError(f"Found marker group {expr} which is not allowed")

    if cse:
        exprs = einx.expr.cse.mark_common_subexpressions(exprs)
    exprs_in, expr_out = exprs[:-1], exprs[-1]

    vmapped_variables = set()
    not_vmapped_variables = set()
    invalid_variable_names = set()
    def is_not_vmapped(expr):
        return (isinstance(expr, stage3.Group) and expr.front == "[") or not expr.parent is None and is_not_vmapped(expr.parent)
    for expr in list(exprs_in) + [expr_out]:
        for v in einx.expr.get_flattened_axes(expr):
            if not is_not_vmapped(v):
                if v.name in not_vmapped_variables:
                    invalid_variable_names.add(v.name)
                vmapped_variables.add(v.name)
            else:
                if v.name in vmapped_variables:
                    invalid_variable_names.add(v.name)
                not_vmapped_variables.add(v.name)
    if len(invalid_variable_names) > 0:
        raise ValueError(f"Variable names {invalid_variable_names} are used both as vmapped and not-vmapped variables")

    return exprs_in, expr_out

def xmap(description, *tensors, op, xmap, conditions=[], output_shape=None, output_ndims=None, return_named=False, cse=True, **parameters):
    backend = einx.backend.get(tensors)
    tensors = [t if util.is_tensor_factory(t) else backend.to_tensor(t) for t in tensors]

    exprs_in, expr_out = _parse(description, *[util.get_shape(t) for t in tensors], conditions=conditions, output_shape=output_shape, output_ndims=output_ndims, cse=cse, **parameters)

    tensors_in = [einx.op.Tensor(tensor, expr, backend=backend) for tensor, expr in zip(tensors, exprs_in)]

    tensor_out = einx.op.xmap(tensors_in, expr_out, op=op, xmap=xmap, backend=backend)

    return tensor_out if return_named else tensor_out.value
xmap.parse = _parse
xmap.op_names = _op_names

def _make(name):
    def func(*args, **kwargs):
        return xmap(*args, xmap=name, **kwargs)
    func.__name__ = name
    func.parse = partial(_parse, op=name)
    globals()[name] = func

for name in _op_names:
    _make(name)
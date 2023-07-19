import einx
import numpy as np
from .tensor import Tensor

@einx.lru_cache
def _make_op(exprs_in, expr_out):
    if len(exprs_in) < 1:
        raise ValueError("Must have at least one input tensor")
    isolated_axes = einx.expr.get_isolated_axes(list(exprs_in) + [expr_out])

    ops_per_tensor = []
    for i, expr_in in enumerate(exprs_in):
        ops = []

        # Reshape nested input to flat input
        shape = tuple(einx.expr.get_flattened_shape(expr_in))

        # Squeeze input dimensions
        if len(isolated_axes[i]) > 0:
            if any(v.value > 1 for v in isolated_axes[i]):
                raise ValueError(f"Non-trivial isolated input axes {[str(v) for v in isolated_axes[i]]} are not allowed")
            shape2 = tuple(v.value for v in einx.expr.get_flattened_axes(expr_in) if not v in isolated_axes[i])
            assert len(shape2) != len(shape)
            shape = shape2
        if shape != expr_in.shape:
            ops.append(lambda x, backend, shape=shape: backend.reshape(x, shape))

        # Transpose to flat output
        in_variables = [v for v in einx.expr.get_flattened_axes(expr_in) if not v in isolated_axes[i]]
        out_variables = [v for v in einx.expr.get_flattened_axes(expr_out) if not v in isolated_axes[-1]]
        assert len(in_variables) == len(shape)
        assert len(out_variables) >= len(in_variables)
        perm = [in_variables.index(out_variable) for out_variable in out_variables if out_variable in in_variables]
        if perm != list(range(len(perm))):
            shape = tuple(shape[i] for i in perm)
            ops.append(lambda x, backend, perm=perm: backend.transpose(x, perm))

        # Insert trivial axes
        shape2 = tuple((out_variable.value if out_variable in in_variables else 1) for out_variable in out_variables)
        if shape2 != shape:
            shape = shape2
            ops.append(lambda x, backend, shape=shape: backend.reshape(x, shape))

        ops_per_tensor.append(ops)

    def input_op(xs, backend, op, ops_per_tensor=ops_per_tensor):
        xs_out = []
        for tensor_idx in range(len(xs)):
            x = xs[tensor_idx]
            for op_step in ops_per_tensor[tensor_idx]:
                x = op_step(x, backend)
            xs_out.append(x)
        return xs_out

    ops = [input_op]


    # Apply elementwise operation
    ops.append(lambda xs, backend, op: op(*xs))

    # Expand and broadcast missing output dimensions
    if len(isolated_axes[-1]) > 0:
        shape = tuple(1 if v in isolated_axes[-1] else v.value for v in einx.expr.get_flattened_axes(expr_out))
        ops.append(lambda x, backend, op, shape=shape: backend.reshape(x, shape))
        shape2 = tuple(einx.expr.get_flattened_shape(expr_out))
        if shape != shape2:
            ops.append(lambda x, backend, op, shape=shape2: backend.broadcast_to(x, shape))
            shape = shape2

    # Reshape flat output to nested output
    if shape != expr_out.shape:
        ops.append(lambda x, backend, op, shape=expr_out.shape: backend.reshape(x, shape))

    def tensor_op(*x, op, backend, ops=ops):
        backend = einx.backend.get(x)
        for op_step in ops:
            x = op_step(x, backend, op)
        return x
    return tensor_op

@einx.lru_cache
def _make_out_expr(exprs_in):
    # Check if one expression contains variables of all other expressions
    candidates = [[v for v in expr.variables if not v.name.startswith("__constantdim")] for expr in exprs_in]
    def is_parent_of(parent, child):
        if parent == child:
            return True
        if len(child) > len(parent):
            return False
        for c in child:
            if not c in parent:
                return False
        return True

    prev_candidate = None
    index = 0
    for i, candidate in enumerate(candidates):
        if prev_candidate is None:
            prev_candidate = candidate
            index = i
        else:
            prev_parent_of_curr = is_parent_of(prev_candidate, candidate)
            curr_parent_of_pred = is_parent_of(candidate, prev_candidate)
            if not prev_parent_of_curr and not curr_parent_of_pred:
                break
            if not prev_parent_of_curr and curr_parent_of_pred:
                prev_candidate = candidate
                index = i
    else:
        if not prev_candidate is None:
            expr_out = exprs_in[index]
            return expr_out.copy()

    raise ValueError("Cannot implicitly determine output expression")

    # # Combine expressions if all direct children appear in same order
    # def join(variables1, expr2):
    #     intersection = set(variables1).intersection(expr2.variables)
    #     if [c for c in variables1 if c in intersection] != [c for c in expr2.variables if c in intersection]:
    #         return None

    #     intersection = [c for c in variables1 if c in intersection]
    #     print("E1", [str(i) for i in intersection])
    #     print("E2", [str(c) for c in variables1])
    #     print("E3", [str(c) for c in expr2.variables])
    #     new_children = []
    #     iter1 = iter(variables1)
    #     iter2 = iter(expr2.variables)
    #     for c in intersection:
    #         try:
    #             while (c1 := next(iter1)) != c:
    #                 print("E4.1", c1)
    #                 new_children.append(c1)
    #         except StopIteration:
    #             pass
    #         try:
    #             while (c2 := next(iter2)) != c:
    #                 print("E4.2", c2)
    #                 new_children.append(c2)
    #         except StopIteration:
    #             pass
    #         print("E3", c)
    #         new_children.append(c)
    #     for c1 in iter1:
    #         print("E6.1", c1)
    #         new_children.append(c1)
    #     for c2 in iter2:
    #         print("E6.2", c2)
    #         new_children.append(c2)

    #     return new_children

    variables = exprs_in[0].variables
    for expr in exprs_in[1:]:
        variables = join(variables, expr)
        if variables is None:
            break
    else:
        variables = [c.copy() for c in variables]
        expr = einx.expr.stage3.Root(variables, einx.expr.stage3.value(variables))

        return expr

    raise ValueError(f"Failed to implicitly determine output expression for inputs {[str(e) for e in exprs_in]}")

def elementwise(tensors_in, expr_out=None, op=None, backend=None):
    if op is None:
        raise ValueError("op cannot be None")
    if backend is None:
        backend = einx.backend.get([t.value for t in tensors_in])
    if isinstance(op, str):
        op = vars(backend)[op]
    if expr_out is None:
        expr_out = _make_out_expr([t.expr for t in tensors_in])
    tensor_op = _make_op([t.expr for t in tensors_in], expr_out)
    value_out = tensor_op(*[t.value for t in tensors_in], op=op, backend=backend)
    return Tensor(value_out, expr_out, backend=backend)

def _make(name):
    def func(*args, **kwargs):
        return elementwise(*args, op=name, **kwargs)
    func.__name__ = name
    globals()[name] = func

for name in ["add", "subtract", "multiply", "true_divide", "floor_divide", "divide", "logical_and", "logical_or", "where", "less", "less_equal", "greater", "greater_equal", "equal", "not_equal"]:
    _make(name)

import einx, sys
import numpy as np

def flatten(exprs, tensors=None, backend=None):
    """Flatten the given expressions and optionally the corresponding tensors.

    Flattening removes all compositions and concatenations and returns a list of new expressions (and optinally a list of flattened tensors).

    Parameters:
        exprs: Expressions to flatten.
        tensors: Tensors corresponding to ``exprs``. If None, flattens and returns only ``exprs``. Defaults to None.
        backend: Backend to use for tensor operations. If None, determines backend from ``tensors``. Defaults to None.

    Returns:
        exprs: The flattened expressions.
        tensors, optional: The flattened tensors. Only returned if ``tensors`` is not None.
    """
    if tensors is None:
        exprs_out = []
        for expr in exprs:
            expr = einx.expr.stage3.decompose(expr)
            expr = einx.expr.stage3.remove_unnamed_trivial_axes(expr)

            if any(isinstance(e, einx.expr.stage3.Concatenation) for e in expr):
                concat_index, concat_expr = [(i, e) for i, e in enumerate(expr) if isinstance(e, einx.expr.stage3.Concatenation)][0]
                for i in range(len(concat_expr.children)):
                    # Extract subexpression
                    subexpr = einx.expr.stage3.replace(expr, lambda expr:
                        expr.children[i].__deepcopy__() if id(expr) == id(concat_expr) else None
                    )

                    exprs_out.extend(flatten([subexpr]))
            else:
                exprs_out.append(expr)

        return exprs_out
    else:
        if len(exprs) != len(tensors):
            raise ValueError("Got different number of expressions and tensors")
        if backend is None:
            backend = einx.backend.get(tensors)
        exprs_out = []
        tensors_out = []
        for expr, tensor in zip(exprs, tensors):
            expr = einx.expr.stage3.decompose(expr)
            expr = einx.expr.stage3.remove_unnamed_trivial_axes(expr)
            assert not tensor.shape is None
            if tensor.shape != expr.shape:
                tensor = backend.reshape(tensor, expr.shape)

            if any(isinstance(e, einx.expr.stage3.Concatenation) for e in expr):
                concat_index, concat_expr = [(i, e) for i, e in enumerate(expr) if isinstance(e, einx.expr.stage3.Concatenation)][0]
                splits = np.cumsum([0] + [c.shape[0] for c in concat_expr.children])

                for i in range(len(concat_expr.children)):
                    # Extract subtensor
                    s = (slice(None),) * concat_index + (slice(splits[i], splits[i + 1]),)
                    subtensor = tensor[s] # TODO: split using np.split?

                    # Extract subexpression
                    subexpr = einx.expr.stage3.replace(expr, lambda expr:
                        expr.children[i].__deepcopy__() if id(expr) == id(concat_expr) else None
                    )

                    flattened_subexprs, flattened_subtensors = flatten([subexpr], [subtensor], backend)
                    exprs_out.extend(flattened_subexprs)
                    tensors_out.extend(flattened_subtensors)
            else:
                exprs_out.append(expr)
                tensors_out.append(tensor)

        return exprs_out, tensors_out

def assignment(exprs_in, exprs_out):
    """Solve the assignment problem between input and output expressions.

    If multiple solutions exist: For each output expression in order, choose the first input expression that matches.

    Args:
        exprs_in: Input expressions.
        exprs_out: Output expressions.

    Returns:
        indices: Indices into ``exprs_in`` with the same ordering as ``exprs_out``.
    """
    if len(exprs_in) != len(exprs_out):
        raise ValueError("Got different number of input and output expressions")
    axes_in = [set([a.name for a in einx.expr.stage3.get_named_axes(expr_in)]) for expr_in in exprs_in]
    axes_out = [set([a.name for a in einx.expr.stage3.get_named_axes(expr_out)]) for expr_out in exprs_out]

    cost_matrix = np.ones((len(exprs_out), len(exprs_in)), dtype=int)
    for i, a_out in enumerate(axes_out):
        for j, a_in in enumerate(axes_in):
            cost_matrix[i, j] = 0 if a_in.issubset(a_out) else 1

    # Simple brute-force assignment problem solver
    def assignment_solver(cost_matrix, r=0):
        if r == cost_matrix.shape[0]:
            return [], []

        # For an expr_out (r), find the first expr_in (c) that matches
        for c in range(cost_matrix.shape[1]):
            if cost_matrix[r, c] == 0:
                cost_matrix2 = cost_matrix.copy()
                cost_matrix2[r, :] = 1
                cost_matrix2[:, c] = 1
                rows, cols = assignment_solver(cost_matrix2, r + 1)
                if not rows is None:
                    return [r] + rows, [c] + cols
        return None, None

    row_ind, col_ind = assignment_solver(cost_matrix)
    if row_ind is None:
        raise RuntimeError("Failed to find assignment between input and output expressions") # TODO:
    assert np.all(row_ind == np.arange(len(exprs_out)))

    return col_ind

def transpose_broadcast(expr_in, tensor, expr_out, broadcast=True, backend=None):
    if backend is None:
        backend = einx.backend.get([tensor])
    assert einx.expr.stage3.is_flat(expr_in) and einx.expr.stage3.is_flat(expr_out), f"'{expr_in}' and '{expr_out}' must be flat"

    # Transpose axes if necessary
    in_axes = [a.name for a in einx.expr.stage3.get_axes(expr_in)]
    out_axes = [a.name for a in einx.expr.stage3.get_axes(expr_out)]
    out_axes_intersect = [a for a in out_axes if a in in_axes]
    out_axes_broadcast = [a for a in out_axes if a not in in_axes]
    if set(out_axes_intersect) != set(in_axes):
        raise RuntimeError("Found input axes that are not in output expression") # TODO:

    perm = [in_axes.index(out_axis) for out_axis in out_axes_intersect]
    if perm != list(range(len(perm))):
        tensor = backend.transpose(tensor, tuple(perm))

    # Expand and broadcast missing output dimensions if necessary
    if len(out_axes_broadcast) > 0:
        pre_broadcast_shape = tuple(1 if a.name in out_axes_broadcast else a.value for a in einx.expr.stage3.get_axes(expr_out))
        assert not tensor.shape is None
        if tensor.shape != pre_broadcast_shape:
            tensor = backend.reshape(tensor, pre_broadcast_shape)
        if broadcast and tensor.shape != expr_out.shape:
            tensor = backend.broadcast_to(tensor, expr_out.shape)

    if not broadcast:
        expr_out = einx.expr.stage3.List([(axis if axis.name in in_axes else einx.expr.stage3.Axis(None, 1)) for axis in expr_out])
    return tensor, expr_out

def _unflatten(exprs_in, tensors_in, expr_out, backend):
    expr_out_flat = einx.expr.stage3.decompose(expr_out)
    expr_out_flat = einx.expr.stage3.remove_unnamed_trivial_axes(expr_out_flat)

    if any(isinstance(e, einx.expr.stage3.Concatenation) for e in expr_out_flat):
        concat_index, concat_expr = [(i, e) for i, e in enumerate(expr_out_flat) if isinstance(e, einx.expr.stage3.Concatenation)][0]

        tensors_out = []
        for i in range(len(concat_expr.children)):
            # Extract subexpression
            subexpr = einx.expr.stage3.replace(expr_out_flat, lambda expr:
                expr.children[i].__deepcopy__() if id(expr) == id(concat_expr) else None
            )
            assert einx.expr.stage3.remove_unnamed_trivial_axes(einx.expr.stage3.decompose(subexpr)) == next(exprs_in)

            # Get subtensor
            subtensor = _unflatten(exprs_in, tensors_in, subexpr, backend)

            tensors_out.append(subtensor)

        tensor_out = backend.concatenate(tensors_out, axis=concat_index)
    else:
        tensor_out = next(tensors_in)

    assert not tensor_out.shape is None
    if tensor_out.shape != expr_out.shape:
        tensor_out = backend.reshape(tensor_out, expr_out.shape)
    return tensor_out

def unflatten(exprs_in, tensors_in, exprs_out, backend=None):
    if len(exprs_in) != len(tensors_in):
        raise ValueError("Got different number of input expressions and tensors")
    if backend is None:
        backend = einx.backend.get(tensors_in)

    iter_exprs_in = iter(exprs_in)
    iter_tensors_in = iter(tensors_in)
    tensors_out = []
    for expr_out in exprs_out:
        t = _unflatten(iter_exprs_in, iter_tensors_in, expr_out, backend)
        assert t.shape == expr_out.shape
        tensors_out.append(t)

    return tensors_out

def _clean_description_and_parameters(description, parameters):
    # Remove parameters that are not used in the description
    exprs = [einx.expr.stage1.parse(d) for d in description.split("->") for d in d.split(",")]
    axis_names = {axis.name for root in exprs for axis in root.all() if isinstance(axis, einx.expr.stage1.NamedAxis)}
    parameters = {k: v for k, v in parameters.items() if k in axis_names}

    return description, parameters

def _op_to_str(op):
    if "__name__" in dir(op):
        return op.__name__
    else:
        return str(op)
import einx, inspect, functools
from . import util
import numpy as np

@einx.lru_cache(trace=lambda k: k[0] in [1, "tensors_in"])
def vmap_stage3(exprs_in, tensors_in, exprs_out, flat=False, backend=None, op=None, kwargs={}, pass_backend=False, verbose=False):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    if op is None:
        raise TypeError("op cannot be None")
    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}")

    # Call tensor factories
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend) for tensor, expr in zip(tensors_in, exprs_in)]

    if verbose:
        print("Expressions:")
        print("    IN:", [str(e) for e in exprs_in])
        print("    OUT:", [str(e) for e in exprs_out])

    # Flatten expressions
    exprs_in_flat, tensors_in = util.flatten(exprs_in, tensors_in, backend)
    exprs_out_flat = util.flatten(exprs_out)
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_in_flat)
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_out_flat)

    if verbose:
        print("Flat expressions:")
        print("    IN:", [str(e) for e in exprs_in_flat])
        print("    OUT:", [str(e) for e in exprs_out_flat])

    # In op: Unflatten input arguments, flatten output arguments
    exprs_in_funcargs = [einx.expr.stage3.get_marked(expr) for expr in exprs_in]
    exprs_out_funcargs = [einx.expr.stage3.get_marked(expr) for expr in exprs_out]
    exprs_in_funcargs_flat = [einx.expr.stage3.get_marked(expr) for expr in exprs_in_flat]
    exprs_out_funcargs_flat = [einx.expr.stage3.get_marked(expr) for expr in exprs_out_flat]

    if verbose:
        print("Expressions used in op:")
        if not flat:
            print("    IN:", [str(e) for e in exprs_in_funcargs])
            print("    OUT:", [str(e) for e in exprs_out_funcargs])
        print("    IN_FLAT:", [str(e) for e in exprs_in_funcargs_flat])
        print("    OUT_FLAT:", [str(e) for e in exprs_out_funcargs_flat])

    op_ = op
    # TODO: lru_cache here?
    def op(*tensors_in_flat, op=op, backend=None):
        if backend is None:
            backend = einx.backend.get(tensors_in_flat)
        if pass_backend:
            kwargs2 = {**kwargs, "backend": backend}
        else:
            kwargs2 = kwargs

        if verbose:
            print("Flat input tensors that arrived in op:", [str(a.shape) for a in tensors_in_flat])
            print("Input types to vmapped function:", [type(t) for t in tensors_in_flat])
        assert len(tensors_in_flat) == len(exprs_in_funcargs_flat)

        if not flat:
            tensors_in = util.unflatten(exprs_in_funcargs_flat, tensors_in_flat, exprs_in_funcargs, backend=backend)
            if verbose:
                print("Unflattened input tensors in op:", [str(a.shape) for a in tensors_in])
            assert len(tensors_in) == len(exprs_in)

            exprs_out_expected = exprs_out_funcargs
        else:
            tensors_in = tensors_in_flat

            exprs_out_expected = exprs_out_funcargs_flat

        if len(exprs_out_expected) == 1:
            output_shapes = np.asarray(exprs_out_expected[0].shape)
        else:
            output_shapes = tuple(np.asarray(expr.shape) for expr in exprs_out_expected)

        tensors_out = backend.apply(op, args=tensors_in, kwargs=kwargs2, output_shapes=output_shapes)
        if not isinstance(tensors_out, (tuple, list)):
            tensors_out = (tensors_out,)
        if len(tensors_out) != len(exprs_out_expected):
            raise ValueError(f"Expected {len(exprs_out_expected)} output tensor(s) from vmapped function, but got {len(tensors_out)}")

        if verbose:
            print("Unflattened output tensors in op:")
            for i, (expr_out, tensor_out) in enumerate(zip(exprs_out_expected, tensors_out)):
                print("    ", expr_out, tensor_out.shape)

        for i, (expr_out, tensor_out) in enumerate(zip(exprs_out_expected, tensors_out)):
            if tensor_out.shape != expr_out.shape:
                raise ValueError(f"Expected output shape {expr_out.shape} from {i}-th (zero-based) output of vmapped function, but got {tensor_out.shape}")

        if not flat:
            exprs_out_funcargs_flat2, tensors_out = util.flatten(exprs_out_funcargs, tensors_out, backend=backend)

            if verbose:
                print("Flattened output tensors in op:", [str(a.shape) for a in tensors_out])
            assert exprs_out_funcargs_flat2 == exprs_out_funcargs_flat, f"{[str(s) for s in exprs_out_funcargs_flat2]} != {[str(s) for s in exprs_out_funcargs_flat]}"

        if verbose:
            print("Returning types from vmapped function:", [type(t) for t in tensors_out])
        return tuple(tensors_out)
    op.__name__ = f"wrapped({op_.__name__ if '__name__' in dir(op_) else str(op_)})"

    # Get ordered list of vmapped axes
    def is_vmapped(expr):
        return not einx.expr.stage3.is_marked(expr)
    vmapped_axes = []
    for root in list(exprs_in_flat):
        for v in root:
            if is_vmapped(v) and not v.name in vmapped_axes:
                vmapped_axes.append(v.name)
    if verbose:
        print(f"Vmapping the following axes: {vmapped_axes}")
    for root in list(exprs_in_flat) + list(exprs_out_flat):
        for v in root:
            if (v.name in vmapped_axes) != is_vmapped(v):
                raise ValueError(f"Axis {v.name} appears both as vmapped and non-vmapped")

    # Apply vmap to op
    axes_names_in = [[a.name for a in root] for root in exprs_in_flat]
    axes_names_in_set = set(a.name for root in exprs_in_flat for a in root)
    is_broadcast_axis = lambda expr: isinstance(expr, einx.expr.stage3.Axis) and not expr.name in axes_names_in_set and not einx.expr.stage3.is_marked(expr)
    exprs_out_flat_without_broadcast = [einx.expr.stage3.remove(expr, is_broadcast_axis) for expr in exprs_out_flat]
    axes_names_out_without_broadcast = [[a.name for a in root] for root in exprs_out_flat_without_broadcast]

    axisname_to_value = {a.name: a.value for root in exprs_out_flat_without_broadcast for a in root}

    if verbose:
        print("Flat output expressions without broadcast:", [str(e) for e in exprs_out_flat_without_broadcast])
        print("Got input axis names:", axes_names_in)
        print("Got output axis names (excluding broadcasted output axes):", axes_names_out_without_broadcast)

    vmaps = []
    for v in reversed(vmapped_axes):
        in_axes = tuple(axes_names.index(v) if v in axes_names else None for axes_names in axes_names_in)
        out_axes = tuple(axes_names.index(v) if v in axes_names else None for axes_names in axes_names_out_without_broadcast)
        if verbose:
            print(f"Applying backend.vmap to axis {v}, with input axis indices {in_axes} and output axis indices {out_axes}")
        for out_axis, expr_out in zip(out_axes, exprs_out_flat):
            if out_axis is None:
                raise ValueError(f"All vmapped axes must appear in the output expression, but '{v}' does not appear in '{expr_out}'")

        vmaps.append((in_axes, out_axes))

        for axes_names in axes_names_in + axes_names_out_without_broadcast:
            if v in axes_names:
                axes_names.remove(v)
            if v in axes_names_out_without_broadcast:
                axes_names_out_without_broadcast.remove(v)
        if verbose:
            print(f"Now has remaining input axes {axes_names_in} and output axes {axes_names_out_without_broadcast}")

    for in_axes, out_axes in reversed(vmaps):
        op = backend.vmap(op, in_axes=in_axes, out_axes=out_axes)

    # Apply op to tensors
    if verbose:
        print("\nSending shapes to backend.vmap:", [str(a.shape) for a in tensors_in])
    tensors = backend.apply(op, args=tensors_in, kwargs={}, output_shapes=tuple(np.asarray(expr.shape) for expr in exprs_out_flat_without_broadcast))
    if verbose:
        for tensor, expr in zip(tensors, exprs_out_flat_without_broadcast):
            print("Got overall flat tensor_out:", tensor.shape, expr)

    # Transpose and broadcast missing output dimensions
    tensors = [util.transpose_broadcast(expr_out_wb, tensor, expr_out)[0] for expr_out_wb, tensor, expr_out in zip(exprs_out_flat_without_broadcast, tensors, exprs_out_flat)]
    if verbose:
        print("Got overall transposed+broadcasted tensors_out:")
        for tensor, expr in zip(tensors, exprs_out_flat):
            print("    ", tensor.shape, expr)

    # Unflatten output expressions
    tensors = util.unflatten(exprs_out_flat, tensors, exprs_out, backend)
    if verbose:
        print("Got overall unflattened tensors_out:", [str(a.shape) for a in tensors])

    return tensors, exprs_out

@einx.lru_cache
def parse(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    if "->" in description:
        # Description: Inputs and output
        description = description.split("->")
        if len(description) != 2:
            raise ValueError("Operation string must contain exactly one '->'")
        exprs_in, exprs_out = description
        exprs_in = exprs_in.split(",")
        exprs_out = exprs_out.split(",")

        if len(exprs_in) != len(tensor_shapes):
            raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensor_shapes)}")

    else:
        # Description: "input -> output" using [|]-choice
        expr = description
        if "," in expr:
            raise ValueError("Only a single expression is allowed when using the choice operator [|]")
        if len(tensor_shapes) != 1:
            raise ValueError(f"Expected 1 input tensor, got {len(tensor_shapes)}")

        expr = einx.expr.stage1.parse(expr)
        expr_in = str(einx.expr.stage1.choose(expr, 0, num=2))
        expr_out = str(einx.expr.stage1.choose(expr, 1, num=2))

        exprs_in = [expr_in]
        exprs_out = [expr_out]

    exprs = einx.expr.solve(
          [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
        + [einx.expr.Condition(expr=expr_out, depth=0) for expr_out in exprs_out] \
        + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
        cse=cse,
        cse_concat=False,
    )[:len(exprs_in) + len(exprs_out)]
    exprs_in, exprs_out = exprs[:len(exprs_in)], exprs[len(exprs_in):]

    return exprs_in, exprs_out

@einx.lru_cache(trace=lambda k: isinstance(k[0], int) and k[0] >= 1)
def vmap_stage0(description, *tensors, op, flat=False, backend=None, cse=True, kwargs={}, pass_backend=False, **parameters):
    exprs_in, exprs_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensors, exprs_out = vmap_stage3(exprs_in, tensors, exprs_out, flat=flat, backend=backend, op=op, kwargs=kwargs, pass_backend=pass_backend)
    return tensors[0] if len(exprs_out) == 1 else tensors

def vmap(arg0, *args, **kwargs):
    """Applies a function to the marked axes of the input tensors using vectorization.

    The function flattens all input tensors, applies the vectorized operation on the tensors and rearranges
    the result to match the output expressions (see :doc:`How does einx handle input and output tensors? </faq/flatten>`).

    The `description` argument specifies the input and output expressions. It must meet one of the following formats:

    1. ``input1, input2, ... -> output1, output2, ...``
        All input and output expressions are specified explicitly. The operation is applied over all axes marked with ``[]``-brackets.
        All other axes are considered batch axes.

    2. ``... [input1|output] ...``
        The function accepts one input and one output tensor. The left and right choices correspond to the input and output tensor, respectively.

        Example: ``b [c1|c2]`` resolves to ``b [c1] -> b [c2]``

    The function ``op`` should accept input tensors and yield output tensors as specified in ``description`` with shapes matching the subexpressions that
    are marked with ``[]``-brackets.

    Args:
        description: Description string in Einstein notation (see above).
        tensors: Input tensors or tensor factories matching the description string.
        op: Function that will be vectorized. If ``op`` is a string, retrieves the attribute of `backend` with the same name.
        flat: Whether to pass the tensors to ``op`` in flattened form or matching the nested layout in the input expressions. Defaults to False.
        kwargs: Additional keyword arguments that are passed to the ``op``.
        pass_backend: Whether to pass the backend as keyword argument to ``op``. Defaults to False.
        backend: Backend to use for all operations. If None, determines the backend from the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults to True.
        graph: Whether to return the graph representation of the operation instead of computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the vectorized operation if `graph=False`, otherwise the graph representation of the operation.

    Examples:
        Compute the mean along rows of a matrix:

        >>> x = np.random.uniform(size=(10, 8))
        >>> einx.vmap("a [b] -> a", x, op=np.mean)
        (10,)

        Vectorize a custom function:

        >>> x, y = np.random.uniform(size=(10, 13, 4)), np.random.uniform(size=(4, 9,))
        >>> def op(x, y): # c, d -> 2
        >>>     return np.stack([np.mean(x), np.max(y)])
        >>> einx.vmap("b1 [c] b2, b2 [d] -> b2 [2] b1", x, y, op=op).shape
        (4, 2, 10)

        Compute a matrix-matrix multiplication

        >>> x, y = np.random.uniform(size=(5, 10)), np.random.uniform(size=(10, 3))
        >>> einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot).shape
        (5, 3)
    """
    if isinstance(arg0, str):
        return vmap_stage0(arg0, *args, **kwargs)
    else:
        return vmap_stage3(arg0, *args, **kwargs)

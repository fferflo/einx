import einx
import functools
from . import util
import numpy as np
from typing import Callable, Union, Mapping
import numpy.typing as npt


@einx.jit(
    trace=lambda t, c: lambda exprs_in, tensors_in, exprs_out, **kwargs: c(
        exprs_in, [t(x) for x in tensors_in], exprs_out, **kwargs
    )
)
def vmap_stage3(
    exprs_in,
    tensors_in,
    exprs_out,
    *,
    flat=False,
    backend=None,
    op=None,
    kwargs=None,
    verbose=False,
):
    if kwargs is None:
        kwargs = {}
    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}")
    for root in list(exprs_in) + list(exprs_out):
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")

    # Call tensor factories
    tensors_in = [
        einx.tracer.call_factory(tensor, expr.shape, backend=backend)
        for tensor, expr in zip(tensors_in, exprs_in)
    ]
    tensors_in = backend.all_to_tensor(tensors_in)

    if verbose:
        print("Expressions:")
        print("    IN:", [str(e) for e in exprs_in])
        print("    OUT:", [str(e) for e in exprs_out])

    # Flatten expressions
    exprs_in_flat, tensors_in = util.flatten(exprs_in, tensors_in, backend=backend)
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

    @einx.trace(args=[einx.tracer.Tensor(expr.shape) for expr in exprs_in_funcargs_flat])
    def op(*tensors_in_flat, op=op):
        if verbose:
            print(
                "Flat input tensors that arrived in op:",
                [str(einx.tracer.get_shape(a)) for a in tensors_in_flat],
            )
            print("Input types to vmapped function:", [type(t) for t in tensors_in_flat])
        assert len(tensors_in_flat) == len(exprs_in_funcargs_flat)

        if not flat:
            tensors_in = util.unflatten(
                exprs_in_funcargs_flat, tensors_in_flat, exprs_in_funcargs, backend=backend
            )
            if verbose:
                print(
                    "Unflattened input tensors in op:",
                    [str(einx.tracer.get_shape(a)) for a in tensors_in],
                )
            assert len(tensors_in) == len(exprs_in)
        else:
            tensors_in = tensors_in_flat
        exprs_out_expected = exprs_out_funcargs if not flat else exprs_out_funcargs_flat

        if isinstance(op, str):
            op = getattr(backend, op)
        elif not isinstance(op, einx.tracer.Tracer):
            concrete_op = op
            op = lambda *args, **kwargs: einx.tracer.apply(
                concrete_op,
                args=args,
                kwargs=kwargs,
                output=[einx.tracer.Tensor(expr.shape) for expr in exprs_out_expected]
                if len(exprs_out_expected) > 1
                else einx.tracer.Tensor(exprs_out_expected[0].shape),
            )

        tensors_out = op(*tensors_in, **kwargs)

        if not isinstance(tensors_out, (tuple, list)):
            tensors_out = (tensors_out,)
        if len(tensors_out) != len(exprs_out_expected):
            raise ValueError(
                f"Expected {len(exprs_out_expected)} output tensor(s) from vmapped "
                f"function, but got {len(tensors_out)}"
            )
        if any(not isinstance(t, einx.tracer.Tensor) for t in tensors_out):
            # TODO: might also be int, float, etc?
            raise ValueError(
                f"Expected tensors from vmapped function, but got {[type(t) for t in tensors_out]}"
            )

        if verbose:
            print("Unflattened output tensors in op:")
            for expr_out, tensor_out in zip(exprs_out_expected, tensors_out):
                print("    ", expr_out, tensor_out.shape)

        for i, (expr_out, tensor_out) in enumerate(zip(exprs_out_expected, tensors_out)):
            if einx.tracer.get_shape(tensor_out) != expr_out.shape:
                raise ValueError(
                    f"Expected output shape {expr_out.shape} from {i}-th (zero-based) "
                    f"output of vmapped function, but got {einx.tracer.get_shape(tensor_out)}"
                )

        if not flat:
            exprs_out_funcargs_flat2, tensors_out = util.flatten(
                exprs_out_funcargs, tensors_out, backend=backend
            )

            if verbose:
                print(
                    "Flattened output tensors in op:",
                    [str(einx.tracer.get_shape(a)) for a in tensors_out],
                )
            assert (
                exprs_out_funcargs_flat2 == exprs_out_funcargs_flat
            ), f"{[str(s) for s in exprs_out_funcargs_flat2]} != "
            f"{[str(s) for s in exprs_out_funcargs_flat]}"

        if verbose:
            print("Returning types from vmapped function:", [type(t) for t in tensors_out])
        return tuple(tensors_out)

    axes_names_in = [[a.name for a in root] for root in exprs_in_flat]
    axes_names_in_set = {a.name for root in exprs_in_flat for a in root}

    def is_broadcast_axis(expr):
        return (
            isinstance(expr, einx.expr.stage3.Axis)
            and expr.name not in axes_names_in_set
            and not einx.expr.stage3.is_marked(expr)
        )

    # Get ordered list of vmapped axes
    def is_vmapped(expr):
        return not einx.expr.stage3.is_marked(expr)

    vmapped_axes = []
    for root in list(exprs_in_flat):
        for v in root:
            if is_vmapped(v) and v.name not in vmapped_axes:
                vmapped_axes.append(v.name)
    if verbose:
        print(f"Vmapping the following axes: {vmapped_axes}")
    for root in list(exprs_in_flat) + list(exprs_out_flat):
        for v in root:
            if (v.name in vmapped_axes or is_broadcast_axis(v)) != is_vmapped(v):
                raise ValueError(f"Axis {v.name} appears both as vmapped and non-vmapped")

    # Apply vmap to op
    exprs_out_flat_without_broadcast = [
        einx.expr.stage3.remove(expr, is_broadcast_axis) for expr in exprs_out_flat
    ]
    axes_names_out_without_broadcast = [
        [a.name for a in root] for root in exprs_out_flat_without_broadcast
    ]

    if verbose:
        print(
            "Flat output expressions without broadcast:",
            [str(e) for e in exprs_out_flat_without_broadcast],
        )
        print("Got input axis names:", axes_names_in)
        print(
            "Got output axis names (excluding broadcasted output axes):",
            axes_names_out_without_broadcast,
        )

    vmaps = []
    input_shapes = tuple(expr.shape for expr in exprs_in_flat)
    output_shapes = tuple(expr.shape for expr in exprs_out_flat_without_broadcast)
    for v in reversed(vmapped_axes):
        in_axes = tuple(
            axes_names.index(v) if v in axes_names else None for axes_names in axes_names_in
        )
        out_axes = tuple(
            axes_names.index(v) if v in axes_names else None
            for axes_names in axes_names_out_without_broadcast
        )
        if verbose:
            print(
                f"Applying backend.vmap to axis {v}, with input axis indices "
                f"{in_axes} and output axis indices {out_axes}"
            )
        for out_axis, expr_out in zip(out_axes, exprs_out_flat):
            if out_axis is None:
                raise ValueError(
                    f"All vmapped axes must appear in the output expression, "
                    f"but '{v}' does not appear in '{expr_out}'"
                )

        vmaps.append((in_axes, out_axes, input_shapes, output_shapes))

        def drop_axis(shape, axis):
            if axis is None:
                return shape
            else:
                return shape[:axis] + shape[axis + 1 :]

        input_shapes = tuple(drop_axis(shape, axis) for shape, axis in zip(input_shapes, in_axes))
        output_shapes = tuple(
            drop_axis(shape, axis) for shape, axis in zip(output_shapes, out_axes)
        )

        for axes_names in axes_names_in + axes_names_out_without_broadcast:
            if v in axes_names:
                axes_names.remove(v)
            if v in axes_names_out_without_broadcast:
                axes_names_out_without_broadcast.remove(v)
        if verbose:
            print(
                f"Now has remaining input axes {axes_names_in} and "
                f"output axes {axes_names_out_without_broadcast}"
            )

    for in_axes, out_axes, input_shapes, output_shapes in reversed(vmaps):
        op = backend.vmap(
            op,
            in_axes=in_axes,
            out_axes=out_axes,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
        )

    # Apply op to tensors
    if verbose:
        print("\nSending shapes to backend.vmap:", [str(a.shape) for a in tensors_in])
    tensors = einx.tracer.apply(  # TODO: replace with tensors = op(*tensors_in)
        op,
        args=tensors_in,
        output=tuple(einx.tracer.Tensor(expr.shape) for expr in exprs_out_flat_without_broadcast),
    )
    if verbose:
        for tensor, expr in zip(tensors, exprs_out_flat_without_broadcast):
            print("Got overall flat tensor_out:", tensor.shape, expr)

    # Transpose and broadcast missing output dimensions
    tensors = [
        util.transpose_broadcast(expr_out_wb, tensor, expr_out, backend=backend)[0]
        for expr_out_wb, tensor, expr_out in zip(
            exprs_out_flat_without_broadcast, tensors, exprs_out_flat
        )
    ]
    if verbose:
        print("Got overall transposed+broadcasted tensors_out:")
        for tensor, expr in zip(tensors, exprs_out_flat):
            print("    ", einx.tracer.get_shape(tensor), expr)

    # Unflatten output expressions
    tensors = util.unflatten(exprs_out_flat, tensors, exprs_out, backend=backend)
    if verbose:
        print(
            "Got overall unflattened tensors_out:", [str(einx.tracer.get_shape(a)) for a in tensors]
        )

    return tensors, exprs_out


@einx.lru_cache
def parse(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(
        description, parameters
    )

    op = einx.expr.stage1.parse_op(description)

    # Implicitly determine output expression
    if len(op) == 1:
        op = einx.expr.stage1.Op([
            op[0],
            op[0].__deepcopy__(),
        ])

    if len(op[0]) != len(tensor_shapes):
        raise ValueError(f"Expected {len(op[0])} input tensors, but got {len(tensor_shapes)}")

    exprs = einx.expr.solve(
        [
            einx.expr.Equation(expr_in, tensor_shape)
            for expr_in, tensor_shape in zip(op[0], tensor_shapes)
        ]
        + [einx.expr.Equation(expr_out) for expr_out in op[1]]
        + [
            einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None)
            for k, v in parameters.items()
        ],
        cse=cse,
        cse_concat=False,
    )[: len(op[0]) + len(op[1])]
    exprs_in, exprs_out = exprs[: len(op[0])], exprs[len(op[0]) :]

    return exprs_in, exprs_out


@einx.traceback_util.filter
@einx.jit(
    trace=lambda t, c: lambda description, *tensors, **kwargs: c(
        description, *[t(x) for x in tensors], **kwargs
    )
)
def vmap(
    description: str,
    *tensors: einx.Tensor,
    op: Callable,
    flat: bool = False,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    kwargs: Mapping = {},
    **parameters: npt.ArrayLike,
):
    """Vectorizes and applies a function to the input tensors using automatic vectorization.

    The function ``op`` must accept input tensors and yield output tensors as specified in
    ``description`` with shapes matching the subexpressions that are marked with ``[]``-brackets.

    Args:
        description: Description string for the operation in einx notation.
        tensors: Input tensors or tensor factories matching the description string.
        op: Function that will be vectorized. If ``op`` is a string, retrieves the attribute
            of ``backend`` with the same name.
        flat: Whether to pass the tensors to ``op`` in flattened form or matching the nested
            layout in the input expressions. Defaults to False.
        kwargs: Additional keyword arguments that are passed to ``op``. Defaults to ``{}``.
        backend: Backend to use for all operations. If None, determines the backend from the
            input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.
        graph: Whether to return the graph representation of the operation instead of
            computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes,
            e.g. ``a=4``.

    Returns:
        The result of the vectorized operation if `graph=False`, otherwise the graph
        representation of the operation.

    Examples:
        Compute the mean along rows of a matrix:

        >>> x = np.random.uniform(size=(10, 8))
        >>> einx.vmap("a [b] -> a", x, op=np.mean)
        (10,)

        Vectorize a custom function:

        >>> x, y = (
        ...     np.random.uniform(size=(10, 13, 4)),
        ...     np.random.uniform(
        ...         size=(
        ...             4,
        ...             9,
        ...         )
        ...     ),
        ... )
        >>> def op(x, y): # c, d -> 2
        >>>     return np.stack([np.mean(x), np.max(y)])
        >>> einx.vmap("b1 [c] b2, b2 [d] -> b2 [2] b1", x, y, op=op).shape
        (4, 2, 10)

        Compute a matrix-matrix multiplication

        >>> x, y = (
        ...     np.random.uniform(size=(5, 10)),
        ...     np.random.uniform(size=(10, 3)),
        ... )
        >>> einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot).shape
        (5, 3)
    """
    exprs_in, exprs_out = parse(
        description, *[einx.tracer.get_shape(tensor) for tensor in tensors], cse=cse, **parameters
    )
    tensors, exprs_out = vmap_stage3(
        exprs_in, tensors, exprs_out, flat=flat, backend=backend, op=op, kwargs=kwargs
    )
    return tensors[0] if len(exprs_out) == 1 else tensors

import einx
from . import util
import numpy as np
from functools import partial
from typing import Callable, Mapping, Union, Tuple
import numpy.typing as npt

_op_names = ["roll", "flip"]


@einx.jit(
    trace=lambda t, c: lambda exprs_in, tensors_in, exprs_out, op, kwargs={}, backend=None: c(
        exprs_in, [t(x) for x in tensors_in], exprs_out, op, kwargs
    )
)
def vmap_with_axis_stage3(exprs_in, tensors_in, exprs_out, op, kwargs=None, backend=None):
    if kwargs is None:
        kwargs = {}
    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}")
    if len(set(exprs_out)) != 1:
        raise ValueError("All output expressions must be the same")
    for root in list(exprs_in) + list(exprs_out):
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    if len(exprs_out) > 1:
        raise ValueError("Only one output tensor allowed")
    if all(einx.tracer.is_scalar(tensor) for tensor in tensors_in):
        raise ValueError("At least one input tensor must be a non-scalar")  # TODO: support this
    kwargs = {**kwargs}

    # Call tensor factories
    tensors_in = [
        einx.tracer.call_factory(tensor, expr.shape, backend=backend)
        for tensor, expr in zip(tensors_in, exprs_in)
    ]
    tensors_in = backend.all_to_tensor(tensors_in)

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend=backend)
    in_axis_names = {axis.name for expr in exprs_in for axis in expr}

    def is_broadcast_axis(expr):
        return isinstance(expr, einx.expr.stage3.Axis) and expr.name not in in_axis_names

    exprs_out_flat = util.flatten(exprs_out)
    exprs_out_flat_without_broadcast = [
        einx.expr.stage3.remove(expr, is_broadcast_axis) for expr in exprs_out_flat
    ]

    transpose_first = len(exprs_in) > 1

    # Ensure that axis markings are consistent
    def is_vmapped(expr):
        return not einx.expr.stage3.is_marked(expr)

    vmapped_axis_names = {
        v.name
        for root in list(exprs_in) + list(exprs_out_flat_without_broadcast)
        for v in root
        if is_vmapped(v)
    }
    for root in list(exprs_in) + list(exprs_out_flat_without_broadcast):
        for v in root:
            if (v.name in vmapped_axis_names) != is_vmapped(v):
                raise ValueError(f"Axis {v.name} appears both as vmapped and non-vmapped")

    marked_input_axes = {
        axis.name
        for expr_in in exprs_in
        for axis in expr_in.all()
        if isinstance(axis, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(axis)
    }
    marked_output_axes = {
        axis.name
        for expr_out in exprs_out_flat_without_broadcast
        for axis in expr_out.all()
        if isinstance(axis, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(axis)
    }
    if marked_output_axes.difference(marked_input_axes):
        raise ValueError("Marked output axes must be a subset of marked input axes")

    if transpose_first:
        # Transpose and insert trivial axes
        if marked_input_axes != marked_output_axes:
            raise ValueError(
                "When using multiple input tensors the same axes must be marked in all tensors"
            )
        x = [
            (tensor_in, expr_in)
            if einx.tracer.is_scalar(tensor_in)
            else util.transpose_broadcast(
                expr_in,
                tensor_in,
                exprs_out_flat_without_broadcast[0],
                broadcast=False,
                backend=backend,
            )
            for expr_in, tensor_in in zip(exprs_in, tensors_in)
        ]
        tensors_in = [x[0] for x in x]
        exprs_in = [x[1] for x in x]
        assert len({len(expr) for expr in exprs_in if len(expr) > 0}) == 1
        marked_input_axes = {
            axis.name
            for expr_in in exprs_in
            for axis in expr_in.all()
            if isinstance(axis, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(axis)
        }
        exprs_op_output = exprs_out_flat_without_broadcast
    else:
        assert len(exprs_in) == 1  # TODO: see above
        expr_in = exprs_in[0]

        def to_op_output(expr_out_flat_wb):
            axis_names = {
                axis.name
                for axis in expr_out_flat_wb.all()
                if isinstance(axis, einx.expr.stage3.Axis)
            }
            new_axes = []
            for axis in expr_in.all():
                if isinstance(axis, einx.expr.stage3.Axis) and axis.name in axis_names:
                    if isinstance(axis.parent, einx.expr.stage3.Marker):
                        axis = axis.parent
                    new_axes.append(axis)
            return einx.expr.stage3.List.maybe(new_axes)

        exprs_op_output = [
            to_op_output(expr_out_flat_wb) for expr_out_flat_wb in exprs_out_flat_without_broadcast
        ]

    # Add axis argument
    if transpose_first:
        axis_indices = tuple(
            i
            for i, axis in enumerate(exprs_out_flat_without_broadcast[0])
            if axis.name in marked_input_axes
        )
    else:
        axes_in = [list(expr) for expr in exprs_in]
        axis_indices = tuple(
            i
            for i in range(len(axes_in[0]))
            if any(axes_in[i].name in marked_input_axes for axes_in in axes_in)
        )
    if len(axis_indices) > 0:
        kwargs["axis"] = axis_indices if len(axis_indices) > 1 else axis_indices[0]

    # Apply operation
    if isinstance(op, str):
        op = getattr(backend, op)
    elif not isinstance(op, einx.tracer.Tracer):
        concrete_op = op
        op = lambda *args, **kwargs: einx.tracer.apply(
            concrete_op,
            args=args,
            kwargs=kwargs,
            output=[einx.tracer.Tensor(expr.shape) for expr in exprs_op_output]
            if len(exprs_op_output) > 1
            else einx.tracer.Tensor(exprs_op_output[0].shape),
        )

    tensors_out = op(*tensors_in, **kwargs)

    if not isinstance(tensors_out, (tuple, list)):
        tensors_out = (tensors_out,)
    if len(tensors_out) != len(exprs_out_flat_without_broadcast):
        raise ValueError(
            f"Expected {len(exprs_out_flat_without_broadcast)} output tensor(s), "
            f"got {len(tensors_out)}"
        )

    # Transpose and broadcast missing output dimensions
    tensors_out = [
        util.transpose_broadcast(expr_in, tensor_out, expr_out, backend=backend)[0]
        for expr_in, tensor_out, expr_out in zip(exprs_op_output, tensors_out, exprs_out_flat)
    ]

    # Unflatten output expressions
    tensors_out = util.unflatten(exprs_out_flat, tensors_out, exprs_out, backend=backend)

    return tensors_out, exprs_out


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
    trace=lambda t, c: lambda description, *tensors, backend=None, **kwargs: c(
        description, *[t(x) for x in tensors], **kwargs
    )
)
def vmap_with_axis(
    description: str,
    *tensors: einx.Tensor,
    op: Callable,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    kwargs: Mapping = {},
    **parameters: npt.ArrayLike,
):
    """Applies a function to the marked axes of the input tensors by passing the ``axis``
    argument and relying on implicit broadcasting rules.

    The function ``op`` must accept input tensors and an ``axis`` argument specifying the
    indices of the axes along which the operation is applied. When the function is applied on
    scalars, the ``axis`` argument is not passed. For multiple input tensors, the function
    must follow
    `Numpy broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

    Args:
        description: Description string for the operation in einx notation.
        tensors: Input tensors or tensor factories matching the description string.
        op: Backend operation. Is called with ``op(tensor, axis=...)``. If ``op`` is a string,
            retrieves the attribute of ``backend`` with the same name.
        kwargs: Additional keyword arguments that are passed to ``op``.
        backend: Backend to use for all operations. If None, determines the backend from the input
            tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults to True.
        graph: Whether to return the graph representation of the operation instead of computing the
            result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the operation if ``graph=False``, otherwise the graph
        representation of the operation.

    Examples:
        Reverse order of elements along an axis:

        >>> x = np.random.uniform(size=(16, 20))
        >>> einx.vmap_with_axis("a [b] -> a [b]", x, op=np.flip).shape
        (16, 20)

        Roll elements along two axes:

        >>> x = np.random.uniform(size=(16, 20))
        >>> einx.vmap_with_axis(
        ...     "a ([b c]) -> a ([b c])",
        ...     x,
        ...     op=partial(np.roll, shift=(2, 2)),
        ...     b=2,
        ... ).shape
        (16, 20)

        Compute sum along axis:

        >>> x = np.random.uniform(size=(16, 20))
        >>> einx.vmap_with_axis("a ([b] c) -> c a", x, op=np.sum, b=2).shape
        (16, 20)
    """
    exprs_in, exprs_out = parse(
        description, *[einx.tracer.get_shape(tensor) for tensor in tensors], cse=cse, **parameters
    )
    tensors, exprs_out = vmap_with_axis_stage3(
        exprs_in, tensors, exprs_out, op=op, kwargs=kwargs, backend=backend
    )
    return tensors[0] if len(exprs_out) == 1 else tensors


vmap_with_axis.parse = parse


@einx.traceback_util.filter
def flip(
    description: str,
    tensor: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
):
    """Specialization of :func:`einx.vmap_with_axis` with ``op="flip"``."""
    return vmap_with_axis(description, tensor, op="flip", backend=backend, cse=cse, **parameters)


@einx.traceback_util.filter
def roll(
    description: str,
    tensor: einx.Tensor,
    shift: Union[int, Tuple[int]],
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
):
    """Specialization of :func:`einx.vmap_with_axis` with ``op="roll"`` and
    ``kwargs={"shift": shift}``.
    """
    return vmap_with_axis(
        description,
        tensor,
        op="roll",
        backend=backend,
        kwargs={"shift": shift},
        cse=cse,
        **parameters,
    )


@einx.traceback_util.filter
def softmax(
    description: str,
    tensor: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
):
    """Specialization of :func:`einx.vmap_with_axis` with ``op="softmax"``"""
    return vmap_with_axis(description, tensor, op="softmax", backend=backend, cse=cse, **parameters)


@einx.traceback_util.filter
def log_softmax(
    description: str,
    tensor: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
):
    """Specialization of :func:`einx.vmap_with_axis` with ``op="log_softmax"``"""
    return vmap_with_axis(
        description, tensor, op="log_softmax", backend=backend, cse=cse, **parameters
    )

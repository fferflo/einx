import einx
from functools import partial
from . import util
import numpy as np
from typing import Callable, Union
import numpy.typing as npt


def _index(*tensors, update, layout, expr_update_inner, expr_common, op=None, backend=None):
    assert backend is not None
    if update:
        tensor_in = tensors[0]
        tensor_coordinates = tensors[1:-1]
        tensor_update = tensors[-1]
    else:
        tensor_in = tensors[0]
        tensor_coordinates = tensors[1:]

    # Split multi-dim coordinate tensors into single-dim coordinate tensors
    layout = [
        (tensor, coordinate_axis_name, expr_coord, ndim)
        for tensor, (coordinate_axis_name, expr_coord, ndim) in zip(tensor_coordinates, layout)
    ]
    layout2 = []
    for tensor_coord, coordinate_axis_name, expr_coord, ndim in layout:
        axis_names = [
            axis.name for axis in expr_coord.all() if isinstance(axis, einx.expr.stage3.Axis)
        ]
        axis = (
            axis_names.index(coordinate_axis_name) if coordinate_axis_name in axis_names else None
        )
        if axis is None:
            assert ndim == 1
            layout2.append((tensor_coord, coordinate_axis_name, expr_coord))
        else:
            axes = [axis for axis in expr_coord.all() if isinstance(axis, einx.expr.stage3.Axis)]
            del axes[axis]
            expr_coord = einx.expr.stage3.List.maybe(axes)
            for i in range(ndim):
                layout2.append((
                    tensor_coord[(slice(None),) * axis + (i,)],
                    coordinate_axis_name,
                    expr_coord,
                ))
    assert len(layout2) == tensor_in.ndim
    layout = layout2

    # Transpose coordinate and update tensors to match common coordinate expression
    def transpose(tensor, expr):
        return util.transpose_broadcast(
            expr, tensor, expr_common, broadcast=False, backend=backend
        )[0]

    tensor_coordinates = tuple(
        transpose(tensor, expr) for tensor, coordinate_axis_name, expr in layout
    )
    if update:
        tensor_update = transpose(tensor_update, expr_update_inner)

    return (
        op(tensor_in, tensor_coordinates)
        if not update
        else op(tensor_in, tensor_coordinates, tensor_update)
    )


@einx.jit(
    trace=lambda t, c: lambda exprs_in, tensors_in, expr_out, **kwargs: c(
        exprs_in, [t(x) for x in tensors_in], expr_out, **kwargs
    )
)
def index_stage3(exprs_in, tensors_in, expr_out, *, update, op=None, backend=None):
    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensors_in)}")
    for expr in exprs_in[0]:
        if isinstance(expr, einx.expr.stage3.Axis) and expr.is_unnamed and expr.value == 1:
            raise ValueError("First expression cannot contain unnamed axes with value 1")
    for root in list(exprs_in) + [expr_out]:
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    if not update:
        # Ensure that no brackets exist in output expression
        for expr in expr_out.all():
            if einx.expr.stage3.is_marked(expr):
                raise ValueError("Brackets in the output expression are not allowed")
    if update:
        axis_names = {
            axis.name
            for root in exprs_in[:-1]
            for axis in root.all()
            if isinstance(axis, einx.expr.stage3.Axis)
        }
        for axis in exprs_in[-1].all():
            if isinstance(axis, einx.expr.stage3.Axis) and axis.name not in axis_names:
                raise ValueError(
                    f"Update expression cannot contain axes that are not in the "
                    f"coordinate or tensor expressions: {axis.name}"
                )

    # Call tensor factories
    def get_name(s):
        if s == "get_at":
            return "embedding"
        else:
            return s

    tensors_in = [
        einx.tracer.call_factory(
            tensor,
            expr.shape,
            backend,
            name=get_name(util._op_to_str(op)),
            init=util._op_to_str(op),
        )
        for tensor, expr in zip(tensors_in, exprs_in)
    ]
    tensors_in = backend.all_to_tensor(tensors_in)

    expr_tensor = exprs_in[0]
    exprs_coordinates = exprs_in[1:-1] if update else exprs_in[1:]
    if update:
        expr_update = exprs_in[-1]

    layout = []
    total_ndim = 0
    for expr_coord in exprs_coordinates:
        marked_coordinate_axes = [
            expr
            for expr in expr_coord.all()
            if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr)
        ]
        if len(marked_coordinate_axes) > 1:
            raise ValueError(
                f"Expected at most one coordinate axis in coordinate expression, "
                f"got {len(marked_coordinate_axes)} in '{expr_coord}'"
            )
        ndim = marked_coordinate_axes[0].value if len(marked_coordinate_axes) == 1 else 1
        coordinate_axis_name = (
            marked_coordinate_axes[0].name
            if len(marked_coordinate_axes) == 1
            and (not marked_coordinate_axes[0].is_unnamed or marked_coordinate_axes[0].value != 1)
            else None
        )
        layout.append((coordinate_axis_name, ndim))
        total_ndim += ndim

    marked_tensor_axis_names = {
        expr.name
        for expr in expr_tensor.all()
        if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr)
    }
    if len(marked_tensor_axis_names) != total_ndim:
        raise ValueError(
            f"Expected {total_ndim} marked axes in tensor, got {len(marked_tensor_axis_names)}"
        )

    if update:
        marked_update_axis_names = {
            expr.name
            for expr in expr_update.all()
            if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr)
        }
        if len(marked_update_axis_names) != 0:
            raise ValueError("Update expression cannot contain marked axes")

    # Add markers around axes in coordinates and update that are not in tensor
    tensor_axis_names = {
        expr.name for expr in expr_tensor.all() if isinstance(expr, einx.expr.stage3.Axis)
    }
    new_marked_axis_names = set()

    def replace(expr):
        if (
            isinstance(expr, einx.expr.stage3.Axis)
            and expr.name not in tensor_axis_names
            and not einx.expr.stage3.is_marked(expr)
        ):
            new_marked_axis_names.add(expr.name)
            return einx.expr.stage3.Marker(expr.__deepcopy__())

    exprs_coordinates = [einx.expr.stage3.replace(expr, replace) for expr in exprs_coordinates]
    expr_update = einx.expr.stage3.replace(expr_update, replace) if update else None

    # Add markers around those same axes in output and update
    def replace(expr):
        if (
            isinstance(expr, einx.expr.stage3.Axis)
            and expr.name in new_marked_axis_names
            and not einx.expr.stage3.is_marked(expr)
        ):
            return einx.expr.stage3.Marker(expr.__deepcopy__())

    expr_out = einx.expr.stage3.replace(expr_out, replace)
    if update:
        expr_update = einx.expr.stage3.replace(expr_update, replace)

    # If updating: Add markers around axes in output that are also marked
    # in tensor (and are not broadcasted axes)
    if update:

        def replace(expr):
            if (
                isinstance(expr, einx.expr.stage3.Axis)
                and expr.name in marked_tensor_axis_names
                and not einx.expr.stage3.is_marked(expr)
            ):
                return einx.expr.stage3.Marker(expr.__deepcopy__())

        expr_out = einx.expr.stage3.replace(expr_out, replace)

    def to_inner(expr):
        expr = einx.expr.stage3.get_marked(expr)
        return util.flatten([expr])[0]

    exprs_coordinates_inner = [to_inner(expr) for expr in exprs_coordinates]
    expr_update_inner = to_inner(expr_update) if update else None

    # Find common expression for coordinates and update in vmapped function
    layout = [
        (coordinate_axis_name, expr_coord, ndim)
        for expr_coord, (coordinate_axis_name, ndim) in zip(exprs_coordinates_inner, layout)
    ]
    if update:
        layout2 = layout + [(None, expr_update_inner, None)]
        longest = sorted(layout2, key=lambda x: len(x[1].shape))[-1]
        all_axes = [
            axis
            for axis in longest[1].all()
            if isinstance(axis, einx.expr.stage3.Axis) and not axis.name == longest[0]
        ]
        axes_names = {axis.name for axis in all_axes}
        for coordinate_axis_name, expr_coord, _ in layout2:
            for axis in expr_coord.all():
                if (
                    isinstance(axis, einx.expr.stage3.Axis)
                    and axis.name != coordinate_axis_name
                    and axis.name not in axes_names
                ):
                    axes_names.add(axis.name)
                    all_axes.append(axis)
        expr_common = einx.expr.stage3.List.maybe(all_axes)
    else:
        expr_common = einx.expr.stage3.get_marked(util.flatten([expr_out])[0])

    # Construct vmapped indexing function
    if isinstance(op, str):
        op = getattr(backend, op)
    op = partial(
        _index,
        op=op,
        update=update,
        layout=layout,
        expr_common=expr_common,
        expr_update_inner=expr_update_inner,
        backend=backend,
    )
    op = einx.trace(op)

    exprs_in = [expr_tensor] + exprs_coordinates + ([expr_update] if update else [])
    tensors_out, exprs_out = einx.vmap_stage3(
        exprs_in, tensors_in, [expr_out], op=op, flat=True, backend=backend
    )
    assert len(tensors_out) == 1 and len(exprs_out) == 1
    return tensors_out[0], exprs_out[0]


@einx.lru_cache
def parse(description, *tensor_shapes, update, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(
        description, parameters
    )

    op = einx.expr.stage1.parse_op(description)

    # Implicitiy determine output expression
    if len(op) == 1:
        if update:
            op = einx.expr.stage1.Op([
                op[0],
                einx.expr.stage1.Args([op[0][0]]),
            ])
        else:
            raise ValueError("Operation string must contain an output expression")

    if len(op[0]) != len(tensor_shapes):
        raise ValueError(f"Expected {len(op[0])} input tensors, but got {len(tensor_shapes)}")
    if len(op[1]) != 1:
        raise ValueError(f"Expected 1 output expression, but got {len(op[1])}")

    def after_stage2(exprs1, exprs2):
        for expr in exprs1[0].all():
            if (
                isinstance(expr, einx.expr.stage2.UnnamedAxis)
                and expr.value == 1
                and einx.expr.stage2.is_marked(expr)
            ):
                raise ValueError("First expression cannot contain unnamed axes with value 1")
        tensor_marked_axes = [
            expr
            for expr in exprs1[0].all()
            if isinstance(expr, (einx.expr.stage2.NamedAxis, einx.expr.stage2.UnnamedAxis))
            and einx.expr.stage2.is_marked(expr)
        ]
        ndim = len(tensor_marked_axes)

        concat_this = []

        coord_exprs = exprs1[1 : len(tensor_shapes)]
        if update:
            coord_exprs = coord_exprs[:-1]
        for expr in coord_exprs:
            marked_coordinate_axes = [
                expr
                for expr in exprs1[1].all()
                if isinstance(expr, (einx.expr.stage2.NamedAxis, einx.expr.stage2.UnnamedAxis))
                and einx.expr.stage2.is_marked(expr)
            ]
            if len(marked_coordinate_axes) > 1:
                raise ValueError(
                    f"Expected at most one marked axis per coordinate tensor"
                    f", got {len(marked_coordinate_axes)}"
                )
            elif len(marked_coordinate_axes) == 1:
                if isinstance(marked_coordinate_axes[0], einx.expr.stage2.NamedAxis):
                    concat_this.append(einx.expr.stage1.NamedAxis(marked_coordinate_axes[0].name))
                else:
                    concat_this.append(
                        einx.expr.stage1.UnnamedAxis(marked_coordinate_axes[0].value)
                    )
            else:
                concat_this.append(einx.expr.stage1.UnnamedAxis(1))

        return [
            einx.expr.Equation(
                einx.expr.stage1.Concatenation.maybe(concat_this), np.asarray([ndim])
            )
        ]

    exprs = einx.expr.solve(
        [
            einx.expr.Equation(expr_in, tensor_shape)
            for expr_in, tensor_shape in zip(op[0], tensor_shapes)
        ]
        + [einx.expr.Equation(op[1][0])]
        + [
            einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None)
            for k, v in parameters.items()
        ],
        cse=cse,
        after_stage2=after_stage2,
    )[: len(op[0]) + 1]
    exprs_in, expr_out = exprs[: len(op[0])], exprs[len(op[0])]

    if update:
        # Check that all axes in first input expression also appear in output expression
        axes_in = {
            axis.name for axis in exprs_in[0].all() if isinstance(axis, einx.expr.stage3.Axis)
        }
        axes_out = {axis.name for axis in expr_out.all() if isinstance(axis, einx.expr.stage3.Axis)}
        if not axes_in.issubset(axes_out):
            raise ValueError(
                f"Output expression does not contain all axes from first input expression: "
                f"{axes_in - axes_out}"
            )

    return exprs_in, expr_out


def _has_zero_shape(tensor):
    shape = einx.tracer.get_shape(tensor)
    return shape is not None and any(s == 0 for s in shape)


@einx.traceback_util.filter
@einx.jit(
    trace=lambda t, c: lambda description, *tensors, backend=None, **kwargs: c(
        description, *[t(x) for x in tensors], **kwargs
    )
)
def index(
    description: str,
    *tensors: einx.Tensor,
    op: Callable,
    update: bool,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Updates and/ or returns values from a tensor at the given coordinates.

    * If ``update`` is True: The first tensor receives updates, the last tensor contains the
      updates, and all other tensors represent the coordinates. If the output expression is
      not given, it is assumed to be equal to the first input expression.

    * If ``update`` is False, values are retrieved from the first tensor and the remaining tensors
      contain the coordinates.

    Using multiple coordinate expressions will yield the same output as concatenating
    the coordinate expressions along the coordinate axis first.

    Args:
        description: Description string for the operation in einx notation.
        *tensors: Tensors that the operation will be applied to.
        op: The update/gather function. If ``op`` is a string, retrieves the attribute of
            ``backend`` with the same name.
        update: Whether to update the tensor or return values from the tensor.
        backend: Backend to use for all operations. If None, determines the backend from the
            input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults
            to True.
        graph: Whether to return the graph representation of the operation instead of computing
            the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the update/ gather operation if ``graph=False``, otherwise the graph
        representation of the operation.

    Examples:
        Get values from a batch of images (different indices per image):

        >>> tensor = np.random.uniform(size=(4, 128, 128, 3))
        >>> coordinates = np.ones((4, 100, 2))
        >>> einx.get_at("b [h w] c, b p [2] -> b p c", tensor, coordinates).shape
        (4, 100, 3)

        >>> tensor = np.random.uniform(size=(4, 128, 128, 3))
        >>> coordinates_x = np.ones((4, 100), "int32")
        >>> coordinates_y = np.ones((4, 100), "int32")
        >>> einx.get_at(
        ...     "b [h w] c, b p, b p -> b p c",
        ...     tensor,
        ...     coordinates_x,
        ...     coordinates_y,
        ... ).shape
        (4, 100, 3)

        Set values in a batch of images (same indices per image):

        >>> tensor = np.random.uniform(size=(4, 128, 128, 3))
        >>> coordinates = np.ones((100, 2), "int32")
        >>> updates = np.random.uniform(size=(100, 3))
        >>> einx.set_at(
        ...     "b [h w] c, p [2], p c -> b [h w] c", tensor, coordinates, updates
        ... ).shape
        (4, 128, 128, 3)

        >>> tensor = np.random.uniform(size=(4, 128, 128, 3))
        >>> coordinates_x = np.ones((100,), "int32")
        >>> coordinates_y = np.ones((100,), "int32")
        >>> updates = np.random.uniform(size=(100, 3))
        >>> einx.set_at(
        ...     "b [h w] c, p, p, p c -> b [h w] c",
        ...     tensor,
        ...     coordinates_x,
        ...     coordinates_y,
        ...     updates,
        ... ).shape
        (4, 128, 128, 3)
    """
    if update and any(_has_zero_shape(tensor) for tensor in tensors[1:]):
        # Skip update if no coordinates are given
        return tensors[0]
    exprs_in, expr_out = parse(
        description,
        *[einx.tracer.get_shape(tensor) for tensor in tensors],
        update=update,
        cse=cse,
        **parameters,
    )
    tensor, expr_out = index_stage3(
        exprs_in, tensors, expr_out, op=op, update=update, backend=backend
    )
    return tensor


index.parse = parse


@einx.traceback_util.filter
def get_at(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.index` with ``op="get_at"`` and ``update=False``"""
    return index(
        description, *tensors, op="get_at", update=False, backend=backend, cse=cse, **parameters
    )


@einx.traceback_util.filter
def set_at(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.index` with ``op="set_at"`` and ``update=True``"""
    return index(
        description, *tensors, op="set_at", update=True, backend=backend, cse=cse, **parameters
    )


@einx.traceback_util.filter
def add_at(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.index` with ``op="add_at"`` and ``update=True``"""
    return index(
        description, *tensors, op="add_at", update=True, backend=backend, cse=cse, **parameters
    )


@einx.traceback_util.filter
def subtract_at(
    description: str,
    *tensors: einx.Tensor,
    backend: Union[einx.Backend, str, None] = None,
    cse: bool = True,
    **parameters: npt.ArrayLike,
) -> einx.Tensor:
    """Specialization of :func:`einx.index` with ``op="subtract_at"`` and ``update=True``"""
    return index(
        description, *tensors, op="subtract_at", update=True, backend=backend, cse=cse, **parameters
    )

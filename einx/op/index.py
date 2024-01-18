import einx
from functools import partial

from ..type_util import assign_global
from . import util
import numpy as np

_op_names = ["get_at", "set_at", "add_at", "subtract_at"]

def _index(tensor, coordinates, update=None, axis=None, op=None, backend=None):
    if axis is None:
        axis = 0
        coordinates = coordinates[None]
    if isinstance(op, str):
        x = backend
        for name in op.split("."):
            x = getattr(x, name)
        op = x
    assert coordinates.shape[axis] == tensor.ndim

    coordinates = tuple(coordinates[(slice(None),) * axis + (i,)] for i in range(tensor.ndim))
    return op(tensor, coordinates) if update is None else op(tensor, coordinates, update)

@einx.lru_cache(trace=lambda t, c: lambda exprs_in, tensors_in, expr_out, op=None, backend=None: c(exprs_in, [t(x) for x in tensors_in], expr_out, op=op))
def index_stage3(exprs_in, tensors_in, expr_out, op=None, backend=None):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    elif isinstance(backend, str):
        backend = einx.backend.get(backend)
    if op is None:
        raise TypeError("op cannot be None")
    with_update = len(exprs_in) == 3
    for expr in exprs_in[0]:
        if isinstance(expr, einx.expr.stage3.Axis) and expr.is_unnamed and expr.value == 1:
            raise ValueError("First expression cannot contain unnamed axes with value 1")
    for root in list(exprs_in) + [expr_out]:
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    if not with_update:
        for expr in expr_out.all():
            if einx.expr.stage3.is_marked(expr):
                raise ValueError("Brackets in the output expression are not allowed")
    exprs_in = list(exprs_in)

    marked_coordinate_axes = [expr for expr in exprs_in[1].all() if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr)]
    if len(marked_coordinate_axes) > 1:
        raise ValueError(f"Expected at most one coordinate axis in the second expression, got {len(marked_coordinate_axes)}")
    ndim = marked_coordinate_axes[0].value if len(marked_coordinate_axes) == 1 else 1
    coordinate_axis_name = marked_coordinate_axes[0].name if len(marked_coordinate_axes) == 1 and (not marked_coordinate_axes[0].is_unnamed or marked_coordinate_axes[0].value != 1) else None

    marked_tensor_axis_names = set(expr.name for expr in exprs_in[0].all() if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr))
    if len(marked_tensor_axis_names) != ndim:
        raise ValueError(f"Expected {ndim} marked axes in tensor, got {len(marked_tensor_axis_names)}")
    if with_update:
        marked_update_axis_names = set(expr.name for expr in exprs_in[2].all() if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr))
        if len(marked_update_axis_names) != 0:
            raise ValueError(f"Update expression cannot contain marked axes")
    else:
        marked_update_axis_names = set()

    # Add markers around axes in coordinates that are not in tensor
    tensor_axis_names = set(expr.name for expr in exprs_in[0].all() if isinstance(expr, einx.expr.stage3.Axis))
    new_marked_axis_names = set()
    def replace(expr):
        if isinstance(expr, einx.expr.stage3.Axis) and not expr.name in tensor_axis_names and not einx.expr.stage3.is_marked(expr):
            new_marked_axis_names.add(expr.name)
            return einx.expr.stage3.Marker(expr.__deepcopy__())
    exprs_in[1] = einx.expr.stage3.replace(exprs_in[1], replace)

    # Add markers around those same axes in output and update
    def replace(expr):
        if isinstance(expr, einx.expr.stage3.Axis) and expr.name in new_marked_axis_names and not einx.expr.stage3.is_marked(expr):
            return einx.expr.stage3.Marker(expr.__deepcopy__())
    expr_out = einx.expr.stage3.replace(expr_out, replace)
    if with_update:
        exprs_in[2] = einx.expr.stage3.replace(exprs_in[2], replace)

    # If updating: Add markers around axes in output that are also marked in tensor
    if with_update:
        def replace(expr):
            if isinstance(expr, einx.expr.stage3.Axis) and expr.name in marked_tensor_axis_names and not einx.expr.stage3.is_marked(expr):
                return einx.expr.stage3.Marker(expr.__deepcopy__())
        expr_out = einx.expr.stage3.replace(expr_out, replace)

    # Find index into coordinates that will arrive at _index
    new_marked_coordinate_axis_names = [expr.name for expr in exprs_in[1].all() if isinstance(expr, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(expr)]
    axis = new_marked_coordinate_axis_names.index(coordinate_axis_name) if not coordinate_axis_name is None else None

    # Call tensor factories
    def get_name(s):
        if s == "get_at":
            return "embedding"
        else:
            return s
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend, name=get_name(str(op)), init=str(op)) for tensor, expr in zip(tensors_in, exprs_in)]

    tensors_out, exprs_out = einx.vmap_stage3(exprs_in, tensors_in, [expr_out], op=_index, flat=True, kwargs={"axis": axis, "op": op}, pass_backend=True, backend=backend)
    assert len(tensors_out) == 1 and len(exprs_out) == 1
    return tensors_out[0], exprs_out[0]

@einx.lru_cache
def parse(description, *tensors_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    description = description.split("->")
    if len(description) != 2:
        raise ValueError("Operation string must contain exactly one '->'")
    exprs_in, expr_out = description
    exprs_in = exprs_in.split(",")
    if not len(exprs_in) in [2, 3]:
        raise ValueError(f"Expected 2 or 3 input expressions, got {len(exprs_in)}")
    if "," in expr_out:
        raise ValueError("Only a single output expression is allowed")
    if len(tensors_shapes) != len(exprs_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensors_shapes)}")

    def after_stage2(exprs1, exprs2):
        for expr in exprs1[0].all():
            if isinstance(expr, einx.expr.stage2.UnnamedAxis) and expr.value == 1 and einx.expr.stage2.is_marked(expr):
                raise ValueError("First expression cannot contain unnamed axes with value 1")
        tensor_marked_axes = [expr for expr in exprs1[0].all() if isinstance(expr, (einx.expr.stage2.NamedAxis, einx.expr.stage2.UnnamedAxis)) and einx.expr.stage2.is_marked(expr)]
        ndim = len(tensor_marked_axes)

        marked_coordinate_axes = [expr for expr in exprs1[1].all() if isinstance(expr, (einx.expr.stage2.NamedAxis, einx.expr.stage2.UnnamedAxis)) and einx.expr.stage2.is_marked(expr)]
        if len(marked_coordinate_axes) > 1:
            raise ValueError(f"Expected at most one coordinate axis, got {len(marked_coordinate_axes)}")
        elif len(marked_coordinate_axes) == 1 and isinstance(marked_coordinate_axes[0], einx.expr.stage2.NamedAxis):
            coordinate_axis = einx.expr.stage1.NamedAxis(marked_coordinate_axes[0].name)
            return [einx.expr.Equation(coordinate_axis, np.asarray([ndim]))]
        else:
            return []

    exprs = einx.expr.solve(
            [einx.expr.Equation(expr_in, tensor_shape) for expr_in, tensor_shape in zip(exprs_in, tensors_shapes)] \
          + [einx.expr.Equation(expr_out)] \
          + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
        cse=cse,
        after_stage2=after_stage2,
    )[:len(exprs_in) + 1]
    exprs_in, expr_out = exprs[:len(exprs_in)], exprs[len(exprs_in)]

    return exprs_in, expr_out

@einx.lru_cache(trace=lambda t, c: lambda description, *tensors, backend=None, **kwargs: c(description, *[t(x) for x in tensors], **kwargs))
def index_stage0(description, *tensors, op=None, backend=None, cse=True, **parameters):
    exprs_in, expr_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensor, expr_out = index_stage3(exprs_in, tensors, expr_out, op=op, backend=backend)
    return tensor

def index(arg0, *args, **kwargs):
    """Updates and/ or returns values from an array at the given coordinates. Specializes :func:`einx.vmap`.

    The `description` argument specifies the input and output expressions and must meet one of the following formats:

    1. ``tensor, update, coordinates -> output``
       when modifying values in the tensor.
    2. ``tensor, coordinates -> output``
       when only returning values from the tensor.

    Brackets in the ``tensor`` expression mark the axes that will be indexed. Brackets in the ``coordinates`` expression mark the single coordinate axis. All other
    axes are considered batch axes.

    Args:
        description: Description string in Einstein notation (see above).
        tensor: The tensor that values will be updates/ gathered from.
        coordinates: The tensor that contains the coordinates of the values to be gathered.
        update: The tensor that contains the update values, or None. Defaults to None.
        op: The update/gather function. If `op` is a string, retrieves the attribute of `backend` with the same name.
        backend: Backend to use for all operations. If None, determines the backend from the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults to True.
        graph: Whether to return the graph representation of the operation instead of computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the update/ gather operation if `graph=False`, otherwise the graph representation of the operation.

    Examples:
        Get values from a batch of images (different indices per image):

        >>> tensor = np.random.uniform(size=(4, 128, 128, 3))
        >>> coordinates = np.random.uniform(size=(4, 100, 2))
        >>> einx.get_at("b [h w] c, b p [2] -> b p c", tensor, coordinates).shape
        (4, 100, 3)

        Set values in a batch of images (same indices per image):

        >>> tensor = np.random.uniform(size=(4, 128, 128, 3))
        >>> coordinates = np.random.uniform(size=(100, 2))
        >>> updates = np.random.uniform(size=(100, 3))
        >>> einx.set_at("b [h w] c, p [2], p c -> b [h w] c", tensor, coordinates, updates).shape
        (4, 128, 128, 3)
    """
    if isinstance(arg0, str):
        return index_stage0(arg0, *args, **kwargs)
    else:
        return index_stage3(arg0, *args, **kwargs)
index._op_names = _op_names
index.parse = parse

def _make(name):
    def func(*args, **kwargs):
        return index(*args, op=name, **kwargs)
    func.__name__ = name
    func.__doc__ = f"Alias for :func:`einx.index` with ``op=\"{name}\"``"
    # globals()[name] = func
    assign_global(
        name,
        func,
        globals(),
        "(description: str, tensor: TArray, coordinates: TArray, update: t.Optional[TArray] = None, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray",
        __file__,
    )

for name in _op_names:
    _make(name)
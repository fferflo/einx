import einx._src.namedtensor.stage3 as stage3
from einx._src.namedtensor import NamedTensor
from ._util import _squeeze_transpose_broadcast
from ._util import _ensure_output
from ._util import _stack
from ._util import _unsqueeze
from ._util import _unravel
import functools
import numpy as np
from einx._src.util.functools import use_name_of
import einx._src.tracer as tracer
import einx._src.adapter as adapter
from einx._src.frontend.errors import OperationNotSupportedError


def _expr_to_axis(expr):
    idxs = []
    for idx, axis in enumerate(expr):
        if stage3.is_in_brackets(axis):
            idxs.append(idx)
    return tuple(idxs)


def _join_exprs(exprs):
    axisname_to_value = {axis.name: axis.value for expr in exprs for axis in expr.nodes() if isinstance(axis, stage3.Axis)}

    axes = [[expr for expr in expr.nodes() if isinstance(expr, stage3.Axis) and expr.value != 1] for expr in exprs]

    def take_one():
        def get_count(name):
            return sum((1 if name == axis.name else 0) for axes2 in axes for axis in axes2)

        first_axisnames = list({axes2[0].name for axes2 in axes if len(axes2) > 0})
        counts = [get_count(name) for name in first_axisnames]
        idx = np.argmax(counts)
        axisname = first_axisnames[idx]
        return axisname

    def remove(axes, name):
        return [axis for axis in axes if axis.name != name]

    joined_axisnames = []
    while any(len(axes2) > 0 for axes2 in axes):
        axisname = take_one()
        joined_axisnames.append(axisname)
        axes = [remove(axes2, axisname) for axes2 in axes]

    return stage3.List.create([stage3.Axis(name, axisname_to_value[name]) for name in joined_axisnames])


def _ravel(classical, expr_tensor, coords, expr_coords, expr_out, verbose=False):
    # (1) Example shapes here:
    # expr_tensor: a [b c] d
    # coords: a [1] d e, [1] d f
    # expr_out: a d e f

    if verbose:
        print("Step 0")
        print(f" tensor: {expr_tensor}")
        print(f" coords: {[str(c) for c in expr_coords]}")
        print(f" out: {expr_out}")

    # Convert list of ND coords to list of 1D coords without brackets
    coords2 = []
    expr_coords2 = []
    for coord, expr_coord in zip(coords, expr_coords, strict=False):
        axis = _expr_to_axis(expr_coord)
        expr_coord_without_brackets = stage3.remove(expr_coord, stage3.Brackets, keep_children=False)
        if len(axis) == 0:
            # No axis is marked -> contains coordinate for 1 axis
            coords2.append(coord)
            expr_coords2.append(expr_coord)
        elif len(axis) == 1:
            # One axis is marked -> contains coordinates for ndim axes
            shape = tuple(a.value for a in expr_coord)
            ndim = shape[axis[0]]
            if ndim == 1:
                coord = classical.reshape(coord, shape[: axis[0]] + shape[axis[0] + 1 :])
                coords2.append(coord)
                expr_coords2.append(expr_coord_without_brackets)
            else:
                assert ndim > 1
                for i in range(ndim):
                    coord_i = classical.get_at(coord, i, axis=axis[0])
                    coords2.append(coord_i)
                    expr_coords2.append(expr_coord_without_brackets)
        else:
            raise AssertionError()
    coords = coords2
    expr_coords = expr_coords2

    if verbose:
        print("Step 1")
        print(f" tensor: {expr_tensor}")
        print(f" coords: {[str(c) for c in expr_coords]}")
        print(f" out: {expr_out}")

    # (2) Example shapes here:
    # expr_tensor: a [b c] d
    # coords: a d e, d f
    # expr_out: a d e f

    # Get coord dtype
    if hasattr(classical, "dtype"):
        tensor_types = (tracer.signature.classical.Tensor, classical.tensor) if isinstance(classical.tensor, type) else (tracer.signature.classical.Tensor,)
        for coord in coords:
            if isinstance(coord, tensor_types):
                coord_dtype = classical.dtype(coord)
                break
        else:
            coord_dtype = classical.dtype(coords[0])
    else:
        coord_dtype = "int32"

    # Add aranged coords for all vectorized axes
    coords2 = []
    expr_coords2 = []
    for axis in expr_tensor:
        assert isinstance(axis, stage3.Axis)
        if stage3.is_in_brackets(axis):
            # Axis is in brackets -> use coordinate argument
            coords2.append(coords[0])
            expr_coords2.append(expr_coords[0])
            coords = coords[1:]
            expr_coords = expr_coords[1:]
        else:
            # Axis is not in brackets -> use np.arange
            coord = classical.arange(axis.value, dtype=coord_dtype)
            expr_coord = axis.__deepcopy__()
            coords2.append(coord)
            expr_coords2.append(expr_coord)
    coords = coords2
    expr_coords = expr_coords2

    if verbose:
        print("Step 2")
        print(f" tensor: {expr_tensor}")
        print(f" coords: {[str(c) for c in expr_coords]}")
        print(f" out: {expr_out}")

    # (3) Example shapes here:
    # expr_tensor: a [b c] d
    # coords: a, a d e, d f, d
    # expr_out: a d e f

    # Compute indices into flattened tensor by ravelling coordinates (-> row-major formula)
    multiplier = 1
    coords2 = []
    for coord, axis in reversed(list(zip(coords, expr_tensor, strict=False))):
        if multiplier != 1:
            coord = classical.multiply(coord, multiplier)
        coords2.insert(0, coord)
        multiplier *= axis.value
    coords = coords2

    elementwise_add = elementwise(classical.add, classical)
    coords = elementwise_add(*[NamedTensor(coord, expr) for coord, expr in zip(coords, expr_coords, strict=False)], out=expr_out).value

    # (4) Example shapes here:
    # expr_tensor: a [b c] d
    # coords: a d e f
    # expr_out: a d e f

    return coords


def reduce(op, expected_type=None):
    op_in = op

    @use_name_of(op)
    def inner(tensor, out, **kwargs):
        expr_in = tensor.expr
        tensor = tensor.value
        expr_out = stage3.remove(expr_in, stage3.Brackets, keep_children=False)

        op = _ensure_output(op_in, (expr_out.shape,), expected_type=expected_type)
        tensor = op(tensor, axis=_expr_to_axis(expr_in), **kwargs)

        return NamedTensor(tensor, expr_out)

    return inner


def preserve_shape(op):
    @use_name_of(op)
    def inner(tensor, out, **kwargs):
        expr = tensor.expr
        tensor = tensor.value

        tensor = op(tensor, axis=_expr_to_axis(expr), **kwargs)

        return NamedTensor(tensor, expr)

    return inner


def argfind(op, classical):
    @use_name_of(op)
    def inner(tensor, out, **kwargs):
        expr = tensor.expr
        tensor = tensor.value

        # Identify marked axes
        marked_axes = [idx for idx, axis in enumerate(expr) if stage3.is_in_brackets(axis)]
        ravel_shape = tuple(axis.value for axis in expr if stage3.is_in_brackets(axis))
        marked_axes_length = int(np.prod([axis.value for axis in expr if stage3.is_in_brackets(axis)]))

        # Rearrange marked axes to the back if they are not contiguous
        is_contiguous = len(marked_axes) < 2 or all(marked_axes[i] + 1 == marked_axes[i + 1] for i in range(len(marked_axes) - 1))
        if not is_contiguous:
            # Transpose all marked axes to the back
            perm = []
            new_expr = []
            for idx, axis in enumerate(expr):
                if not stage3.is_in_brackets(axis):
                    perm.append(idx)
                    new_expr.append(axis)
            for idx, axis in enumerate(expr):
                if stage3.is_in_brackets(axis):
                    perm.append(idx)
                    new_expr.append(stage3.Brackets(axis))
            marked_axes = list(range(len(expr) - len(marked_axes), len(expr)))
            expr = stage3.List.create(new_expr)
            tensor = classical.transpose(tensor, perm)
        unmarked_expr = stage3.remove(expr, stage3.Brackets, keep_children=False)

        # Flatten all marked axes to a single axis
        new_shape = []
        for axis in range(tensor.ndim):
            if axis in marked_axes:
                if axis == marked_axes[0]:
                    new_shape.append(marked_axes_length)
            else:
                new_shape.append(tensor.shape[axis])
        tensor = classical.reshape(tensor, new_shape)
        axis = marked_axes[0]

        # Apply argfind operation to remove marked axis
        tensor = op(tensor, axis=axis, **kwargs)
        assert tensor.shape == unmarked_expr.shape

        # Rearrange to output expression
        out_expr_without_brackets = stage3.remove(out, stage3.Brackets, keep_children=False)
        unmarked_expr, tensor = _squeeze_transpose_broadcast(classical, unmarked_expr, tensor, out_expr_without_brackets)

        # Unravel and stack along marked axis in output
        marked_out_axes_pos = [idx for idx, axis in enumerate(out) if isinstance(axis, stage3.Axis) and stage3.is_in_brackets(axis)]
        if len(marked_out_axes_pos) == 0:
            axis = None
        elif len(marked_out_axes_pos) == 1:
            axis = marked_out_axes_pos[0]
        else:
            raise ValueError("At most one marked axis is allowed in the output expression.")
        tensor = _unravel(classical, tensor, ravel_shape, axis=axis)

        return NamedTensor(tensor, out)

    return inner


def elementwise(op, classical, expected_type=None):
    op_in = op

    @use_name_of(op)
    def inner(*tensors, out, **kwargs):
        exprs_in = [t.expr for t in tensors]
        tensors = [t.value for t in tensors]

        # (1) Example shapes here:
        # tensors: a, b
        # out: a b c

        # Transpose tensors and insert unitary dimensions to match output expression
        tensors2 = []
        exprs_in2 = []
        for tensor, expr_in in zip(tensors, exprs_in, strict=False):
            expr_in, tensor = _squeeze_transpose_broadcast(classical, expr_in, tensor, out, broadcast_to_unitary=True)
            tensors2.append(tensor)
            exprs_in2.append(expr_in)
        tensors = tensors2
        exprs_in = exprs_in2

        # (2) Example shapes here:
        # tensors: a 1 1, 1 b 1
        # out: a b c

        # Apply operation
        in_axes = [[axis for axis in expr.nodes() if isinstance(axis, stage3.Axis)] for expr in exprs_in]
        assert len({len(a) for a in in_axes}) == 1
        out_axes = []
        for i in range(len(in_axes[0])):
            in_axes_i = [axes[i] for axes in in_axes]
            idx = np.argmax([axis.value for axis in in_axes_i])
            out_axis_i = in_axes_i[idx].__deepcopy__()
            out_axes.append(out_axis_i)
        expr_out = stage3.List.create(out_axes)
        op = _ensure_output(op_in, (expr_out.shape,), expected_type=expected_type)
        tensor = op(*tensors, **kwargs)

        # (3) Example shapes here:
        # tensors: a b 1
        # out: a b c

        return NamedTensor(tensor, expr_out)

    return inner


def dot(classical):
    def dot(*tensors, out):
        if len(tensors) != 2:
            raise OperationNotSupportedError("dot operation with numpylike backend does not support more than two argument tensors.")
        tensor1, tensor2 = tensors

        # Follows the algorithm described here: https://github.com/jcmgray/einsum_bmm/blob/main/einsum_bmm.py#L139

        id = adapter.namedtensor_from_decomposednamedtensor.id(None, classical)
        expr1 = tensor1.expr
        expr2 = tensor2.expr
        tensor1 = tensor1.value
        tensor2 = tensor2.value

        # Get axes: left, right -> out
        left_axis_names = [axis.name for axis in expr1.nodes() if isinstance(axis, stage3.Axis)]
        right_axis_names = [axis.name for axis in expr2.nodes() if isinstance(axis, stage3.Axis)]
        out_axis_names = [axis.name for axis in out.nodes() if isinstance(axis, stage3.Axis)]

        # Classify axes into: batch, contracted, left-keep, right-keep
        batch_axis_names = []
        contract_axis_names = []
        left_keep_axis_names = []
        right_keep_axis_names = []
        for axis in left_axis_names:
            if axis in right_axis_names:
                if axis in out_axis_names:
                    batch_axis_names.append(axis)
                else:
                    contract_axis_names.append(axis)
            else:
                assert axis in out_axis_names
                left_keep_axis_names.append(axis)
        for axis in right_axis_names:
            if axis not in left_axis_names:
                assert axis in out_axis_names
                right_keep_axis_names.append(axis)

        # Build new expressions for left, right, out
        lengths = {axis.name: axis.value for expr in [expr1, expr2, out] for axis in expr.nodes() if isinstance(axis, stage3.Axis)}
        left_matmul_expr = stage3.List.create([
            stage3.FlattenedAxis.create(stage3.List.create([stage3.Axis(name, lengths[name]) for name in batch_axis_names])),
            stage3.FlattenedAxis.create(stage3.List.create([stage3.Axis(name, lengths[name]) for name in left_keep_axis_names])),
            stage3.FlattenedAxis.create(stage3.List.create([stage3.Axis(name, lengths[name]) for name in contract_axis_names])),
        ])
        right_matmul_expr = stage3.List.create([
            stage3.FlattenedAxis.create(stage3.List.create([stage3.Axis(name, lengths[name]) for name in batch_axis_names])),
            stage3.FlattenedAxis.create(stage3.List.create([stage3.Axis(name, lengths[name]) for name in contract_axis_names])),
            stage3.FlattenedAxis.create(stage3.List.create([stage3.Axis(name, lengths[name]) for name in right_keep_axis_names])),
        ])
        out_matmul_expr = stage3.List.create([
            stage3.FlattenedAxis.create(stage3.List.create([stage3.Axis(name, lengths[name]) for name in batch_axis_names])),
            stage3.FlattenedAxis.create(stage3.List.create([stage3.Axis(name, lengths[name]) for name in left_keep_axis_names])),
            stage3.FlattenedAxis.create(stage3.List.create([stage3.Axis(name, lengths[name]) for name in right_keep_axis_names])),
        ])

        # Rearrange tensors to matmul expressions
        left = id(NamedTensor(tensor1, expr1), out=left_matmul_expr).value
        right = id(NamedTensor(tensor2, expr2), out=right_matmul_expr).value
        assert left.ndim == 3 and right.ndim == 3

        # Perform batched matmul
        tensor = classical.matmul(left, right)
        expr = out_matmul_expr

        # Rearrange to output expression
        tensor = id(NamedTensor(tensor, expr), out=out).value

        return NamedTensor(tensor, out)

    return dot


def get_at_ravelled(classical):
    def get_at(tensor, *coordinates, out):
        expr_tensor = tensor.expr
        tensor = tensor.value
        expr_coords = [c.expr for c in coordinates]
        coords = [c.value for c in coordinates]

        # (1) Example shapes here:
        # tensor: a [b c] d
        # coords (ND): a [1] d e, [1] d f
        # out: a d e f

        coords = _ravel(classical, expr_tensor, coords, expr_coords, out)

        # (2) Example shapes here:
        # tensor: a [b c] d
        # coords (1D-ravelled): a d e f
        # out: a d e f

        # Flatten tensor and retrieve values
        tensor = classical.reshape(tensor, (expr_tensor.value,))
        tensor = classical.get_at(tensor, coords, axis=0)

        # (3) Example shapes here:
        # tensor: a d e f
        # out: a d e f

        return NamedTensor(tensor, out)

    return get_at


def update_at_ravelled(op, classical):
    @use_name_of(op)
    def inner(*tensors, out):
        expr_tensor = tensors[0].expr
        tensor = tensors[0].value
        expr_updates = tensors[-1].expr
        updates = tensors[-1].value
        expr_coords = [c.expr for c in tensors[1:-1]]
        coords = [c.value for c in tensors[1:-1]]

        expr_intermediate = _join_exprs([  # TODO: what order of axes should be chosen here?
            stage3.remove(expr, lambda expr: isinstance(expr, stage3.Brackets), keep_children=False) for expr in expr_coords + [expr_updates, expr_tensor]
        ])

        # (1) Example shapes here:
        # tensor: a [b c] d
        # coords (ND): a [1] d e, [1] d f
        # updates: a d f
        # expr_intermediate: a d e f

        coords = _ravel(classical, expr_tensor, coords, expr_coords, expr_intermediate)
        expr_updates, updates = _squeeze_transpose_broadcast(classical, expr_updates, updates, expr_intermediate, broadcast_to_unitary=True)

        # (2) Example shapes here:
        # tensor: a [b c] d
        # coords (1D-ravelled): a d e f
        # updates: a d 1 f

        # Update values in flattened tensor
        tensor = classical.reshape(tensor, (expr_tensor.value,))
        tensor = op(tensor, coords, updates)
        tensor = classical.reshape(tensor, expr_tensor.shape)

        # (3) Example shapes here:
        # tensor: a [b c] d

        return NamedTensor(tensor, expr_tensor)

    return inner

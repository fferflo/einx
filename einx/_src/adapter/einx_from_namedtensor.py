import einx._src.namedtensor.stage1 as stage1
import einx._src.namedtensor.stage2 as stage2
import einx._src.namedtensor.stage3 as stage3
from einx._src.namedtensor import solve
from einx._src.namedtensor import NamedTensor
from einx._src.frontend.types import Tensor
import numpy as np
import functools
import types
import einx._src.tracer as tracer
import einx._src.adapter as adapter
import uuid
from functools import partial
from einx._src.frontend.errors import SemanticError
from collections import defaultdict
from einx._src.util.functools import use_name_of


def _tensor_to_string(tensor):
    if tensor.shape is None:
        return "Tensor factory"
    else:
        return f"Tensor with shape {tensor.shape}"


def _kwarg_to_string(v):
    s = str(v)
    if len(s) < 20:
        return s
    else:
        return "..."


class Invocation:
    def __init__(self, expression, name, tensors, kwargs):
        self.expression = expression
        self.name = name
        self.tensors = tensors
        self.kwargs = kwargs

        from einx._src.namedtensor import ExpressionIndicator

        self.indicator = ExpressionIndicator(expression)

    def to_call_signature_string(self):
        message = "The operation was called with the following arguments:\n"
        for i, tensor in enumerate(self.tensors):
            message += f"  - Positional argument #{i + 1}: {_tensor_to_string(tensor)}\n"
        for k, v in self.kwargs.items():
            message += f"  - Keyword argument '{k}': {_kwarg_to_string(v)}\n"
        return message


def _to_el_expr(expr):
    if isinstance(expr, list):
        newlist = [_to_el_expr(e) for e in expr]
        newlist = [expr for expr in newlist if expr.ndim != 0]
        return newlist
    elif isinstance(expr, stage1.List):
        return stage1.List.create(_to_el_expr(expr.children), expr.begin_pos, expr.end_pos)
    elif isinstance(expr, stage1.Axis):
        return stage1.List([])
    elif isinstance(expr, stage1.FlattenedAxis):
        inner = _to_el_expr(expr.inner)
        if inner.ndim == 0:
            return stage1.List([])
        else:
            return stage1.FlattenedAxis.create(inner, expr.begin_pos, expr.end_pos)
    elif isinstance(expr, stage1.ConcatenatedAxis):
        # ConcatenatedAxis cannot be used with brackets
        assert not any(stage1.is_in_brackets(c) for c in expr.nodes())
        return stage1.List([])
    elif isinstance(expr, stage1.Brackets):
        return expr.inner.__deepcopy__()
    elif isinstance(expr, stage1.Ellipsis):
        return stage1.Ellipsis.create(_to_el_expr(expr.inner), expr.begin_pos, expr.end_pos, expr.ellipsis_id)
    elif isinstance(expr, stage1.Op):
        return stage1.Op([_to_el_expr(c) for c in expr.children], expr.begin_pos, expr.end_pos)
    elif isinstance(expr, stage1.Args):
        return stage1.Args([_to_el_expr(c) for c in expr.children], expr.begin_pos, expr.end_pos)
    else:
        raise TypeError(f"Invalid expression type {type(expr)}")


def _parse_op(description, el_op, invocation, allow_concat=False, implicit_output=None, mark_reduced_axes=False, allow_duplicate_el_axes=True, keepdims=False):
    if not isinstance(description, str):
        raise ValueError("The operation description must be a string.")
    if keepdims is None:
        keepdims = False
    op = stage1.parse_op(description)

    if not allow_concat:
        # Disallow concatenation
        if any(isinstance(expr, stage1.ConcatenatedAxis) for expr in op.nodes()):
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_concat(op),
                message="The concatenation operator (+) is not allowed in this function.\n%EXPR%",
            )

    exprs_in = op.children[0].children

    # Get signature that is expected by the elementary operation
    el_subop = _to_el_expr(op)
    if el_op is None:
        # No signature is explicitly provided -> extract the signature from the expression itself
        if len(el_subop.children) != 2:
            raise SemanticError(invocation=invocation, message="The operation expects an output expression, but no '->' was found.\n%EXPR%")
        el_op = el_subop
    elif isinstance(el_op, str):
        el_op = stage1.parse_op(el_op)
    elif callable(el_op):
        el_op = el_op(el_subop)
        el_op = stage1.parse_op(el_op)
    else:
        assert False
    assert len(el_op.children) == 2

    # Check number of input and output expressions
    if len(el_op.children[0].children) != len(el_subop.children[0].children):
        raise SemanticError(
            invocation=invocation,
            message=f"The operation expects {len(el_op.children[0].children)} input expression(s), but found {len(el_subop.children[0].children)}.\n%EXPR%",
        )
    if len(el_subop.children) > 1 and len(el_op.children[1].children) != len(el_subop.children[1].children):
        raise SemanticError(
            invocation=invocation,
            message=f"The operation expects {len(el_op.children[1].children)} output expression(s), but found {len(el_subop.children[1].children)}.\n%EXPR%",
        )

    # Determin output expression if it is not given
    if len(op.children) == 1:
        if implicit_output == "bijective":
            if len(el_op.children[0].children) == 1 and len(el_op.children[1].children) == 1 and el_op.children[0] == el_op.children[1]:
                # single input == single output
                # -> Use input expression as output expression
                exprs_out = [expr_in.__deepcopy__() for expr_in in exprs_in]
            elif len(el_op.children[0].children) == 1 and len(el_op.children[1].children) == 1 and el_op.children[1].children[0].ndim == 0:
                # Single input and single scalar output
                # -> Remove brackets from input expression for output expression
                if keepdims:

                    def _replace(expr):
                        if isinstance(expr, stage1.Brackets):
                            return stage1.FlattenedAxis.create(stage1.List.create([]))

                    exprs_out = [stage1.map(expr_in, _replace, include_children=True) for expr_in in exprs_in]
                else:
                    exprs_out = [stage1.remove(expr_in, stage1.Brackets, keep_children=False) for expr_in in exprs_in]
            elif all(c.ndim == 0 for c in list(el_op.children[0].children) + list(el_op.children[1].children)):
                # Scalar inputs and outputs
                # -> Find superset expression
                if len(exprs_in) == 1:
                    # Only one input -> use it as output
                    exprs_out = [exprs_in[0].__deepcopy__()]
                else:
                    # Use one of the input expression if it contains the axis names of
                    # all others and if this choice is unique
                    in_axis_names = [{expr.name for expr in root.nodes() if isinstance(expr, stage1.Axis) and expr.value != 1} for root in exprs_in]

                    valid_parents = set()
                    for i, parent in enumerate(in_axis_names):
                        for j, child in enumerate(in_axis_names):
                            if i != j and not child.issubset(parent):
                                break
                        else:
                            # Found valid parent
                            valid_parents.add(exprs_in[i])

                    if len(valid_parents) != 1:
                        raise SemanticError(
                            invocation=invocation,
                            message="The output expression is missing in this operation. einx allows implicitly determining the output expression"
                            " in this operation according to the rule: If one of the input expressions contains the axis names of all other input"
                            " expressions (excluding 1s) and if this choice is unique, then this expression is used as output expression. However, no unique"
                            " input expression was found.\n%EXPR%",
                        )
                    exprs_out = [valid_parents.pop().__deepcopy__()]
            elif len(el_op.children[0].children) == 1 and len(el_op.children[1].children) == 1:
                # Single input and single output
                # -> Replace single input bracket with single output bracket
                def _to_output(expr):
                    bracket_num = len([e for e in expr.nodes() if isinstance(e, stage1.Brackets)])
                    assert bracket_num > 0
                    if bracket_num == 1:

                        def _replace(expr):
                            if isinstance(expr, stage1.Brackets):
                                return stage1.Brackets.create(stage1.Axis.create("output.axis"))

                        return stage1.map(expr, _replace, include_children=False)
                    else:
                        raise SemanticError(
                            invocation=invocation,
                            pos=invocation.indicator.get_pos_for_brackets(expr),
                            message=(
                                "The output expression is missing in this operation. The output expression can only be determined implicitly, "
                                "if the input expression contains exactly one usage of brackets ([]).\n%EXPR%"
                            ),
                        )

                exprs_out = [_to_output(expr_in) for expr_in in exprs_in]
            else:
                raise SemanticError(invocation=invocation, message="The operation expects an output expression, but no '->' was found.\n%EXPR%")
        elif isinstance(implicit_output, int):
            exprs_out = [exprs_in[implicit_output].__deepcopy__()]
        elif isinstance(implicit_output, tuple | list) and all(isinstance(i, int) for i in implicit_output):
            exprs_out = [exprs_in[i].__deepcopy__() for i in implicit_output]
        elif implicit_output is None:
            raise SemanticError(invocation=invocation, message="The operation expects an output expression, but no '->' was found.\n%EXPR%")
        else:
            assert False, f"Invalid value for implicit_output: {implicit_output}"
    else:
        exprs_out = op.children[1].children
    op = stage1.Op([stage1.Args(exprs_in), stage1.Args(exprs_out)])
    el_subop = _to_el_expr(op)
    assert len(el_op.children[0].children) == len(el_subop.children[0].children)
    assert len(el_op.children[1].children) == len(el_subop.children[1].children)

    # Check bracket usage
    def _to_ordinal_str(i):
        if i == 0:
            return "1st"
        elif i == 1:
            return "2nd"
        elif i == 2:
            return "3rd"
        else:
            return f"{i + 1}th"

    def check(i, el_arg, el_subarg, arg, inoutput):
        if el_arg.ndim == 0 and el_subarg.ndim != 0:
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_brackets(arg),
                message=f"Brackets ([]) are not allowed in the {_to_ordinal_str(i)} {inoutput} expression of this operation.\n%EXPR%",
            )
        elif el_arg.ndim != 0 and el_subarg.ndim == 0:
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_exprs(arg),
                message=f"The {_to_ordinal_str(i)} {inoutput} expression of this operation requires brackets, but no brackets were found.\n%EXPR%",
            )

    for i, (el_arg, el_subarg, arg) in enumerate(zip(el_op.children[0].children, el_subop.children[0].children, op.children[0].children, strict=False)):
        check(i, el_arg, el_subarg, arg, "input")
    for i, (el_arg, el_subarg, arg) in enumerate(zip(el_op.children[1].children, el_subop.children[1].children, op.children[1].children, strict=False)):
        check(i, el_arg, el_subarg, arg, "output")

    marking_reduced_axes = mark_reduced_axes and not any(isinstance(expr, stage1.Brackets) for expr_in in exprs_in for expr in expr_in.nodes())
    if marking_reduced_axes:
        # Check that no axis name is used twice per input
        for expr_in in exprs_in:
            counts = defaultdict(lambda: 0)
            for axis in expr_in.nodes():
                if isinstance(axis, stage1.Axis):
                    counts[axis.name] += 1
            duplicates = [name for name, count in counts.items() if count > 1]
            if len(duplicates) > 0:
                raise SemanticError(
                    invocation=invocation,
                    pos=invocation.indicator.get_pos_for_axisnames([expr_in], duplicates),
                    message=(
                        "If no axes are marked with brackets in this operation, brackets are placed automatically around all axes that do not appear "
                        "in the output expression. Since this mode is used here, no axis name may appear more than once in the same input tensor (to avoid "
                        f"confusion with diagonal extraction). However, the following axis names are used more than once: {', '.join(duplicates)}.\n%EXPR%"
                    ),
                )

        assert len(exprs_out) == 1
        expr_out = exprs_out[0]

        # If no brackets appear in exprs_in, mark all axes that don't appear in expr_out.
        axes_names_out = {axis.name for axis in expr_out.nodes() if isinstance(axis, stage1.Axis)}

        def _mark(expr):
            if isinstance(expr, stage1.Axis) and expr.name not in axes_names_out:
                return True
            else:
                return False

        exprs_in = [stage1.map(expr_in, lambda expr: stage1.Brackets(expr) if _mark(expr) else None, include_children=False) for expr_in in exprs_in]

    # Check that no two vectorized axes in any (unconcatenated) output have the same name
    for expr_out in exprs_out:
        for expr_out in stage1.split_concatenated_axes(expr_out):
            axis_names = [expr.name for expr in expr_out.nodes() if isinstance(expr, stage1.Axis) and not stage1.is_in_brackets(expr)]
            if len(axis_names) != len(set(axis_names)):
                duplicates = {name for name in axis_names if axis_names.count(name) > 1}
                raise SemanticError(
                    invocation=invocation,
                    pos=invocation.indicator.get_pos_for_axisnames([expr_out], duplicates),
                    message="The output expression must not contain multiple vectorized axes with the same name (after splitting concatenated axes).\n%EXPR%",
                )

    if not allow_duplicate_el_axes:
        # Check that no two marked axes in any tensor have the same name
        for expr in exprs_in + exprs_out:
            axis_names = [expr.name for expr in expr.nodes() if isinstance(expr, stage1.Axis) and stage1.is_in_brackets(expr)]
            if len(axis_names) != len(set(axis_names)):
                duplicates = {name for name in axis_names if axis_names.count(name) > 1}

                message = "The expression must not contain multiple axes with the same name in brackets ([]).\n%EXPR%"
                raise SemanticError(invocation=invocation, pos=invocation.indicator.get_pos_for_axisnames([expr], duplicates), message=message)

    return exprs_in, exprs_out


def _semantic_checks_dot(exprs_in, exprs_out, invocation):
    expr_out = exprs_out[0]

    # Ensure at least two inputs
    if len(exprs_in) < 2:
        raise SemanticError(
            invocation=invocation, message=f"The dot operation requires at least 2 input expressions, but only {len(exprs_in)} is given.\n%EXPR%"
        )

    # Ensure that all marked axes appear in exactly two input expressions

    def is_marked_axis(expr):
        return isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)

    marked_axis_names = {expr.name for expr_in in exprs_in for expr in expr_in.nodes() if is_marked_axis(expr)}
    invalid_axis_names = []
    for axis_name in marked_axis_names:
        count = 0
        for expr_in in exprs_in:
            if axis_name in {axis.name for axis in expr_in.nodes() if isinstance(axis, stage3.Axis)}:
                count += 1
        if count != 2:
            invalid_axis_names.append(axis_name)
    if len(invalid_axis_names) > 0:
        raise SemanticError(
            invocation=invocation,
            pos=invocation.indicator.get_pos_for_axisnames(exprs_in + [expr_out], invalid_axis_names),
            message="All contracted axes must appear in exactly two input expressions.\n%EXPR%",
        )


def _semantic_checks_get_at(exprs_in, exprs_out, invocation):
    expr_out = exprs_out[0]
    if len(exprs_in) < 2:
        raise SemanticError(invocation=invocation, message=f"The operation expects at least 2 input expressions, but found {len(exprs_in)}.\n%EXPR%")
    all_exprs = list(exprs_in) + [expr_out]
    tensor_expr = exprs_in[0]
    coords_exprs = exprs_in[1:]

    # Ensure that at most one axis is marked in each coordinate expression
    for coord_expr in coords_exprs:
        marked_axisnames = [expr.name for expr in coord_expr.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)]
        if len(marked_axisnames) > 1:
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_axisnames(all_exprs, marked_axisnames),
                message="Each coordinate expression must contain at most one axis in brackets.\n%EXPR%",
            )

    # Ensure that summed number of coordinates is equal to the number of marked axes in first input expression
    n = 0
    for coord_expr in coords_exprs:
        marked_axis = [expr for expr in coord_expr.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)]
        assert len(marked_axis) <= 1
        if len(marked_axis) == 1:
            n += marked_axis[0].value
        else:
            n += 1
    marked_axes = [expr for expr in tensor_expr.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)]
    if len(marked_axes) != n:
        raise SemanticError(
            invocation=invocation,
            pos=invocation.indicator.get_pos_for_axisnames(all_exprs, [expr.name for expr in marked_axes]),
            message="The number of coordinates must match the number of axes in brackets in the first input expression.\n%EXPR%",
        )


def _semantic_checks_sort(exprs_in, exprs_out, invocation):
    assert len(exprs_in) == 1 and len(exprs_out) == 1
    for expr in exprs_in + exprs_out:
        marked_axisnames = [expr.name for expr in expr.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)]
        if len(marked_axisnames) != 1:
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_axisnames(exprs_in + exprs_out, marked_axisnames),
                message="The expression for this operation must contain exactly one axis in brackets.\n%EXPR%",
            )


def _semantic_checks_update_at(exprs_in, exprs_out, invocation):
    expr_out = exprs_out[0]
    if len(exprs_in) < 2:
        raise SemanticError(invocation=invocation, message=f"The operation expects at least 3 input expressions, but found {len(exprs_in)}.\n%EXPR%")

    all_exprs = list(exprs_in) + [expr_out]
    tensor_expr = exprs_in[0]
    coords_exprs = exprs_in[1:-1]
    update_expr = exprs_in[-1]

    # Ensure same set of axes is marked in first input and output expression
    input_axisnames = {expr.name for expr in tensor_expr.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)}
    output_axisnames = {expr.name for expr in expr_out.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)}
    if input_axisnames != output_axisnames:
        raise SemanticError(
            invocation=invocation,
            pos=invocation.indicator.get_pos_for_axisnames(all_exprs, input_axisnames.symmetric_difference(output_axisnames)),
            message="The first input and output expressions must have the same set of axes in brackets.\n%EXPR%",
        )

    # Ensure that at most one axis is marked in each coordinate expression (aside from reduced axes that are also marked in updates)
    for coord_expr in coords_exprs:
        marked_axisnames = {expr.name for expr in coord_expr.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)}
        if len(marked_axisnames) > 1:
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_axisnames(all_exprs, marked_axisnames),
                message="Each coordinate expression must contain at most one coordinate axis in brackets.\n%EXPR%",
            )

    # Ensure marked axes in (1) target tensor, and (2) coordinate and update expressions are non-overlapping
    target_axisnames = {expr.name for expr in tensor_expr.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)}
    coordupdate_axisnames = {
        expr.name for expr in coords_exprs + [update_expr] for expr in expr.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)
    }
    intersection = target_axisnames.intersection(coordupdate_axisnames)
    if len(intersection) > 0:
        raise SemanticError(
            invocation=invocation,
            pos=invocation.indicator.get_pos_for_axisnames(all_exprs, intersection),
            message="Axes may not appear in brackets both in the first input tensor and in a coordinate or update expression.\n%EXPR%",
        )

    # Ensure that summed number of coordinates is equal to the number of marked axes in first input expression
    n = 0
    for coord_expr in coords_exprs:
        marked_axis = {expr for expr in coord_expr.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)}
        assert len(marked_axis) <= 1
        if len(marked_axis) == 1:
            n += marked_axis.pop().value
        else:
            n += 1
    marked_axes = [expr for expr in tensor_expr.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)]
    if len(marked_axes) != n:
        raise SemanticError(
            invocation=invocation,
            pos=invocation.indicator.get_pos_for_axisnames(all_exprs, [expr.name for expr in marked_axes]),
            message="The number of coordinates must match the number of axes in brackets in the first input expression.\n%EXPR%",
        )


def _semantic_checks_argfind(exprs_in, exprs_out, invocation):
    if len(exprs_in) != 1:
        raise SemanticError(invocation=invocation, message=f"The operation expects exactly one input expression, but found {len(exprs_in)}.\n%EXPR%")
    if len(exprs_out) != 1:
        raise SemanticError(invocation=invocation, message=f"The operation expects exactly one output expression, but found {len(exprs_in)}.\n%EXPR%")
    expr_in = exprs_in[0]
    expr_out = exprs_out[0]
    all_exprs = [expr_in, expr_out]

    # Ensure that at most one axis is marked in output expression
    marked_output_axes = [expr for expr in expr_out.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)]
    if len(marked_output_axes) > 1:
        raise SemanticError(
            invocation=invocation,
            pos=invocation.indicator.get_pos_for_axisnames(all_exprs, [a.name for a in marked_output_axes]),
            message="The output expression must contain at most one axis in brackets.\n%EXPR%",
        )
    marked_output_axis = marked_output_axes[0] if len(marked_output_axes) == 1 else None

    # Ensure that number of marked axes in input expression is equal to value of marked axis in output expression
    marked_input_axes = [expr for expr in expr_in.nodes() if isinstance(expr, stage3.Axis) and stage3.is_in_brackets(expr)]
    if marked_output_axis is None:
        if len(marked_input_axes) != 1:
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_axisnames(all_exprs, [expr.name for expr in marked_input_axes]),
                message=(
                    f"If no axis is marked in the output expression, exactly one axis must be marked in the input expression, "
                    f"but found {len(marked_input_axes)} marked axes.\n%EXPR%"
                ),
            )
    else:
        if len(marked_input_axes) != marked_output_axis.value:
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_axisnames(all_exprs, [a.name for a in marked_input_axes] + [marked_output_axis.name]),
                message=(
                    f"The number of axes in brackets in the input expression ({len(marked_input_axes)}) must match the value "
                    "of the marked axis in the output expression ({marked_output_axis.value}).\n%EXPR%"
                ),
            )


def _cast_shape(tensor, expr):
    if tensor.shape is None:
        assert isinstance(tensor, tracer.signature.classical.ConvertibleTensor)
        return tracer.cast(tensor, lambda origin: tracer.signature.classical.ConvertibleTensor(origin, tensor.concrete, expr.shape))
    else:
        assert tuple(tensor.shape) == tuple(expr.shape)
        return tensor


def op(
    op,
    el_op=None,
    allow_concat=False,
    implicit_output=None,
    mark_reduced_axes=False,
    add_keepdims_param=False,
    check=None,
    cse_in_brackets=True,
    iskwarg=lambda name: False,
    equations_stage3=None,
    allow_nontrivial_unmarked_reduced_axes=False,
    allow_duplicate_el_axes=True,
    no_el_axis_permute=False,
):
    if isinstance(iskwarg, list | tuple | set):
        iskwarg = lambda name, iskwarg=iskwarg: name in iskwarg

    def inner(description, *tensors: Tensor, **kwargs):
        invocation = Invocation(
            description, name=op.__name__ if hasattr(op, "__name__") else "operation", tensors=tensors, kwargs=kwargs
        )  # Used for error reporting

        if add_keepdims_param:
            keepdims = {"keepdims": kwargs["keepdims"] if "keepdims" in kwargs else False}
            kwargs.pop("keepdims", None)
        else:
            keepdims = {}

        exprs_in, exprs_out = _parse_op(
            description,
            el_op,
            invocation=invocation,
            allow_concat=allow_concat,
            implicit_output=implicit_output,
            mark_reduced_axes=mark_reduced_axes,
            allow_duplicate_el_axes=allow_duplicate_el_axes,
            **keepdims,
        )

        if len(exprs_in) != len(tensors):

            def choose_s(i):
                return "s" if i != 1 else ""

            raise ValueError(
                f"The operation is defined with {len(exprs_in)} input expression{choose_s(len(exprs_in))}, but {len(tensors)} "
                f"input tensor{choose_s(len(tensors))} {'are' if len(tensors) != 1 else 'is'} given as argument{choose_s(len(tensors))}."
            )

        used_axis_names = {expr.name for expr in exprs_in + exprs_out for expr in expr.nodes() if isinstance(expr, stage1.Axis)}
        if any(iskwarg(name) for name in used_axis_names):
            invalid_kwargnames = [name for name in used_axis_names if iskwarg(name)]
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_axisnames(exprs_in + exprs_out, invalid_kwargnames),
                message=(
                    f"The following axis names may not be used in the expression, since they are keyword arguments of the elementary "
                    f"operation: {', '.join(invalid_kwargnames)}.\n%EXPR%"
                ),
            )

        # Split kwargs from axis parameters
        parameters = {}
        new_kwargs = {}
        for key, value in kwargs.items():
            if iskwarg(key):
                new_kwargs[key] = value
            else:
                parameters[key] = value
        kwargs = new_kwargs
        del new_kwargs

        if no_el_axis_permute:
            for expr_in, expr_out in zip(exprs_in, exprs_out, strict=False):
                axisnames_in = []
                for axis in expr_in.nodes():
                    if isinstance(axis, stage1.Axis) and stage1.is_in_brackets(axis):
                        axisnames_in.append(axis.name)
                axisnames_out = []
                for axis in expr_out.nodes():
                    if isinstance(axis, stage1.Axis) and stage1.is_in_brackets(axis):
                        axisnames_out.append(axis.name)
                if axisnames_in != axisnames_out:
                    raise SemanticError(
                        invocation=invocation,
                        pos=invocation.indicator.get_pos_for_axisnames(exprs_in + exprs_out, set(axisnames_in).union(set(axisnames_out))),
                        message="All axes marked with brackets must appear in the same order in the inputs and outputs of this operation.\n%EXPR%",
                    )

        exprs_in, exprs_out = solve(
            exprs_in,
            exprs_out,
            [tensor.shape for tensor in tensors],
            invocation,
            parameters,
            cse_in_brackets=cse_in_brackets,
            equations_stage3=partial(equations_stage3, invocation=invocation) if equations_stage3 is not None else None,
        )

        if not allow_nontrivial_unmarked_reduced_axes:
            axis_names_in = {
                axis.name
                for expr_in in exprs_in
                for axis in expr_in.nodes()
                if isinstance(axis, stage3.Axis) and not stage3.is_in_brackets(axis) and axis.value != 1
            }
            axis_names_out = {
                axis.name
                for expr_out in exprs_out
                for axis in expr_out.nodes()
                if isinstance(axis, stage3.Axis) and not stage3.is_in_brackets(axis) and axis.value != 1
            }
            axis_names_reduced = axis_names_in - axis_names_out
            if len(axis_names_reduced) > 0:
                raise SemanticError(
                    invocation=invocation,
                    pos=invocation.indicator.get_pos_for_axisnames(exprs_in + exprs_out, axis_names_reduced),
                    message=f"The input axes {axis_names_reduced} must appear in the output expression.\n%EXPR%",
                )

        if check is not None:
            check(exprs_in, exprs_out, invocation)

        tensors = [_cast_shape(tensor, expr_in) for tensor, expr_in in zip(tensors, exprs_in, strict=False)]

        tensors = [NamedTensor(tensor, expr_in) for tensor, expr_in in zip(tensors, exprs_in, strict=False)]
        try:
            tensors = op(*tensors, out=exprs_out[0] if len(exprs_out) == 1 else exprs_out, **kwargs)
        except SemanticError as e:
            raise SemanticError(invocation=invocation, pos=e.pos, message=str(e)) from e

        if len(exprs_out) > 1:
            return tuple(t.value for t in tensors)
        else:
            return tensors.value

        return tensor

    inner.__name__ = op.__name__
    inner.__qualname__ = op.__qualname__
    inner.__doc__ = op.__doc__
    inner.__module__ = op.__module__
    return inner


def id(op, **kwargs):
    if "allow_concat" not in kwargs:
        kwargs["allow_concat"] = True

    def el_op(op):
        n_in = len(op.children[0].children)
        n_out = n_in if len(op.children) == 1 else len(op.children[1].children)
        args_in = ", ".join("" for i in range(n_in))
        args_out = ", ".join("" for i in range(n_out))
        return f"{args_in} -> {args_out}"

    return globals()["op"](op, el_op=el_op, implicit_output="bijective", **kwargs)


def elementwise(op, **kwargs):
    def el_op(op):
        n_in = len(op.children[0].children)
        args_in = ", ".join("" for i in range(n_in))
        return f"{args_in} ->"

    return globals()["op"](op, el_op=el_op, implicit_output="bijective", **kwargs)


def dot(op, **kwargs):
    def el_op(op):
        return f"{op.children[0]} ->"

    return globals()["op"](
        op, el_op=el_op, implicit_output="bijective", mark_reduced_axes=True, allow_duplicate_el_axes=False, check=_semantic_checks_dot, **kwargs
    )


def _equations_stage3_index_at(exprs_in, exprs_out, invocation, is_update):
    if len(exprs_in) <= 1:
        return []

    tensor_expr = exprs_in[0]
    coord_exprs = exprs_in[1:-1] if is_update else exprs_in[1:]
    coords_axes = []
    for coord_expr in coord_exprs:
        marked_axes = [expr for expr in coord_expr.nodes() if isinstance(expr, stage2.Axis) and stage2.is_in_brackets(expr)]
        if len(marked_axes) == 0:
            coords_axes.append(stage2.Axis(f"unnamed.{uuid.uuid4().int}", 1, ellipsis_indices=[]))
        elif len(marked_axes) == 1:
            coords_axes.append(marked_axes[0])
        else:
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_axisnames(exprs_in + exprs_out, [expr.name for expr in marked_axes]),
                message="Each coordinate expression must contain at most one axis in brackets.\n%EXPR%",
            )
    marked_coord_axis = stage2.ConcatenatedAxis.create(coords_axes, ellipsis_indices=[])
    marked_axes_in = [expr for expr in tensor_expr.nodes() if isinstance(expr, stage2.Axis) and stage2.is_in_brackets(expr)]

    if marked_coord_axis.value is not None and marked_coord_axis.value != len(marked_axes_in):
        raise SemanticError(
            invocation=invocation,
            pos=invocation.indicator.get_pos_for_axisnames(exprs_in + exprs_out, [marked_coord_axis.name] + [expr.name for expr in marked_axes_in]),
            message=(
                f"The sum of the lengths of marked coordinate axes ({marked_coord_axis.value}) must match the number of marked axes "
                f"in the first input expression ({len(marked_axes_in)}).\n%EXPR%"
            ),
        )

    return [stage3.Equation(marked_coord_axis, stage2.Axis(f"unnamed.{uuid.uuid4().int}", len(marked_axes_in), ellipsis_indices=[]))]


def get_at(op, **kwargs):
    def el_op(op):
        return ", ".join(str(c) for c in op.children[0].children) + " ->"

    return globals()["op"](
        op,
        el_op=el_op,
        implicit_output="bijective",
        check=_semantic_checks_get_at,
        equations_stage3=partial(_equations_stage3_index_at, is_update=False),
        **kwargs,
    )


def update_at(op, **kwargs):
    def el_op(op):
        return ", ".join(str(c) for c in op.children[0].children[:-1]) + f", -> {op.children[0].children[0]}"

    op = globals()["op"](
        op,
        el_op=el_op,
        implicit_output=0,
        check=_semantic_checks_update_at,
        equations_stage3=partial(_equations_stage3_index_at, is_update=True),
        allow_nontrivial_unmarked_reduced_axes=True,
        no_el_axis_permute=True,
        **kwargs,
    )

    @use_name_of(op)
    def op_with_zerosized_args(description, *tensors: Tensor, **kwargs):
        if len(tensors) > 0 and any(int(i) == 0 for tensor in [t for t in tensors[1:] if t.shape is not None] for i in tensor.shape):
            return tensors[0]  # TODO: remove this when dynamic dimensions are supported?
        else:
            return op(description, *tensors, **kwargs)

    return op_with_zerosized_args


def reduce(op, **kwargs):
    def el_op(op):
        return f"{op.children[0].children[0]} ->"

    return globals()["op"](op, el_op=el_op, implicit_output="bijective", mark_reduced_axes=True, add_keepdims_param=True, **kwargs)


def _equations_stage3_argfind(exprs_in, exprs_out, invocation):
    marked_axes_in = [expr for expr in exprs_in[0].nodes() if isinstance(expr, stage2.Axis) and stage2.is_in_brackets(expr)]
    marked_axes_out = [expr for expr in exprs_out[0].nodes() if isinstance(expr, stage2.Axis) and stage2.is_in_brackets(expr)]
    if len(marked_axes_out) == 0:
        if len(marked_axes_in) != 1:
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_axisnames(exprs_in + exprs_out, [a.name for a in marked_axes_in]),
                message=(
                    f"If no axis is marked with brackets in the output expression, exactly one axis must be marked in the "
                    f"input expression, but found {len(marked_axes_in)} marked axes.\n%EXPR%"
                ),
            )
        return []
    elif len(marked_axes_out) == 1:
        marked_axis_out = marked_axes_out[0]
        if marked_axis_out.value is not None and marked_axis_out.value != len(marked_axes_in):
            raise SemanticError(
                invocation=invocation,
                pos=invocation.indicator.get_pos_for_axisnames(exprs_in + exprs_out, [marked_axis_out.name] + [a.name for a in marked_axes_in]),
                message=(
                    f"The value of the marked axis in the output expression ({marked_axis_out.value}) must match the number of marked "
                    f"axes in the input expression ({len(marked_axes_in)}).\n%EXPR%"
                ),
            )
        return [stage3.Equation(marked_axis_out, stage2.Axis(f"unnamed.{uuid.uuid4().int}", len(marked_axes_in), ellipsis_indices=[]))]
    else:
        raise SemanticError(
            invocation=invocation,
            pos=invocation.indicator.get_pos_for_axisnames(exprs_in + exprs_out, [a.name for a in marked_axes_out]),
            message="The output expression must contain at most one axis in brackets.\n%EXPR%",
        )


def argfind(op, **kwargs):
    def el_op(op):
        if len(op.children) == 1 or op.children[1].children[0].ndim != 0:
            return f"{op.children[0].children[0]} -> a{uuid.uuid4().int}"
        else:
            return f"{op.children[0].children[0]} ->"

    return globals()["op"](op, el_op=el_op, implicit_output="bijective", check=_semantic_checks_argfind, equations_stage3=_equations_stage3_argfind, **kwargs)


def preserve_shape(op, **kwargs):
    def el_op(op):
        return f"{op.children[0].children[0]} -> {op.children[0].children[0]}"

    return globals()["op"](op, el_op=el_op, implicit_output="bijective", no_el_axis_permute=True, **kwargs)


_name_to_op = (
    {name: elementwise for name in adapter.ops.elementwise}
    | {name: partial(reduce, cse_in_brackets=True) for name in adapter.ops.reduce}
    | {name: update_at for name in adapter.ops.update_at}
    | {name: argfind for name in adapter.ops.argfind}
    | {
        "get_at": get_at,
        "dot": dot,
        "id": id,
        "roll": partial(preserve_shape, iskwarg=["shift"]),
        "flip": preserve_shape,
        "sort": partial(preserve_shape, check=_semantic_checks_sort),
        "argsort": partial(preserve_shape, check=_semantic_checks_sort),
        "softmax": preserve_shape,
        "log_softmax": preserve_shape,
    }
)


def ops(namedtensor_ops, **kwargs):
    return {name: _name_to_op[name](namedtensor_ops[name], **kwargs) for name in namedtensor_ops.keys()}

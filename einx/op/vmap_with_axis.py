import einx
from . import util
import numpy as np
from functools import partial
from typing import Callable, Mapping, Union, Tuple
import numpy.typing as npt

_op_names = ["roll", "flip"]

@einx.lru_cache(trace=lambda t, c: lambda exprs_in, tensors_in, exprs_out, op, kwargs={}, backend=None: c(exprs_in, [t(x) for x in tensors_in], exprs_out, op, kwargs))
def vmap_with_axis_stage3(exprs_in, tensors_in, exprs_out, op, kwargs={}, backend=None):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    elif isinstance(backend, str):
        backend = einx.backend.get(backend)
    op = backend.op(op, tracable=False)
    if len(exprs_in) != len(tensors_in):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_in)}")
    if len(set(exprs_out)) != 1:
        raise ValueError("All output expressions must be the same")
    for root in list(exprs_in) + list(exprs_out):
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    kwargs = {**kwargs}

    # Call tensor factories
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend) for tensor, expr in zip(tensors_in, exprs_in)]

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend)
    exprs_out_flat = util.flatten(exprs_out)
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_out_flat)

    transpose_first = len(exprs_in) > 1 # TODO: and inputs dont have matching expressions
    if not transpose_first and len(exprs_in) > 1:
        raise ValueError("When multiple input expressions are given, they have to be transposed to the same layout before applying the operation (transpose_first has to be set to True)")

    # Ensure that axis markings are consistent
    def is_vmapped(expr):
        return not einx.expr.stage3.is_marked(expr)
    vmapped_axis_names = set(v.name for root in list(exprs_in) + list(exprs_out_flat) for v in root if is_vmapped(v))
    for root in list(exprs_in) + list(exprs_out_flat):
        for v in root:
            if (v.name in vmapped_axis_names) != is_vmapped(v):
                raise ValueError(f"Axis {v.name} appears both as vmapped and non-vmapped")

    marked_input_axes = set(axis.name for expr_in in exprs_in for axis in expr_in.all() if isinstance(axis, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(axis))
    marked_output_axes = set(axis.name for expr_out in exprs_out_flat for axis in expr_out.all() if isinstance(axis, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(axis))

    if transpose_first:
        # Transpose and insert trivial axes
        if marked_input_axes != marked_output_axes:
            raise ValueError("transpose_first can only be used when axes are unchanged")
        x = [util.transpose_broadcast(expr_in, tensor_in, exprs_out_flat[0], broadcast=False) for expr_in, tensor_in in zip(exprs_in, tensors_in)]
        tensors_in = [x[0] for x in x]
        exprs_in = [x[1] for x in x]
        assert len(set(len(expr) for expr in exprs_in)) == 1
        marked_input_axes = set(axis.name for expr_in in exprs_in for axis in expr_in.all() if isinstance(axis, einx.expr.stage3.Axis) and einx.expr.stage3.is_marked(axis))

    # Add axis argument
    if transpose_first:
        axis_indices = tuple(i for i, axis in enumerate(exprs_out_flat[0]) if axis.name in marked_input_axes)
    else:
        axes_in = [list(expr) for expr in exprs_in]
        axis_indices = tuple(i for i in range(len(axes_in[0])) if any(axes_in[i].name in marked_input_axes for axes_in in axes_in))
    if len(axis_indices) > 0:
        kwargs["axis"] = axis_indices if len(axis_indices) > 1 else axis_indices[0]

    # Apply operation
    in_axis_names = set(axis.name for expr in exprs_in for axis in expr)
    output_shape = np.asarray([(axis.value if axis.name in in_axis_names else 1) for axis in exprs_out_flat[0]])
    output_shapes = (output_shape,) * len(exprs_out_flat) if len(exprs_out_flat) > 1 else output_shape
    tensors_out = backend.apply(op, args=tensors_in, kwargs=kwargs, output_shapes=output_shapes)
    if not isinstance(tensors_out, (tuple, list)):
        tensors_out = (tensors_out,)
    if len(tensors_out) != len(exprs_out_flat):
        raise ValueError(f"Expected {len(exprs_out_flat)} output tensor(s), got {len(tensors_out)}")

    if not transpose_first:
        # Transpose and broadcast missing output dimensions
        def replace(expr):
            if isinstance(expr, einx.expr.stage3.Marker):
                if len(marked_output_axes) == 0:
                    return []
                else:
                    return expr.__deepcopy__()
        exprs_in = [einx.expr.stage3.replace(expr_in, replace) for expr_in in exprs_in]
        tensors_out = [util.transpose_broadcast(exprs_in[0], tensor_out, expr_out)[0] for tensor_out, expr_out in zip(tensors_out, exprs_out_flat)]
    else:
        # Already transposed, only broadcast to flat output shape
        def broadcast(tensor, expr):
            if tensor.shape != expr.shape:
                tensor = backend.broadcast_to(tensor, expr.shape)
            return tensor
        tensors_out = [broadcast(tensor, expr) for tensor, expr in zip(tensors_out, exprs_out_flat)]

    # Unflatten output expressions
    tensors_out = util.unflatten(exprs_out_flat, tensors_out, exprs_out, backend)

    return tensors_out, exprs_out

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
        # Description: Single expression
        expr = description
        if "," in expr:
            raise ValueError("Only a single expression is allowed when using the choice operator [|]")
        if len(tensor_shapes) != 1:
            raise ValueError(f"Expected 1 input tensor, got {len(tensor_shapes)}")

        expr = einx.expr.stage1.parse(expr)
        if any(isinstance(expr, einx.expr.stage1.Choice) for expr in expr.all()):
            # "input -> output" using [|]-choice
            expr_in = str(einx.expr.stage1.choose(expr, 0, num=2))
            expr_out = str(einx.expr.stage1.choose(expr, 1, num=2))
        else:
            # Same input and output
            expr_in = expr_out = str(expr)

        exprs_in = [expr_in]
        exprs_out = [expr_out]

    exprs = einx.expr.solve(
          [einx.expr.Equation(expr_in, tensor_shape) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
        + [einx.expr.Equation(expr_out) for expr_out in exprs_out] \
        + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
        cse=cse,
        cse_concat=False,
    )[:len(exprs_in) + len(exprs_out)]
    exprs_in, exprs_out = exprs[:len(exprs_in)], exprs[len(exprs_in):]

    return exprs_in, exprs_out

@einx.lru_cache(trace=lambda t, c: lambda description, *tensors, backend=None, **kwargs: c(description, *[t(x) for x in tensors], **kwargs))
def vmap_with_axis(description: str, *tensors: einx.Tensor, op: Callable, backend: Union[einx.Backend, str, None] = None, cse: bool = True, kwargs: Mapping = {}, **parameters: npt.ArrayLike):
    """Applies a function to the marked axes of the input tensors by passing the ``axis`` argument.

    The function flattens all input tensors, applies the given operation and rearranges
    the result to match the output expressions (see :doc:`How does einx handle input and output tensors? </faq/flatten>`).

    The `description` argument specifies the input and output expressions, as well as axes along which the operation is applied. It must meet one of the following formats:

    1. ``input1, input2, ... -> output1, output2, ...``
        All input and output expressions are specified explicitly. Axes that the operation is applied along are marked with ``[]``-brackets.

    2. ``... [input|output] ...``
        The left and right choices correspond to the input and output tensors, respectively. Axes that the operation is applied along are marked with ``[]``-brackets.

        Example: ``a [b1|b2]`` resolves to ``a [b1] -> a [b2]``. ``a [b]`` resolves to ``a [b] -> a [b]``.

    When the function is applied on scalars, the ``axis`` argument is not passed. For multiple input tensors, the function must follow
    `Numpy broadcasting rules <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.

    Args:
        description: Description string in Einstein notation (see above).
        tensors: Input tensors or tensor factories matching the description string.
        op: Backend operation. Is called with ``op(tensor, axis=...)``. If `op` is a string, retrieves the attribute of `backend` with the same name. 
        kwargs: Additional keyword arguments that are passed to ``op``.
        backend: Backend to use for all operations. If None, determines the backend from the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults to True.
        graph: Whether to return the graph representation of the operation instead of computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the operation if ``graph=False``, otherwise the graph representation of the operation.

    Examples:
        Reverse order of elements along an axis:

        >>> x = np.random.uniform(size=(16, 20))
        >>> einx.vmap_with_axis("a [b] -> a [b]", x, op=np.flip).shape
        (16, 20)

        Roll elements along two axes:

        >>> x = np.random.uniform(size=(16, 20))
        >>> einx.vmap_with_axis("a ([b c]) -> a ([b c])", x, op=partial(np.roll, shift=(2, 2)), b=2).shape
        (16, 20)

        Compute sum along axis:

        >>> x = np.random.uniform(size=(16, 20))
        >>> einx.vmap_with_axis("a ([b] c) -> c a", x, op=np.sum, b=2).shape
        (16, 20)
    """
    exprs_in, exprs_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensors, exprs_out = vmap_with_axis_stage3(exprs_in, tensors, exprs_out, op=op, kwargs=kwargs, backend=backend)
    return tensors[0] if len(exprs_out) == 1 else tensors
vmap_with_axis.parse = parse

def flip(description: str, tensor: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike):
    """Specialization of :func:`einx.vmap_with_axis` with ``op="flip"``."""
    return vmap_with_axis(description, tensor, op="flip", backend=backend, cse=cse, **parameters)

def roll(description: str, tensor: einx.Tensor, shift: Union[int, Tuple[int]], backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike):
    """Specialization of :func:`einx.vmap_with_axis` with ``op="roll"`` and ``kwargs={"shift": shift}``."""
    return vmap_with_axis(description, tensor, op="roll", backend=backend, kwargs={"shift": shift}, cse=cse, **parameters)

def softmax(description: str, tensor: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike):
    """Specialization of :func:`einx.vmap_with_axis` with ``op="softmax"``"""
    return vmap_with_axis(description, tensor, op="softmax", backend=backend, cse=cse, **parameters)

def log_softmax(description: str, tensor: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike):
    """Specialization of :func:`einx.vmap_with_axis` with ``op="log_softmax"``"""
    return vmap_with_axis(description, tensor, op="log_softmax", backend=backend, cse=cse, **parameters)
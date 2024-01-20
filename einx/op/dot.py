import einx
from . import util
import numpy as np
from typing import Union
import numpy.typing as npt

@einx.lru_cache(trace=lambda t, c: lambda exprs_in, tensors_in, expr_out, backend=None: c(exprs_in, [t(x) for x in tensors_in], expr_out))
def dot_stage3(exprs_in, tensors_in, expr_out, backend=None):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    elif isinstance(backend, str):
        backend = einx.backend.get(backend)
    if any(isinstance(expr, einx.expr.stage3.Concatenation) for expr in expr_out.all()):
        raise ValueError("Output expression cannot contain concatenations")
    for root in list(exprs_in) + [expr_out]:
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Marker):
                raise ValueError(f"Marker is not allowed")
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")

    # Call tensor factories
    output_axis_names = {a.name for a in expr_out.all() if isinstance(a, einx.expr.stage3.Axis)}
    def get_fans(idx):
        other_input_axis_names = {a.name for i, expr_in in enumerate(exprs_in) for a in expr_in.all() if i != idx and isinstance(a, einx.expr.stage3.Axis)}
        in_axis = []
        out_axis = []
        batch_axis = []
        for i, child in enumerate(exprs_in[idx]):
            any_in_other_input = any(isinstance(a, einx.expr.stage3.Axis) and a.name in other_input_axis_names for a in child.all())
            any_in_output = any(isinstance(a, einx.expr.stage3.Axis) and a.name in output_axis_names for a in child.all())
            if any_in_other_input and not any_in_output:
                in_axis.append(i)
            elif any_in_output and not any_in_other_input:
                out_axis.append(i)
            else:
                batch_axis.append(i)
        return {"in_axis": tuple(in_axis), "out_axis": tuple(out_axis), "batch_axis": tuple(batch_axis)}
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend, **get_fans(i), name="weight", init="dot") for i, (tensor, expr) in enumerate(zip(tensors_in, exprs_in))]

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend)
    expr_out_flat = util.flatten([expr_out])[0]
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert einx.expr.stage3.is_flat(expr_out_flat)

    # Apply einsum
    einsum_variables = {}
    def get_einsum_variable(key):
        if key in einsum_variables:
            return einsum_variables[key]
        else:
            v = chr(ord("a") + len(einsum_variables))
            if ord(v) > ord("z"):
                raise ValueError(f"Only supports up to {ord('z') - ord('a') + 1} unique input axes")
            einsum_variables[key] = v
            return v
    def to_einsum(axes):
        return "".join(get_einsum_variable(a.name) for a in axes)

    input_axis_names = set(a.name for expr in exprs_in for a in einx.expr.stage3.get_axes(expr))

    einsum_str = ",".join(to_einsum(einx.expr.stage3.get_axes(expr)) for expr in exprs_in) \
               + "->" + to_einsum([a for a in einx.expr.stage3.get_axes(expr_out_flat) if a.name in input_axis_names])

    tensor = backend.einsum(einsum_str, *tensors_in)
    expr = einx.expr.stage3.List([a.__deepcopy__() for a in einx.expr.stage3.get_axes(expr_out_flat) if a.name in input_axis_names])

    # Transpose and broadcast missing output dimensions
    tensor = util.transpose_broadcast(expr, tensor, expr_out_flat)[0]

    # Unflatten output expression
    assert not tensor.shape is None
    if tensor.shape != expr_out.shape:
        tensor = backend.reshape(tensor, expr_out.shape)

    return tensor, expr_out

@einx.lru_cache
def parse(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    description = description.split("->")
    if len(description) == 1:
        # Description: "input -> output" using [|]-choice
        expr = description[0]
        if "," in expr:
            raise ValueError("Only a single input expression is allowed when output expression is not given")
        if len(tensor_shapes) != 2:
            raise ValueError(f"Expected 2 input tensors, got {len(tensor_shapes)}")

        expr = einx.expr.stage1.parse(expr)
        expr_in1 = str(einx.expr.stage1.choose(expr, 0, num=2))
        expr_out = str(einx.expr.stage1.choose(expr, 1, num=2))

        exprs_in = [expr_in1]
    else:
        # Description: "inputs... -> output"
        if len(description) > 2:
            raise ValueError("Operation can contain at most one '->'")
        exprs_in, expr_out = description
        exprs_in = exprs_in.split(",")

    if len(exprs_in) == 1 and len(tensor_shapes) == 2:
        # Description: input1 -> output, determine input2 implicitly
        expr_in1 = einx.expr.stage1.parse(exprs_in[0])
        expr_out = einx.expr.stage1.parse(expr_out)

        for root in [expr_in1, expr_out]:
            for expr in root.all():
                if isinstance(expr, einx.expr.stage1.UnnamedAxis) and expr.value != 1 and einx.expr.stage1.is_marked(expr):
                    raise ValueError(f"Cannot mark unnamed non-trivial axes, but found {expr}")

        # Get ordered list of axes for second input
        names = []
        for root in [expr_in1, expr_out]:
            for expr in root.all():
                if isinstance(expr, einx.expr.stage1.NamedAxis) and einx.expr.stage1.is_marked(expr):
                    name = expr.name
                    for _ in range(expr.depth):
                        name = name + einx.expr.stage1._ellipsis
                    if not name in names:
                        names.append(name)
        expr_in2 = " ".join(names)

        expr_in1 = einx.expr.stage1.demark(expr_in1)
        expr_out = einx.expr.stage1.demark(expr_out)
        exprs_in = [expr_in1, expr_in2]

    if len(exprs_in) != len(tensor_shapes):
        raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensor_shapes)}")

    exprs = einx.expr.solve(
            [einx.expr.Equation(expr_in, tensor_shape) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
          + [einx.expr.Equation(expr_out)] \
          + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
        cse=cse,
        cse_concat=False,
    )[:len(exprs_in) + 1]
    exprs_in, expr_out = exprs[:-1], exprs[-1]

    return exprs_in, expr_out

@einx.lru_cache(trace=lambda t, c: lambda description, *tensors, backend=None, **kwargs: c(description, *[t(x) for x in tensors], **kwargs))
def dot(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Computes a general dot-product of the input tensors.

    The function flattens all input tensors, applies the general dot-product yielding a single output tensor, and rearranges
    the result to match the output expression (see :doc:`How does einx handle input and output tensors? </faq/flatten>`).

    The `description` argument specifies the input and output expressions. It must meet one of the following formats:

    1. ``input1, input2, ... -> output``
        All input and output expressions are specified explicitly. Similar to `np.einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_ notation.
        
    2. ``input1 -> output``
        The function accepts two input tensors. ``[]``-brackets mark all axes in ``input1`` and ``output`` that should also appear in the second input.
        The second input is then determined as an ordered list of all marked axes (without duplicates).
        
        Example: ``[b c1] -> [b c2]`` resolves to ``b c1, b c1 c2 -> b c2``

    3. ``... [input1|output] ...``
        The function accepts two input tensors. The left and right choices correspond to the first input tensor and the output tensor, respectively.

        Example: ``b [c1|c2]`` resolves to ``b [c1] -> b [c2]``

    The function additionally passes the ``in_axes``, ``out_axes`` and ``batch_axes`` arguments to tensor factories that can be used to determine the fan-in
    and fan-out of a neural network layer and initialize weights accordingly
    (see e.g. `jax.nn.initializers.lecun_normal <https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.lecun_normal.html#jax.nn.initializers.lecun_normal>`_)

    Args:
        description: Description string in Einstein notation (see above).
        tensors: Input tensors or tensor factories matching the description string.
        backend: Backend to use for all operations. If None, determines the backend from the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults to True.
        graph: Whether to return the graph representation of the operation instead of computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the dot-product operation if `graph=False`, otherwise the graph representation of the operation.

    Examples:
        Compute an inner product between two vectors:

        >>> a, b = np.random.uniform(size=(10,)), np.random.uniform(size=(10,))
        >>> einx.dot("a, a ->", a, b).shape
        ()

        Compute a matrix-vector product:

        >>> a, b = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))
        >>> einx.dot("a b, b -> a", a, b).shape
        (10,)
        >>> einx.dot("a [b] -> a", a, b).shape
        (10,)
        >>> einx.dot("a [b|]", a, b).shape
        (10,)

        Compute a vector-matrix product:

        >>> a, b = np.random.uniform(size=(10,)), np.random.uniform(size=(10, 10))
        >>> einx.dot("a, a b -> b", a, b).shape
        (10,)
        >>> einx.dot("[a] -> [b]", a, b).shape
        (10,)
        >>> einx.dot("[a|b]", a, b).shape
        (10,)

        Multiply a tensor with a weight matrix:

        >>> x, w = np.random.uniform(size=(4, 16, 16, 64)), np.random.uniform(size=(64, 32,))
        >>> einx.dot("b... [c1|c2]", x, w).shape
        (4, 16, 16, 32)
    """
    exprs_in, expr_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensor, expr = dot_stage3(exprs_in, tensors, expr_out, backend=backend)
    return tensor
dot.parse = parse
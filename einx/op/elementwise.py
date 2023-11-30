import einx
from . import util
from functools import partial
import numpy as np

_op_names = ["add", "subtract", "multiply", "true_divide", "floor_divide", "divide", "logical_and", "logical_or", "where", "less", "less_equal", "greater", "greater_equal", "equal", "not_equal", "maximum", "minimum"]

@einx.lru_cache(trace=lambda k: k[0] in [1, "tensors_in"])
def elementwise_stage3(exprs_in, tensors_in, expr_out, backend=None, op=None):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    if op is None:
        raise TypeError("op cannot be None")
    if isinstance(op, str):
        op = getattr(backend, op)
    else:
        op = partial(backend.elementwise, op=op)

    # Implicitly determine output expression
    if expr_out is None:
        # Check if one input expression is parent of all others
        children_str = [str(einx.expr.stage3.remove_unnamed_trivial_axes(expr)) for expr in exprs_in]
        for i, parent in enumerate(exprs_in):
            parent_str = str(parent)
            for j, child_str in enumerate(children_str):
                if i != j and not child_str in parent_str:
                    break
            else:
                # Found valid parent
                expr_out = parent.__deepcopy__()
                break
        else:
            raise ValueError(f"Could not implicitly determine output expression for input expressions {[str(expr) for expr in exprs_in]}")

    if any(isinstance(expr, einx.expr.stage3.Marker) for root in list(exprs_in) + [expr_out] for expr in root.all()):
        raise ValueError(f"Marker '{expr}' is not allowed")
    if any(isinstance(expr, einx.expr.stage3.Concatenation) for expr in expr_out.all()):
        raise ValueError("Output expression cannot contain concatenation")

    # Call tensor factories
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend) for tensor, expr in zip(tensors_in, exprs_in)]

    # Flatten expressions
    exprs_in, tensors_in = util.flatten(exprs_in, tensors_in, backend)
    expr_out_flat = util.flatten([expr_out])[0]
    assert all(einx.expr.stage3.is_flat(expr) for expr in exprs_in)
    assert einx.expr.stage3.is_flat(expr_out_flat)

    # Transpose and insert trivial axes
    tensors = [util.transpose_broadcast(expr_in, tensor, expr_out_flat, broadcast=False) for expr_in, tensor in zip(exprs_in, tensors_in)]

    # Apply elementwise operation
    tensor = op(*tensors)
    if tensor.shape != expr_out_flat.shape:
        tensor = backend.broadcast_to(tensor, expr_out_flat.shape)

    # Unflatten output expression
    assert not tensor.shape is None
    if tensor.shape != expr_out.shape:
        tensor = backend.reshape(tensor, expr_out.shape)

    return tensor, expr_out

@einx.lru_cache
def parse(description, *tensor_shapes, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    if "->" in description:
        # Description: Inputs and output
        description = description.split("->")
        if len(description) != 2:
            raise ValueError("Operation cannot contain more than one '->'")
        exprs_in, expr_out = description
        exprs_in = exprs_in.split(",")
        if len(exprs_in) != len(tensor_shapes):
            raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensor_shapes)}")

        exprs = einx.expr.solve(
              [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
            + [einx.expr.Condition(expr=expr_out, depth=0)] \
            + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
            cse=cse,
            cse_concat=False,
        )[:len(exprs_in) + 1]
        exprs_in, expr_out = exprs[:-1], exprs[-1]
    else:
        # Description: Only inputs
        exprs_in = description.split(",")

        if len(exprs_in) == 1 and len(tensor_shapes) == 2:
            # Expression contains markers -> add second input expression from marked subexpressions

            expr_in1 = einx.expr.solve(
                [einx.expr.Condition(expr=exprs_in[0], value=tensor_shapes[0], depth=0)] \
              + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
                cse=cse,
                cse_concat=False,
            )[0]

            expr_in2 = einx.expr.stage3.get_marked(expr_in1)
            if not tensor_shapes[1] is None and expr_in2.shape != tensor_shapes[1]:
                raise einx.expr.stage3.SolveError(f"Failed to solve axis values. Expected shape {expr_in2.shape} for second input tensor, got {tensor_shapes[1]}")
            expr_in1 = einx.expr.stage3.demark(expr_in1)
            exprs_in = [expr_in1, expr_in2]
        else:
            if len(exprs_in) != len(tensor_shapes):
                raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensor_shapes)}")

            exprs_in = einx.expr.solve(
                [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
              + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
                cse=cse,
                cse_concat=False,
            )[:len(exprs_in)]

        expr_out = None

    return exprs_in, expr_out

@einx.lru_cache(trace=lambda k: isinstance(k[0], int) and k[0] >= 1)
def elementwise_stage0(description, *tensors, op, backend=None, cse=True, **parameters):
    exprs_in, expr_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensor, expr_out = elementwise_stage3(exprs_in, tensors, expr_out, op=op, backend=backend)
    return tensor

def elementwise(arg0, *args, **kwargs):
    """Applies an element-by-element operation over the given tensors.

    The function flattens all input tensors, applies the given element-by-element operation yielding a single output tensor, and rearranges
    the result to match the output expression (see :doc:`How does einx handle input and output tensors? </faq/flatten>`).

    The `description` argument specifies the input and output expressions. It must meet one of the following formats:

    1. ``input1, input2, ... -> output``
        All input and output expressions are specified explicitly.

    2. ``input1, input2, ...``
        All input expressions are specified explicitly. If one of the input expressions is a parent of or equal to all other input expressions,
        it is used as the output expression. Otherwise, an exception is raised.
        
        Example: ``a b, a`` resolves to ``a b, a -> a b``.

    3. ``input1`` with ``[]``-brackets
        The function accepts two input tensors. `[]`-brackets mark all subexpressions in the
        first input that should also appear in the second input.
        
        Example: ``a [b]`` resolves to ``a b, b``

    Args:
        description: Description string in Einstein notation (see above).
        tensors: Input tensors or tensor factories matching the description string.
        op: Backend elemebt-by-element operation. Must accept the same number of tensors as specified in the description string and comply with numpy broadcasting rules. If `op` is a string, retrieves the attribute of `backend` with the same name.
        backend: Backend to use for all operations. If None, determines the backend from the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults to True.
        graph: Whether to return the graph representation of the operation instead of computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the elementwise operation if `graph=False`, otherwise the graph representation of the operation.

    The following specializations of this function are provided in the same namespace: `add`, `subtract`, `multiply`, `true_divide`, `floor_divide`, `divide`, `logical_and`, `logical_or`, `where`, `less`, `less_equal`, `greater`, `greater_equal`, `equal`, `not_equal`, `maximum`, `minimum`.

    Examples:
        Compute a sum of two vectors:

        >>> a, b = np.random.uniform(size=(10,)), np.random.uniform(size=(10,))
        >>> einx.elementwise("a, a -> a", a, b, op=np.add).shape
        (10,)

        Add a vector on all rows of a matrix:

        >>> a, b = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))
        >>> einx.add("a b, a -> a b", a, b).shape
        (10, 10,)

        Subtract a vector from all columns of a matrix:

        >>> a, b = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))
        >>> einx.subtract("a b, b -> a b", a, b).shape
        (10, 10,)

        Select from one of two choices according to a boolean mask:

        >>> x, mask = np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,))
        >>> einx.where("a, a b, -> a b", mask, x, 0).shape
        (10, 10,)

        Add a bias onto all channels of a tensor:

        >>> x, w = np.random.uniform(size=(4, 16, 16, 64)), np.random.uniform(size=(64,))
        >>> einx.add("b... [c]", x, w).shape
        (4, 16, 16, 64)

        Split a tensor in two parts and add them together:

        >>> x = np.random.uniform(size=(4, 64))
        >>> einx.add("a (b + b) -> a b", x).shape
        (4, 32)
    """
    if isinstance(arg0, str) or (isinstance(arg0, tuple) and isinstance(arg0[0], str)):
        return elementwise_stage0(arg0, *args, **kwargs)
    else:
        return elementwise_stage3(arg0, *args, **kwargs)
elementwise._op_names = _op_names
elementwise.parse = parse


def _make(name):
    def func(*args, **kwargs):
        return elementwise(*args, op=name, **kwargs)
    func.__name__ = name
    globals()[name] = func

for name in _op_names:
    _make(name)

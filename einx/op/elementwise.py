import einx
from . import util
from functools import partial
import numpy as np

_op_names = ["add", "subtract", "multiply", "true_divide", "floor_divide", "divide", "logical_and", "logical_or", "where", "less", "less_equal", "greater", "greater_equal", "equal", "not_equal", "maximum", "minimum"]

@einx.lru_cache(trace=lambda k: k[0] in [1, "tensors_in"])
def elementwise_stage3(exprs_in, tensors_in, expr_out, op, backend=None):
    assert not any(einx.expr.stage3.is_marked(expr) for root in exprs_in for expr in root.all())
    assert not any(einx.expr.stage3.is_marked(expr) for expr in expr_out.all())
    tensors_out, exprs_out = einx.vmap_with_axis_stage3(exprs_in, tensors_in, [expr_out], op, backend=backend)
    assert len(tensors_out) == 1 and len(exprs_out) == 1
    return tensors_out[0], exprs_out[0]

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
            expr_in1 = einx.expr.stage1.parse(exprs_in[0])
            expr_in2 = einx.expr.stage1.get_marked(expr_in1)
            expr_in1 = einx.expr.stage1.demark(expr_in1)
            exprs_in = [expr_in1, expr_in2]

        if len(exprs_in) != len(tensor_shapes):
            raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensor_shapes)}")

        exprs_in = einx.expr.solve(
                [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
              + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
            cse=cse,
            cse_concat=False,
        )[:len(exprs_in)]

        # Implicitly determine output expression: Check if one input expression contains the axis names of all others, and this choice is unique
        in_axis_names = [set(expr.name for expr in root.all() if isinstance(expr, einx.expr.stage3.Axis) and not expr.is_unnamed) for root in exprs_in]

        valid_parents = set()
        for i, parent in enumerate(in_axis_names):
            for j, child in enumerate(in_axis_names):
                if i != j and not child.issubset(parent):
                    break
            else:
                # Found valid parent
                valid_parents.add(exprs_in[i])

        if len(valid_parents) != 1:
            raise ValueError(f"Could not implicitly determine output expression for input expressions {[str(expr) for expr in exprs_in]}")
        expr_out = next(iter(valid_parents)).__deepcopy__()

    return exprs_in, expr_out

@einx.lru_cache(trace=lambda k: isinstance(k[0], int) and k[0] >= 1)
def elementwise_stage0(description, *tensors, op, backend=None, cse=True, **parameters):
    exprs_in, expr_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensor, expr_out = elementwise_stage3(exprs_in, tensors, expr_out, op=op, backend=backend)
    return tensor

def elementwise(arg0, *args, **kwargs):
    """Applies an element-by-element operation over the given tensors. Specializes :func:`einx.vmap_with_axis`.

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
    if isinstance(arg0, str):
        return elementwise_stage0(arg0, *args, **kwargs)
    else:
        return elementwise_stage3(arg0, *args, **kwargs)
elementwise._op_names = _op_names
elementwise.parse = parse


def _make(name):
    def func(*args, **kwargs):
        return elementwise(*args, op=name, **kwargs)
    func.__name__ = name
    func.__doc__ = f"Alias for :func:`einx.elementwise` with ``op=\"{name}\"``"
    globals()[name] = func

for name in _op_names:
    _make(name)

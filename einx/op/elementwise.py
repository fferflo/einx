import einx
from . import util
from functools import partial
import numpy as np
from typing import Callable, Union
import numpy.typing as npt

@einx.lru_cache(trace=lambda t, c: lambda exprs_in, tensors_in, expr_out, op, backend=None: c(exprs_in, [t(x) for x in tensors_in], expr_out, op))
def elementwise_stage3(exprs_in, tensors_in, expr_out, op, backend=None):
    if backend is None:
        backend = einx.backend.get(tensors_in)
    elif isinstance(backend, str):
        backend = einx.backend.get(backend)
    for root in list(exprs_in) + [expr_out]:
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")

    assert not any(einx.expr.stage3.is_marked(expr) for root in exprs_in for expr in root.all())
    assert not any(einx.expr.stage3.is_marked(expr) for expr in expr_out.all())

    # Call tensor factories
    def get_name(s):
        if s == "add":
            return "bias"
        elif s == "multiply":
            return "scale"
        else:
            return s
    tensors_in = [einx.param.instantiate(tensor, expr.shape, backend, name=get_name(util._op_to_str(op)), init=util._op_to_str(op)) for tensor, expr in zip(tensors_in, exprs_in)]

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
              [einx.expr.Equation(expr_in, tensor_shape) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
            + [einx.expr.Equation(expr_out,)] \
            + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
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
                [einx.expr.Equation(expr_in, tensor_shape) for expr_in, tensor_shape in zip(exprs_in, tensor_shapes)] \
              + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
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

@einx.lru_cache(trace=lambda t, c: lambda description, *tensors, backend=None, **kwargs: c(description, *[t(x) for x in tensors], **kwargs))
def elementwise(description: str, *tensors: einx.Tensor, op: Callable, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
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
    """
    exprs_in, expr_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], cse=cse, **parameters)
    tensor, expr_out = elementwise_stage3(exprs_in, tensors, expr_out, op=op, backend=backend)
    return tensor
elementwise.parse = parse



def add(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="add"``"""
    return elementwise(description, *tensors, op="add", backend=backend, cse=cse, **parameters)

def add_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="add", **kwargs)

def subtract(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="subtract"``"""
    return elementwise(description, *tensors, op="subtract", backend=backend, cse=cse, **parameters)

def subtract_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="subtract", **kwargs)

def multiply(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="multiply"``"""
    return elementwise(description, *tensors, op="multiply", backend=backend, cse=cse, **parameters)

def multiply_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="multiply", **kwargs)

def true_divide(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="true_divide"``"""
    return elementwise(description, *tensors, op="true_divide", backend=backend, cse=cse, **parameters)

def true_divide_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="true_divide", **kwargs)

def floor_divide(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="floor_divide"``"""
    return elementwise(description, *tensors, op="floor_divide", backend=backend, cse=cse, **parameters)

def floor_divide_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="floor_divide", **kwargs)

def divide(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="divide"``"""
    return elementwise(description, *tensors, op="divide", backend=backend, cse=cse, **parameters)

def divide_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="divide", **kwargs)

def logical_and(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="logical_and"``"""
    return elementwise(description, *tensors, op="logical_and", backend=backend, cse=cse, **parameters)

def logical_and_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="logical_and", **kwargs)

def logical_or(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="logical_or"``"""
    return elementwise(description, *tensors, op="logical_or", backend=backend, cse=cse, **parameters)

def logical_or_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="logical_or", **kwargs)

def where(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="where"``"""
    return elementwise(description, *tensors, op="where", backend=backend, cse=cse, **parameters)

def where_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="where", **kwargs)

def less(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="less"``"""
    return elementwise(description, *tensors, op="less", backend=backend, cse=cse, **parameters)

def less_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="less", **kwargs)

def less_equal(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="less_equal"``"""
    return elementwise(description, *tensors, op="less_equal", backend=backend, cse=cse, **parameters)

def less_equal_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="less_equal", **kwargs)

def greater(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="greater"``"""
    return elementwise(description, *tensors, op="greater", backend=backend, cse=cse, **parameters)

def greater_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="greater", **kwargs)

def greater_equal(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="greater_equal"``"""
    return elementwise(description, *tensors, op="greater_equal", backend=backend, cse=cse, **parameters)

def greater_equal_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="greater_equal", **kwargs)

def equal(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="equal"``"""
    return elementwise(description, *tensors, op="equal", backend=backend, cse=cse, **parameters)

def equal_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="equal", **kwargs)

def not_equal(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="not_equal"``"""
    return elementwise(description, *tensors, op="not_equal", backend=backend, cse=cse, **parameters)

def not_equal_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="not_equal", **kwargs)

def maximum(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="maximum"``"""
    return elementwise(description, *tensors, op="maximum", backend=backend, cse=cse, **parameters)

def maximum_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="maximum", **kwargs)

def minimum(description: str, *tensors: einx.Tensor, backend: Union[einx.Backend, str, None] = None, cse: bool = True, **parameters: npt.ArrayLike) -> einx.Tensor:
    """Specialization of :func:`einx.elementwise` with ``op="minimum"``"""
    return elementwise(description, *tensors, op="minimum", backend=backend, cse=cse, **parameters)

def minimum_stage3(*args, **kwargs):
    return elementwise_stage3(*args, op="minimum", **kwargs)
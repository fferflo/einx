import einx
from . import util
import numpy as np
from functools import partial

_op_names = ["sum", "mean", "var", "std", "prod", "count_nonzero", "any", "all", "max", "min"]
_any = any # Is overwritten below

@einx.lru_cache(trace=lambda k: k[0] in [1, "tensors_in"])
def reduce_stage3(exprs_in, tensors_in, exprs_out, op, backend=None):
    return einx.vmap_with_axis_stage3(exprs_in, tensors_in, exprs_out, op, backend=backend)

@einx.lru_cache
def parse(description, *tensors_shapes, keepdims=None, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    if "->" in description:
        if not keepdims is None:
            raise ValueError("keepdims cannot be given when using '->'")
        description = description.split("->")
        if len(description) != 2:
            raise ValueError("Operation cannot contain more than one '->'")

        exprs_in, exprs_out = description
        exprs_in = exprs_in.split(",")
        exprs_out = exprs_out.split(",")
        exprs = exprs_in + exprs_out
        if len(exprs_in) != len(tensors_shapes):
            raise ValueError(f"Expected {len(exprs_in)} input tensors, got {len(tensors_shapes)}")

        exprs = einx.expr.solve(
                [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensors_shapes)] \
              + [einx.expr.Condition(expr=expr_out, depth=0) for expr_out in exprs_out] \
              + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
            cse=cse,
        )[:len(exprs_in) + len(exprs_out)]
        exprs_in, exprs_out = exprs[:len(exprs_in)], exprs[len(exprs_in):]

        # If no axes are marked for reduction in exprs_in, mark all axes that don't appear in exprs_out
        if not _any(einx.expr.stage3.is_marked(axis) for expr_in in exprs_in for axis in expr_in.all()):
            axes_names_out = set(axis.name for expr in exprs_out for axis in expr.all() if isinstance(axis, einx.expr.stage3.Axis))
            exprs_in = [einx.expr.stage3.mark(expr, lambda expr: isinstance(expr, einx.expr.stage3.Axis) and expr.name not in axes_names_out) for expr in exprs_in]

    else:
        exprs_in = description.split(",")
        if len(exprs_in) != 1:
            raise ValueError("Operation with implicit output shape cannot contain more than one input expression")
        if len(exprs_in) != len(tensors_shapes):
            raise ValueError(f"Expected {len(exprs_in)} input tensor(s), got {len(tensors_shapes)}")

        exprs_in = einx.expr.solve(
                [einx.expr.Condition(expr=expr_in, value=tensor_shape, depth=0) for expr_in, tensor_shape in zip(exprs_in, tensors_shapes)] \
              + [einx.expr.Condition(expr=k, value=np.asarray(v)[..., np.newaxis]) for k, v in parameters.items()],
            cse=cse,
        )[:len(exprs_in)]

        if not _any(isinstance(expr, einx.expr.stage3.Marker) for root in exprs_in for expr in root.all()):
            raise ValueError("No axes are marked for reduction")

        # Determine output expressions by removing markers from input expressions
        def replace(expr):
            if isinstance(expr, einx.expr.stage3.Marker):
                if keepdims:
                    return [einx.expr.stage3.Axis(None, 1)]
                else:
                    return []
        exprs_out = [einx.expr.stage3.replace(expr_in, replace) for expr_in in exprs_in]

    return exprs_in, exprs_out

@einx.lru_cache(trace=lambda k: isinstance(k[0], int) and k[0] >= 1)
def reduce_stage0(description, *tensors, op, keepdims=None, backend=None, cse=True, **parameters):
    exprs_in, exprs_out = parse(description, *[einx.param.get_shape(tensor) for tensor in tensors], keepdims=keepdims, cse=cse, **parameters)
    tensors, exprs_out = reduce_stage3(exprs_in, tensors, exprs_out, op=op, backend=backend)
    return tensors[0] if len(exprs_out) == 1 else tensors

def reduce(arg0, *args, **kwargs):
    """Applies a reduction operation on the given tensors. Specializes :func:`einx.vmap_with_axis`.

    The function flattens all input tensors, applies the given reduction operation and rearranges
    the result to match the output expression (see :doc:`How does einx handle input and output tensors? </faq/flatten>`).

    The `description` argument specifies the input and output expressions, as well as reduced axes. It must meet one of the following formats:

    1. ``input1, input2, ... -> output1, output2, ...``
        All input and output expressions are specified explicitly. Reduced axes are marked with ``[]``-brackets in the input expressions. If no axes are
        marked, reduces all axes that do not appear in one of the output expressions.

    2. ``input1``
        A single input expression is specified. Reduced axes are marked with ``[]``-brackets. The output expression is determined by removing all marked expressions
        from the input expression.

        Example: ``a [b]`` resolves to ``a b -> a``.

    Args:
        description: Description string in Einstein notation (see above).
        tensors: Input tensors or tensor factories matching the description string.
        op: Backend reduction operation. Is called with ``op(tensor, axis=...)``. If `op` is a string, retrieves the attribute of `backend` with the same name.
        keepdims: Whether to replace marked expressions with 1s instead of removing them. Defaults to False.
        backend: Backend to use for all operations. If None, determines the backend from the input tensors. Defaults to None.
        cse: Whether to apply common subexpression elimination to the expressions. Defaults to True.
        graph: Whether to return the graph representation of the operation instead of computing the result. Defaults to False.
        **parameters: Additional parameters that specify values for single axes, e.g. ``a=4``.

    Returns:
        The result of the reduction operation if ``graph=False``, otherwise the graph representation of the operation.

    Examples:
        Compute mean along rows of a matrix:

        >>> x = np.random.uniform(size=(16, 20))
        >>> einx.mean("a b -> b", x).shape
        (20,)
        >>> einx.mean("[a] b -> b", x).shape
        (20,)
        >>> einx.mean("[a] b", x).shape
        (20,)

        Compute sum along rows of a matrix and broadcast to the original shape:

        >>> x = np.random.uniform(size=(16, 20))
        >>> einx.sum("[a] b -> a b", x).shape
        (16, 20,)

        Sum pooling with kernel size 2:

        >>> x = np.random.uniform(size=(4, 16, 16, 3))
        >>> einx.sum("b (s [s2])... c", x, s2=2).shape
        (4, 8, 8, 3)

        Compute variance per channel over an image:

        >>> x = np.random.uniform(size=(256, 256, 3))
        >>> einx.var("[...] c", x).shape
        (3,)
    """
    if isinstance(arg0, str):
        return reduce_stage0(arg0, *args, **kwargs)
    else:
        return reduce_stage3(arg0, *args, **kwargs)
reduce._op_names = _op_names
reduce.parse = parse

def _make(name):
    def func(*args, **kwargs):
        return reduce(*args, op=name, **kwargs)
    func.__name__ = name
    func.__doc__ = f"Alias for :func:`einx.reduce` with ``op=\"{name}\"``"
    globals()[name] = func

for name in _op_names:
    _make(name)
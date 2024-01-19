import einx
from . import util
import numpy as np
from functools import partial

_op_names = ["sum", "mean", "var", "std", "prod", "count_nonzero", "any", "all", "max", "min", "logsumexp"]
_any = any # Is overwritten below

@einx.lru_cache(trace=lambda t, c: lambda expr_in, tensor_in, expr_out, op, backend=None: c(expr_in, t(tensor_in), expr_out, op=op))
def reduce_stage3(expr_in, tensor_in, expr_out, op, backend=None):
    for root in [expr_in, expr_out]:
        for expr in root.all():
            if isinstance(expr, einx.expr.stage3.Concatenation):
                raise ValueError("Concatenation not allowed")
    tensors_out, exprs_out = einx.vmap_with_axis_stage3([expr_in], [tensor_in], [expr_out], op, backend=backend)
    return tensors_out[0], exprs_out[0]

@einx.lru_cache
def parse(description, tensor_shape, keepdims=None, cse=True, **parameters):
    description, parameters = einx.op.util._clean_description_and_parameters(description, parameters)

    if "," in description:
        raise ValueError("Only a single input and output expression is allowed")

    if "->" in description:
        if not keepdims is None:
            raise ValueError("keepdims cannot be given when using '->'")
        description = description.split("->")
        if len(description) != 2:
            raise ValueError("Operation cannot contain more than one '->'")

        expr_in, expr_out = description

        expr_in, expr_out = einx.expr.solve(
                [einx.expr.Equation(expr_in, tensor_shape)] \
              + [einx.expr.Equation(expr_out)] \
              + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
            cse=cse,
            cse_in_markers=True,
        )[:2]

        # If no axes are marked for reduction in expr_in, mark all axes that don't appear in expr_out
        if not _any(einx.expr.stage3.is_marked(expr) for expr in expr_in.all()):
            axes_names_out = set(axis.name for axis in expr_out.all() if isinstance(axis, einx.expr.stage3.Axis))
            expr_in = einx.expr.stage3.mark(expr_in, lambda expr: isinstance(expr, einx.expr.stage3.Axis) and expr.name not in axes_names_out)

    else:
        expr_in = description

        expr_in = einx.expr.solve(
                [einx.expr.Equation(expr_in, tensor_shape)] \
              + [einx.expr.Equation(k, np.asarray(v)[..., np.newaxis], depth1=None, depth2=None) for k, v in parameters.items()],
            cse=cse,
            cse_in_markers=True,
        )[0]

        if not _any(isinstance(expr, einx.expr.stage3.Marker) for expr in expr_in.all()):
            raise ValueError("No axes are marked for reduction")

        # Determine output expressions by removing markers from input expressions
        def replace(expr):
            if isinstance(expr, einx.expr.stage3.Marker):
                if keepdims:
                    return [einx.expr.stage3.Axis(None, 1)]
                else:
                    return []
        expr_out = einx.expr.stage3.replace(expr_in, replace)

    return expr_in, expr_out

@einx.lru_cache(trace=lambda t, c: lambda description, tensor, backend=None, **kwargs: c(description, t(tensor), **kwargs))
def reduce_stage0(description, tensor, op, keepdims=None, backend=None, cse=True, **parameters):
    expr_in, expr_out = parse(description, einx.param.get_shape(tensor), keepdims=keepdims, cse=cse, **parameters)
    tensor, expr_out = reduce_stage3(expr_in, tensor, expr_out, op=op, backend=backend)
    return tensor

def reduce(arg0, *args, **kwargs):
    """Applies a reduction operation on the given tensors. Specializes :func:`einx.vmap_with_axis`.

    The function flattens all input tensors, applies the given reduction operation and rearranges
    the result to match the output expression (see :doc:`How does einx handle input and output tensors? </faq/flatten>`).

    The `description` argument specifies the input and output expressions, as well as reduced axes. It must meet one of the following formats:

    1. ``input -> output``
        Input and output expressions are specified explicitly. Reduced axes are marked with ``[]``-brackets in the input expression. If no axes are
        marked, reduces all axes that do not appear in the output expression.

    2. ``input``
        A single input expression is specified. Reduced axes are marked with ``[]``-brackets. The output expression is determined by removing all marked expressions
        from the input expression.

        Example: ``a [b]`` resolves to ``a b -> a``.

    Args:
        description: Description string in Einstein notation (see above).
        tensor: Input tensor or tensor factory matching the description string.
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
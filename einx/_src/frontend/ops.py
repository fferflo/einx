from .api import api
from .types import Tensor
from .backend import Backend
import numpy.typing as npt
from typing import Union
import warnings


def _args_return(single, kwargs=""):
    s = "" if single else "s"
    ies = "y" if single else "ies"
    return f"""Args:
    description: Description string for the operation in einx notation.
    {"tensor" if single else "*tensors"}: Input tensor{s} or tensor factor{ies} matching the description string.
    backend: Backend to use for all operations. If None, uses the :doc:`default backend </gettingstarted/backends>` for the given setting. Defaults to None.
    graph: Whether to return the compiled code representation of this operation instead of
        computing the result. Defaults to False.
    {kwargs}
    **parameters: Additional parameters that specify dimension sizes, e.g. ``a=4``.

Returns:
    The result of the operation if ``graph=False``, otherwise the compiled code
    representation of the operation.
"""


@api
def id(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor | tuple[Tensor, ...]:
    return backend.id(description, *tensors, **parameters)


id.__doc__ = f"""Compute the identity-map of values in the given tensor.

In the simplest case, the elementary operation has the signature ``[] -> []`` and returns the input as-is. If there are more than one
input/ output or concatenated inputs/ outputs, the ordered tuple of inputs is returned as-is.

{_args_return(False)}
"""


def _make_reduction_doc(op, line1, output):
    return f"""{line1}.

The elementary operation has the signature ``[...] -> []`` and {output}.

If there is no output expression, it is determined implicitly by removing all bracketed expressions from the input expression.
For example, the following operations compute the same output:

..  code-block:: python

    y = einx.{op}("a [b]", x)
    y = einx.{op}("a [b] -> a", x)

If there are no brackets in the expression, brackets are implicitly placed around all axes that do not appear in the output expression.
For example, the following operations compute the same output:

..  code-block:: python

    y = einx.{op}("a b -> a", x)
    y = einx.{op}("a [b] -> a", x)

{_args_return(True)}
"""


def _keepdims_warning(keepdims):
    if keepdims is not None:
        warnings.warn(
            "The 'keepdims' argument in einx reduction operations is deprecated and will be removed in a future version. "
            "Please use a flattened axis instead.\nFor example, instead of\n\n"
            '    einx.{op}("a [b]", x, keepdims=True)\n\nwrite\n\n    einx.{op}("a ([b])", x).\n',
            DeprecationWarning,
            stacklevel=6,
        )


@api
def sum(description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.sum(description, tensor, keepdims=keepdims, **parameters)


sum.__doc__ = _make_reduction_doc("sum", "Compute the sum of values in the given tensor", "computes the sum of all values of the input tensor")


@api
def mean(description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.mean(description, tensor, keepdims=keepdims, **parameters)


mean.__doc__ = _make_reduction_doc("mean", "Compute the mean of values in the given tensor", "computes the mean of all values of the input tensor")


@api
def var(description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.var(description, tensor, keepdims=keepdims, **parameters)


var.__doc__ = _make_reduction_doc("var", "Compute the variance of values in the given tensor", "computes the variance of all values of the input tensor")


@api
def std(description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.std(description, tensor, keepdims=keepdims, **parameters)


std.__doc__ = _make_reduction_doc(
    "std", "Compute the standard deviation of values in the given tensor", "computes the standard deviation of all values of the input tensor"
)


@api
def prod(description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.prod(description, tensor, keepdims=keepdims, **parameters)


prod.__doc__ = _make_reduction_doc("prod", "Compute the product of values in the given tensor", "computes the product of all values of the input tensor")


@api
def count_nonzero(
    description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike
) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.count_nonzero(description, tensor, keepdims=keepdims, **parameters)


count_nonzero.__doc__ = _make_reduction_doc(
    "count_nonzero", "Counts non-zero values in the given tensor", "counts the number of all non-zero values of the input tensor"
)


@api
def any(description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.any(description, tensor, keepdims=keepdims, **parameters)


any.__doc__ = _make_reduction_doc(
    "any", "Compute the logical disjunction (OR) of values in the given tensor", "computes the logical disjunction (OR) of all values of the input tensor"
)


@api
def all(description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.all(description, tensor, keepdims=keepdims, **parameters)


all.__doc__ = _make_reduction_doc(
    "all", "Compute the logical conjunction (AND) of values in the given tensor", "computes the logical conjunction (AND) of all values of the input tensor"
)


@api
def max(description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.max(description, tensor, keepdims=keepdims, **parameters)


max.__doc__ = _make_reduction_doc("max", "Compute the maximum of values in the given tensor", "computes the maximum of all values of the input tensor")


@api
def min(description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.min(description, tensor, keepdims=keepdims, **parameters)


min.__doc__ = _make_reduction_doc("min", "Compute the minimum of values in the given tensor", "computes the minimum of all values of the input tensor")


@api
def logsumexp(description: str, tensor: Tensor, *, keepdims: bool | None = None, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    _keepdims_warning(keepdims)
    return backend.logsumexp(description, tensor, keepdims=keepdims, **parameters)


logsumexp.__doc__ = _make_reduction_doc(
    "logsumexp", "Compute the log-sum-exp of values in the given tensor", "computes the log-sum-exp of all values of the input tensor"
)


@api
def dot(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.dot(description, *tensors, **parameters)


dot.__doc__ = f"""Compute the dot-product of values in the given tensors.

In the simplest case, the elementary operation has the signature ``[a], [a] -> []`` and computes the dot-product over the two input vectors.
If more than two tensors or more than two axes are passed to the elementary operation, the dot-product is applied sequentially in an unspecified
order to all pairs of dimensions with the same name.

If there are no brackets in the expression, brackets are placed implicitly around all axes that do not appear in the output expression. For example,
the following operations compute the same output:

..  code-block:: python

    z = einx.dot("a b, b c -> a c", x, y)
    z = einx.dot("a [b], [b] c -> a c", x, y)

{_args_return(False)}
    """


def _make_elwise_doc(op, line1, output, nargs):
    if nargs == 2:
        code = f"""
    z = einx.{op}("a b, a", x, y)
    z = einx.{op}("a b, a -> a b", x, y)

    z = einx.{op}("a b, a b", x, y)
    z = einx.{op}("a b, a b -> a b", x, y)

    z = einx.{op}("a b, b a", x, y)
    # raises an exception due to ambiguous output expression
"""
    elif nargs == 3:
        code = f"""
    w = einx.{op}("a b, a, b", x, y, z)
    w = einx.{op}("a b, a, b -> a b", x, y, z)

    w = einx.{op}("a b, a b, a b", x, y, z)
    w = einx.{op}("a b, a b, a b -> a b", x, y, z)

    w = einx.{op}("a b, b a, a b", x, y, z)
    # raises an exception due to ambiguous output expression
"""
    else:
        raise ValueError("Invalid nargs")

    return f"""{line1}.

The elementary operation {output}.

If there is no output expression, one of the input expressions is implicitly used as output expression if it contains the axis names of all other
inputs and if this choice is unique. For example, the following pairs of operations compute the same output:

..  code-block:: python

    {code}

{_args_return(False)}
"""


@api
def add(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.add(description, *tensors, **parameters)


add.__doc__ = _make_elwise_doc("add", "Compute the sum of values of multiple given tensors", "takes any number of scalars as input and returns their sum", 2)


@api
def subtract(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.subtract(description, *tensors, **parameters)


subtract.__doc__ = _make_elwise_doc(
    "subtract", "Computes the difference between values of two given tensors", "takes two scalars as input and subtracts the second from the first", 2
)


@api
def multiply(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.multiply(description, *tensors, **parameters)


multiply.__doc__ = _make_elwise_doc(
    "multiply", "Compute the product of values of multiple given tensors", "takes any number of scalars as input and returns their product", 2
)


@api
def true_divide(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.true_divide(description, *tensors, **parameters)


true_divide.__doc__ = _make_elwise_doc(
    "true_divide", "Computes the true division between values of two given tensors", "takes two scalars as input and divides the first by the second", 2
)


@api
def floor_divide(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.floor_divide(description, *tensors, **parameters)


floor_divide.__doc__ = _make_elwise_doc(
    "floor_divide", "Computes the floor division between values of two given tensors", "takes two scalars as input and divides the first by the second", 2
)


@api
def divide(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.divide(description, *tensors, **parameters)


divide.__doc__ = _make_elwise_doc(
    "divide", "Computes the division between values of two given tensors", "takes two scalars as input and divides the first by the second", 2
)


@api
def logical_and(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.logical_and(description, *tensors, **parameters)


logical_and.__doc__ = _make_elwise_doc(
    "logical_and",
    "Compute the logical conjunction (AND) of values of multiple given tensors",
    "takes any number of scalars as input and returns their logical conjunction (AND)",
    2,
)


@api
def logical_or(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.logical_or(description, *tensors, **parameters)


logical_or.__doc__ = _make_elwise_doc(
    "logical_or",
    "Compute the logical disjunction (OR) of values of multiple given tensors",
    "takes any number of scalars as input and returns their logical disjunction (OR)",
    2,
)


@api
def where(description: str, mask: Tensor, x: Tensor, y: Tensor, *, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.where(description, mask, x, y, **parameters)


where.__doc__ = _make_elwise_doc(
    "where",
    "Conditionally select values from two tensors based on a boolean mask",
    "takes three scalars as input (mask, true_val, false_val) and returns true_val if mask is true, otherwise false_val",
    3,
)


@api
def maximum(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.maximum(description, *tensors, **parameters)


maximum.__doc__ = _make_elwise_doc(
    "maximum", "Compute the maximum of values of multiple given tensors", "takes any number of scalars as input and returns their maximum", 2
)


@api
def minimum(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.minimum(description, *tensors, **parameters)


minimum.__doc__ = _make_elwise_doc(
    "minimum", "Compute the minimum of values of multiple given tensors", "takes any number of scalars as input and returns their minimum", 2
)


@api
def less(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.less(description, *tensors, **parameters)


less.__doc__ = _make_elwise_doc(
    "less",
    "Computes the less-than comparison between values of two given tensors",
    "takes two scalars as input and returns true if the first is less than the second, otherwise false",
    2,
)


@api
def less_equal(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.less_equal(description, *tensors, **parameters)


less_equal.__doc__ = _make_elwise_doc(
    "less_equal",
    "Computes the less-than-or-equal comparison between values of two given tensors",
    "takes two scalars as input and returns true if the first is less than or equal to the second, otherwise false",
    2,
)


@api
def greater(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.greater(description, *tensors, **parameters)


greater.__doc__ = _make_elwise_doc(
    "greater",
    "Computes the greater-than comparison between values of two given tensors",
    "takes two scalars as input and returns true if the first is greater than the second, otherwise false",
    2,
)


@api
def greater_equal(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.greater_equal(description, *tensors, **parameters)


greater_equal.__doc__ = _make_elwise_doc(
    "greater_equal",
    "Computes the greater-than-or-equal comparison between values of two given tensors",
    "takes two scalars as input and returns true if the first is greater than or equal to the second, otherwise false",
    2,
)


@api
def equal(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.equal(description, *tensors, **parameters)


equal.__doc__ = _make_elwise_doc(
    "equal",
    "Computes the equality comparison between values of two given tensors",
    "takes two scalars as input and returns true if they are equal, otherwise false",
    2,
)


@api
def not_equal(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.not_equal(description, *tensors, **parameters)


not_equal.__doc__ = _make_elwise_doc(
    "equal",
    "Computes the non-equality comparison between values of two given tensors",
    "takes two scalars as input and returns false if they are equal, otherwise true",
    2,
)


@api
def logaddexp(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.logaddexp(description, *tensors, **parameters)


logaddexp.__doc__ = _make_elwise_doc(
    "logaddexp", "Compute the log-sum-exp of values of multiple given tensors", "takes any number of scalars as input and returns their log-sum-exp", 2
)


@api
def get_at(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.get_at(description, *tensors, **parameters)


get_at.__doc__ = f"""Retrieves values from a tensor at the coordinates specified by another tensor.

The elementary operation has the signature ``[...] , [n] -> []``. The first argument is the n-dimensional value tensor, the second argument
specifies a single n-dimensional coordinate, and the result is the value at that coordinate.

For 1-dimensional value tensors, the elementary operation also accepts the signature ``[...] , [] -> []``. For example, the following two
operations compute the same output:

..  code-block:: python

    y = einx.get_at("[h], p [1] -> p", x, idx)
    y = einx.get_at("[h], p     -> p", x, idx[:, 0])

The elementary operation also accepts multiple coordinate tensors as input, in which case they are concatenated first. The length of the resulting
coordinate vector must equal the number of dimensions of the value tensor. For example, the following two operations compute the same output:

..  code-block:: python

    y = einx.get_at("[a b c d], p [n]           -> p", x, idx)
    y = einx.get_at("[a b c d], p, p [2], p [1] -> p", x, idx[:, 0], idx[:, 1:3], idx[:, 3:4])

{_args_return(False)}
"""


def _update_at_doc(op, line1, value1, value2):
    return f"""{line1} at the coordinates specified by an indexing tensor.

The elementary operation has the signature ``[...] , [n], [] -> [...]``. The first argument is the n-dimensional value tensor, the second argument
specifies a single n-dimensional coordinate, the third argument is the {value1} value, and the result is the value tensor with the value {value2}
at that coordinate.

For 1-dimensional value tensors, the elementary operation also accepts the signature ``[...] , [], [] -> []``. For example, the following two
operations compute the same output:

..  code-block:: python

    y = einx.{op}("p [h], p [1], p -> p [h]", x, idx,       update)
    y = einx.{op}("p [h], p,     p -> p [h]", x, idx[:, 0], update)

The elementary operation also accepts multiple coordinate tensors as input, in which case they are concatenated first. The length of the resulting
coordinate vector must equal the number of dimensions of the value tensor. The update tensor always is the last argument.
For example, the following two operations compute the same output:

..  code-block:: python

    y = einx.{op}("p [a b c d], p [n],           p -> p", x, idx,                                 update)
    y = einx.{op}("p [a b c d], p, p [2], p [1], p -> p", x, idx[:, 0], idx[:, 1:3], idx[:, 3:4], update)

If no output expression is given, it is implicitly chosen to be the same as the input expression of the value tensor. For example,
the following two operations compute the same output:

..  code-block:: python

    y = einx.{op}("b [h w] c, b p [2], b p c", x, idx, update)
    y = einx.{op}("b [h w] c, b p [2], b p c -> b [h w] c", x, idx, update)

The order in which the updates are applied depends on the chosen backend. The operation also may or may not update the target tensor inplace.
Please check by inspecting the code representation (by passing ``graph=True``).

{_args_return(False)}
    """


@api
def set_at(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.set_at(description, *tensors, **parameters)


set_at.__doc__ = _update_at_doc("set_at", "Sets values in a target tensor from an update tensor", "new", "overwritten")


@api
def add_at(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.add_at(description, *tensors, **parameters)


add_at.__doc__ = _update_at_doc("add_at", "Adds values from an update tensor to a target tensor", "added", "added")


@api
def subtract_at(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.subtract_at(description, *tensors, **parameters)


subtract_at.__doc__ = _update_at_doc("subtract_at", "Subtracts values from a target tensor by an update tensor", "subtracted", "subtracted")


@api
def softmax(description: str, tensor: Tensor, *, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.softmax(description, tensor, **parameters)


softmax.__doc__ = f"""Compute the softmax of values in the given tensor.

The elementary operation has the signature ``[...] -> [...]`` and computes the softmax over all input values.

If there is no output expression, it is chosen to be the same as the input expression. For example, the following operations compute the same output:

..  code-block:: python

    y = einx.softmax("a [b]", x)
    y = einx.softmax("a [b] -> a [b]", x)

{_args_return(True)}
"""


@api
def log_softmax(description: str, tensor: Tensor, *, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.log_softmax(description, tensor, **parameters)


log_softmax.__doc__ = f"""Compute the log-softmax of values in the given tensor.

The elementary operation has the signature ``[...] -> [...]`` and computes the log-softmax over all input values.

If there is no output expression, it is chosen to be the same as the input expression. For example, the following operations compute the same output:

..  code-block:: python

    y = einx.log_softmax("a [b]", x)
    y = einx.log_softmax("a [b] -> a [b]", x)

{_args_return(True)}
"""


@api
def sort(description: str, tensor: Tensor, *, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.sort(description, tensor, **parameters)


sort.__doc__ = f"""Returns values in the given tensor sorted in ascending order.

The elementary operation has the signature ``[a] -> [a]`` and returns the values sorted along the single axis in ascending order.

If there is no output expression, it is chosen to be the same as the input expression. For example, the following operations compute the same output:

..  code-block:: python

    y = einx.sort("a [b]", x)
    y = einx.sort("a [b] -> a [b]", x)

{_args_return(True)}
"""


@api
def argsort(description: str, tensor: Tensor, *, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.argsort(description, tensor, **parameters)


argsort.__doc__ = f"""Returns the indices that would sort values in the given tensor in ascending order.

The elementary operation has the signature ``[a] -> [a]`` and returns the indices that would sort values along the single axis in ascending order.

If there is no output expression, it is chosen to be the same as the input expression. For example, the following operations compute the same output:

..  code-block:: python

    y = einx.argsort("a [b]", x)
    y = einx.argsort("a [b] -> a [b]", x)

{_args_return(True)}
"""


@api
def flip(description: str, tensor: Tensor, *, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.flip(description, tensor, **parameters)


flip.__doc__ = f"""Reverse the order of elements in the given tensor.

The elementary operation has the signature ``[...] -> [...]`` and reverses the order of elements along all axes.

If there is no output expression, it is chosen to be the same as the input expression. For example, the following operations compute the same output:

..  code-block:: python

    y = einx.flip("a [b]", x)
    y = einx.flip("a [b] -> a [b]", x)

{_args_return(True)}
"""


@api
def roll(description: str, tensor: Tensor, *, shift: npt.ArrayLike, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.roll(description, tensor, shift=shift, **parameters)


roll.__doc__ = f"""Rolls the elements in the given tensor by the specified shift amounts.

The elementary operation has the signature ``[...] -> [...]`` and rolls elements separately along all axes. Elements that are rolled beyond the last position
are re-introduced at the first position.

If there is no output expression, it is chosen to be the same as the input expression. For example, the following operations compute the same output:

..  code-block:: python

    y = einx.roll("a [b]", x, shift=4)
    y = einx.roll("a [b] -> a [b]", x, shift=4)

{
    _args_return(
        True,
        (
            "shift: Amounts by which elements are shifted along each axis. Can be a single integer or a list of integers matching "
            "the number of axes in the tensor."
        ),
    )
}
"""


def _make_argfind_doc(op, find):
    return f"""Find the coordinates of the {find} values in the given tensor.

The elementary operation has the signature ``[...] -> [n]``. It takes an n-dimensional tensor as input and returns the n-dimensional coordinate vector of
the {find} value.

For 1-dimensional tensors, the elementary operation also accepts the signature ``[a] -> []``. For example, the following two operations compute
the same output:

..  code-block:: python

    y = einx.argma{op}x("a [b] -> a [1]", x)
    y = einx.{op}("a [b] -> a ", x)

If no output is given, it is determined implicitly by replacing a single bracketed expression in the input with ``[n]``. For example, the
following two operations compute the same output:

..  code-block:: python

    y = einx.{op}("a [b c]", x)
    y = einx.{op}("a [b c] -> a [2]", x)

{_args_return(True)}
"""


@api
def argmax(description: str, tensor: Tensor, *, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.argmax(description, tensor, **parameters)


argmax.__doc__ = _make_argfind_doc("argmax", "maximum")


@api
def argmin(description: str, tensor: Tensor, *, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor:
    return backend.argmin(description, tensor, **parameters)


argmin.__doc__ = _make_argfind_doc("argmin", "minimum")


ops = [
    globals()[name] for name in dir() if name[0].islower() and name not in ["api", "npt", "Backend", "Tensor", "Union", "Tuple"] and callable(globals()[name])
]

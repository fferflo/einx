def _make_doc_adapt_numpylike_reduce():
    return """Adapts an operation with a Numpy-like reduce signature to einx notation.

The operation is expected to have the following Numpy-like signature:

..  code-block:: python

    def op(tensor: Tensor, axis: Tuple[int]) -> Tensor:
        ...

It should return a tensor with the same shape as the input tensor, but with axes at positions specified in `axis` removed.

Args:
    op: The operation that will be adapted to einx notation.

Returns:
    A new operation that follows einx notation and internally invokes the original operation.
"""


def _make_doc_adapt_numpylike_elementwise():
    return """Adapts an operation with a Numpy-like element-wise signature to einx notation.

The operation is expected to have the following Numpy-like signature:

..  code-block:: python

    def op(*tensors: Tensor) -> Tensor:
        ...

All input tensors are guaranteed to be broadcastable according to Numpy's `broadcasting rules
<https://numpy.org/doc/stable/user/basics.broadcasting.html#broadcastable-arrays>`__. They are
additionally guaranteed to have the same number of dimensions. The function should return a tensor
with the broadcasted shape of the input tensors.

Args:
    op: The operation that will be adapted to einx notation.

Returns:
    A new operation that follows einx notation and internally invokes the original operation.
"""


def _make_doc_adapt_with_vmap(framework, vmap):
    return f"""Adapts an operation to einx notation using {vmap}.

The operation is expected to have one of the following signatures:

..  code-block:: python

    def op(*tensors: Tensor) -> Tensor:
        ...

    def op(*tensors: Tensor) -> Tuple[Tensor]:
        ...

The number and shapes of input and output tensors match the signature of the elementary operation specified
in the einx expression (i.e. containing all non-vectorized axes). For example:

..  code-block:: python

    @einx.{framework}.adapt_with_vmap
    def einop(x, y):
        # shape of x is (b, c)
        # shape of y is (c)
        ...
        return ... # shape of result must be (c, b)

    z = einop("a [b c] x, d [c] -> a d x [c b]", x, y)

Args:
    op: The operation that will be adapted to einx notation.

Returns:
    A new operation that follows einx notation and internally invokes the original operation.
"""

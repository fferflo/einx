How does einx handle input and output tensors?
##############################################

einx functions accept a description string that specifies Einstein expressions for the input and output tensors. The expressions potentially
contain nested compositions and concatenations that prevent the backend functions from directly accessing the required axes. To resolve this, einx
first flattens the input tensors in each operation such that they contain only a flat list of axes. After the backend operation is applied, the
resulting tensors are unflattened to match the requested output expressions.

Compositions are flattened by applying a `reshape` operation:

..  code::

    einx.rearrange("(a b) -> a b", x, a=10, b=20)
    # same as
    np.reshape(x, (10, 20))

Concatenations are flattened by splitting the input tensor into multiple tensors along the concatenated axis:

..  code::

    einx.rearrange("(a + b) -> a, b", x, a=10, b=20)
    # same as
    np.split(x, [10], axis=0)

Using a concatenated tensor as input performs the same operation as passing the split tensors as separate inputs to the operation. einx handles
expressions with multiple nested compositions and concatenations gracefully.

After the operation is applied to the flattened tensors, the results are reshaped and concatenated and missing axes are inserted and broadcasted
to match the requested output expressions.

When multiple input and output tensors are specified, einx tries to find a valid assignment between inputs and outputs for the given axis names. This
can sometimes lead to ambiguous assignments:

..  code::

    # Broadcast and stack x and y along the last axis. x or y first?
    einx.rearrange("a, b -> a b (1 + 1)", x, y)

To find an assignment, einx iterates over the outputs in the order they appear in the operation string, and for each output tries to find the first input
expression that allows for a successful assignment. In most cases, this leads to input and output expressions being assigned in the same order:

..  code::

    einx.rearrange("a, b -> a b (1 + 1)", x, y)
    # same as
    np.stack([x, y], axis=-1)

The function `einx.rearrange` can be used to perform flattening and unflattening of the input tensors as described in the operation string. Other functions
such as `einx.reduce` and `einx.dot` perform the same flattening and unflattening, in addition to applying some operation to the flattened tensors.

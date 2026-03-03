Named tensor notation
#####################

Overview
========

Named tensor notation proposes to annotate tensor objects with symbolic axis names, resulting in so-called *named tensors*. Unlike notations for tensor operations such as Numpy-like or ein* notations,
named tensor notation addresses the representation of tensor objects themselves. While tensor dimensions in classical tensor objects
are identified by their position in the shape (e.g., the 1st, 2nd or 3rd dimension), dimensions in named tensors are identified by their symbolic names
(e.g., the *batch*, *feature* or *time* dimension). The following pseudo-code illustrates the difference between classical positional tensors and named tensors:

..  code-block:: python

    # Classical tensor
    x = create_tensor((32, 128, 128))
    y = sum(x, axis=2)

    # Named tensor
    x = create_tensor({"batch": 32, "feature": 128, "time": 128})
    y = sum(x, axis="time")

Examples of frameworks that follow named tensor notation are `Haliax <https://github.com/marin-community/haliax>`_,
`Penzai <https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html>`_ and `functorch.dim <https://github.com/pytorch/pytorch/blob/main/functorch/dim/README.md>`_.
Named tensor notation has also been discussed more extensively in `Tensor Considered Harmful <https://nlp.seas.harvard.edu/NamedTensor.html>`_ and
`Named Tensor Notation <https://arxiv.org/pdf/2102.13196>`_.

Importantly, einx notation is compatible both with named tensors and classical positional tensors. An einx operation may accept named tensors akin to classical positional tensors
by matching the string expression against the axis names rather than or in addition to the axis positions:

..  code-block:: python

    # Classical tensor: Match positional shape of x with "batch feature time" expression
    y = einx.sum("batch feature [time]", x)

    # Named tensor: Match axis names of x with "batch feature time" expression
    y = einx.sum("batch feature [time]", x)

Several advantages and disadvantages of using named tensors, such as persistent axis names, may thus be similarly leveraged in einx notation.

In the following, we consider characteristics of named tensor notation, their pros and cons, and how they relate to einx notation.

Human-readable axis names
=========================

**Pro: Self-documenting.** Human-readable axis names make tensor code more self-documenting and interpretable to the user. In contrast, classical positional tensor code is often annotated
with comments to clarify axis semantics, or requires the user to rely on conventions (*e.g.*, "channels-last layout") or trace through earlier parts of the code:

..  code-block:: python

    # Classical tensor: Requires comments, conventions, or checking prior code to understand
    x = create_tensor((32, 128, 128)) # (batch, feature, time)
    y = sum(x, axis=2) # sum along time axis

    # Named tensor
    x = create_tensor({"batch": 32, "feature": 128, "time": 128})
    y = sum(x, axis="time")

Human-readable axis names are available among others through the use of named tensors, or through einx notation (or other ein*-notations) which use a string of axis names to
define operations. Besides strings, axis symbols may also be represented by special Axis/Dimension objects in Python.

Persistent axis names
=====================

Annotating tensor objects themselves with symbolic axis names allows the names to persist across subsequent operations and enforce a consistent usage of axes.
Matching axis labels with einx expressions may similarly be used to persist axis names across einx operations.

**Pro: Enforce consistent usage.** Persistent axis names enforce consistent usage of axes across operations and help avoid some shape errors that may arise from axis misalignment.
For example, a channel-first and channel-last tensor layout may be confused in classical tensor code, resulting in silent failures due to axis misalignment:

..  code-block:: python

    # Classical tensor
    def sum_along_channel(x):
        # x: (batch, channel, height, width) - channels-first!
        return x.sum(axis=1)

    x = create_tensor((32, 128, 128, 3)) # (batch, height, width, channel) - channels-last!
    y = sum_along_channel(x) # silent failure due to axis misalignment

Such errors are avoided in named tensor code, since axes are identified by their symbolic names rather than their positional index:

..  code-block:: python

    # Named tensor
    def sum_along_channel(x):
        return x.sum("channel")

    x = create_tensor({"batch": 32, "height": 128, "width": 128, "channel": 3})
    y = sum_along_channel(x)

**Con: Verbose renaming.** Named tensors may require frequent (re)naming of axes and result in more verbose code than positional-style tensors. For example, consider a simple neural
net that consists among others of subsequent fully-connected layers:

..  code-block:: python

    def net(x, weights):
        for i in range(10):
            x = norm(x)
            x = fully_connected(x, weights[i])
            x = relu(x)
        return x

In named tensor code, the channel dimension is always identified by some ``"channel"`` name, which might require additional renaming to align the output of one layer with
with the expected input of the next layer:

..  code-block:: python

    # Named tensor
    def fully_connected(x, weight):
        # x has a "channel" axis
        # weight has "channel_in" and "channel_out" axes

        # Contract "channel" of x with "channel_in" of weight
        x = dot(x, weight, axis=("channel", "channel_in")) 

        # x now has "channel_out" axis, but the next layer expects a "channel" axis,
        # so we need to rename it:
        x = x.rename({"channel_out": "channel"})

        return x

In contrast, in positional tensor code, the channel dimension is identified by convention (*i.e.* the last dimension of the tensor), and no additional renaming is required:

..  code-block:: python

    # Classical tensor
    def fully_connected(x, weight):
        # channel axis is last in x
        x = einx.dot("... [channel_in], [channel_in] channel_out -> ... channel_out", x, weight)
        # channel axis is still last in x
        return x

.. _hidden-axes-when-implementing-operations:

Hiding axes when implementing operations
========================================

Implementing an operation with named tensors allows ignoring all vectorized dimensions from the function definition,
and focusing only on the axes that are relevant to the operation.

For example, consider the following pseudo-code implementation of a dot-product attention operation:

..  code-block:: python

    def attention(query, key, value):
        # query has (at least) dimensions {"channel_in"}
        # key has (at least) dimensions {"key", "channel_in"}
        # value has (at least) dimensions {"key"}
        weights = dot(query, key, axis="channel_in")
        weights = softmax(weights, axis="key")
        return dot(weights, value, axis="key")

The operation may be invoked on higher-dimensional tensors, but does not need to know about any additional dimensions since they are implicitly vectorized over:

..  code-block:: python

    # query: {"batch", "query", "channel_in"}
    # key: {"batch", "key", "channel_in"}
    # value: {"batch", "key", "channel_out"}

    output = attention(query, key, value)

    # output: {"batch", "query", "channel_out"}

Concretely: ``attention`` does not need to know about the ``"batch"``, ``"query"``, and ``"channel_out"`` dimensions.

The ability to hide vectorized dimensions from an inner operation, however, is not unique to named tensor notation. For example,
vmap notation forwards only non-vectorized axes to an inner operation, and thereby also allows hiding vectorized dimensions. This may
be utilized, *e.g.*, in einx with the ``adapt_with_vmap`` adapter:

..  code-block:: python

    @einx.jax.adapt_with_vmap
    def attention(query, key, value):
        # Vectorized dimensions are hidden here. Tensors only have "channel" and "key" dimensions
        weights = einx.dot("[channel], key [channel] -> key", query, key)
        weights = einx.softmax("[key]", weights)
        return einx.dot("[key], [key] ->", weights, value)

    # query: (batch, query, channel_in)
    # key: (batch, key, channel_in)
    # value: (batch, key, channel_out)

    output = attention("batch query [channel_in], batch [key channel_in], batch [key] channel_out -> batch query channel_out", query, key, value)

    # output: (batch, query, channel_out)

**Pro: More concise.** Hiding vectorized dimensions in function definitions leads to more concise tensor code.
In the above examples, the implementation of ``attention`` is simpler than if all additional vectorized dimensions were specified explicitly.

**Pro: Flexible vectorization.** Hiding vectorized dimensions allows the same inner operation to be vectorized along different dimensions of the argument tensors without
requiring changes to the operation itself.

**Con: Undefined axis ordering.** Hiding vectorized dimensions also removes the ability to explicitly specify the order of axes, which may among others impact performance.
In the above examples, the order of axes in the intermediate tensors (e.g., ``weights``) is implicit and not defined by the user.

Hiding axes when calling operations
===================================

Calling an operation with named tensors handles all vectorized axes implicitly and only requires the user to specify axes that the operation is applied along.
This is done by (1) vectorizing all axes that are not explicitly specified in an operation, and
(2) aligning dimensions of multiple tensors using their symbolic names. For example:

..  code-block:: python

    # x has dimensions {"a", "b"}
    # y has dimensions {"b"}
    z = x + y
    # z has dimensions {"a", "b"}

..  code-block:: python

    # x has dimensions {"a", "b"}
    # y has dimensions {"b", "c"}
    z = dot(x, y, axis="b")
    # z has dimensions {"a", "c"}

**Pro: More concise.** Hiding vectorized dimensions leads to more concise expressions when invoking operations. For example,
the line ``z = x + y`` represents *all* possible vectorizations of the scalar addition.

**Con: Less self-documenting.** Hiding vectorized dimensions leads to less self-documenting code, especially if used in sequences of multiple operations.
In these cases, the user often has to trace through earlier parts of the code to understand which axes a given tensor is defined with.

For example, consider the following simple implementation of a dot-product attention operation:

..  code-block:: python

    def attention(query, key, value):
        weights = dot(query, key, axis="channel_in")
        weights = softmax(weights, axis="key")
        return dot(weights, value, axis="key")

The axes that the input, intermediate and output tensors are defined with are not explicitly documented in the code.
We could spell out the questions that arise when reading this function definition as follows:

..  code-block:: python

    def attention(query, key, value):
        # What axes are available in query, key, and value here?
        weights = dot(query, key, axis="channel_in")
        # What axes are available in weights here? Not 'channel_in' since it was removed just above,
        # but which axes are left?
        weights = softmax(weights, axis="key")
        # Okay, weights has at least a 'key' axis, but any other axes that might be relevant here?
        return dot(weights, value, axis="key")
        # What axes are returned from the operation? Not 'key' since it was removed just above,
        # but which axes are left? We have to trace at least through the previous two operations
        # to find out.

In contrast, einx notation fully specifies the axes of all tensors inside a given context,
while still allowing to hide additional vectorized dimensions (such as ``"batch"``) that are added outside of this context (see :ref:`hidden-axes-when-implementing-operations`):

..  code-block:: python

    @einx.jax.adapt_with_vmap
    def attention(query, key, value):
        # This tells us: query just has a 'channel' axis, key has 'channel' and 'key' axes,
        # and weights is defined along only a 'key' axis:
        weights = einx.dot("[channel], key [channel] -> key", query, key)
        # This also tells us immediately which axes are available in weights:
        weights = einx.softmax("[key]", weights)
        # And this tells us that the operation returns a scalar as output
        return einx.dot("[key], [key] ->", weights, value)

**Con: Undefined axis ordering.** Hiding vectorized dimensions also removes the ability to explicitly specify the order of axes, which may among others impact performance. For example:

..  code-block:: python

    # x has dimensions {"width"}
    # y has dimensions {"height"}
    z = x + y
    # z has dimensions {"width", "height"},
    # but is it represented in memory in a layout (width, height) or (height, width)?

In contrast, einx notation explicitly specifies the axis ordering:

..  code-block:: python

    z = einx.add("width, height -> height width", x, y)

**Con: Silent failures.** Typos may result in axis misalignment and silent shape errors when using strings as axis symbols, since vectorized axes are not specified explicitly. For example:

..  code-block:: python

    # x has shape {"tokens"}
    # y has shape {"token"}
    z = x + y
    # z has shape {"tokens", "token"}, but we might have expected {"tokens"} or {"token"}

Such exceptions are avoided in einx notation:

..  code-block:: python

    z = einx.add("tokens, token -> token", x, y)  # raises an error
    z = einx.add("tokens, token -> tokens", x, y) # raises an error


Compatibility with Python ecosystem
===================================

**Con:** Most major tensor libraries such as Numpy, PyTorch, Jax, Tensorflow, Scipy, and OpenCV primarily or only support positional-style tensors.

To use functions from these libraries with named tensors, one of two options has to be followed:

*   The operation is adapted by the user to support named tensors. This has to be done for each operation in a given library, as well as for each new operation that might be added
    down the line.
*   The named tensor is converted to a positional tensor before the operation, and then converted back to a named tensor after the operation. This leads to more verbose, less readable code.

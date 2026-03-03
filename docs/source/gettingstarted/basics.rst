Basic tutorial
##############

.. _tutorial-loop-notation:

Overview
********

einx is a Python library that provides a universal interface to formulate tensor operations in frameworks such as Numpy, PyTorch, Jax and Tensorflow.

Every operation in einx is expressed according to the following, general pattern:

..  code-block:: python

    outputs... = einx.{elementary_operation}("{vectorization}", inputs...)

The operation ``einx.{elementary_operation}`` accepts one or more input tensors ``inputs...`` and returns one or more output tensors ``outputs...``.
It consists of

1.  an elementary operation ``{elementary_operation}`` that is applied to sub-tensors of the inputs and outputs, and
2.  a vectorization string ``"{vectorization}"`` that describes how the elementary operation is vectorized across the dimensions of the input and output tensors,
    *i.e.* which sub-tensors it is applied to.

The operation ``einx.{elementary_operation}`` may be chosen from a set of built-in operations in the einx API (*e.g.*, ``einx.sum``, ``einx.dot``, ``einx.add``),
or by adapting custom operations to einx notation (see :ref:`Supported operations <tutorial-supported-operations>` below).

The notation that is used for constructing the vectorization string is independent of any particular operation that is used with: The same notational rules apply across
all operations. To illustrate these rules, we will consider the following example tensor operation:

..  code-block:: python

    z = einx.{elementary_operation}("[c d] a, b -> a [e] b", x, y)

The operation is invoked with input tensors ``x`` and ``y``, and returns an output tensor ``z``.
The vectorization string ``"[c d] a, b -> a [e] b"`` describes the tensor dimensions and vectorization of the elementary operation, and serves the following three purposes:

1.  **Signature of the full operation:** The full expression defines the signature of the *full* operation ``einx.{elementary_operation}``.
    Inputs and outputs in the expression are delimited with an arrow (``->``), tensors on each side are delimited with arrows (``,``), and names indicate tensor dimensions:

    ..  code-block:: python

        General pattern: "{input1}, {input2}, ... -> {output1}, {output2}, ..."
        Above example:   "[c d] a,  b             -> a [e] b"

    This indicates that the operation accepts two input tensors with shapes (c, d, a) and (b), and returns an output tensor with shape (a, e, b).

2.  **Signature of the elementary operation:** The sub-expressions in brackets define the signature of the *elementary* operation ``{elementary_operation}``.

    ..  code-block:: python

        General pattern: "[{input1}], [{input2}], ... -> [{output1}], [{output2}], ..."
        Above example:   "[c d],      []              -> [e]"

    This indicates that the elementary operation accepts two input tensors with shapes (c, d) and (), and returns an output tensor with shape (e).

3.  **Vectorization:** All axes that are not part of the elementary operation's signature are used to define its vectorization. The definition is done by analogy
    with loop notation: We may map any expression in einx notation to an equivalent expression in loop notation to understand what the output of the operation will be.

    The above example corresponds to the loop expression

    ..  code-block:: python

        for a in range(...):
            for b in range(...):
                z[a, :, b] = {elementary_operation}(x[:, :, a], y[b])

    and is constructed according to the following rules:

    a.  Write one for-loop for each of the vectorized axes: ``a`` and ``b``. The loop ranges are determined implicitly by the lengths of the respective tensor dimensions.
    b.  Use the loop indices to extract sub-tensors from the inputs and outputs. Use ``:`` for axes that are part of the elementary operation's signature (*i.e.* marked with brackets):

        *   Use ``x[:, :, a]`` for the expression ``"[c d] a"``.
        *   Use ``y[b]`` for the expression ``"b"``.
        *   Use ``z[a, :, b]`` for the expression ``"a [e] b"``.

    c.  Invoke the elementary operation ``{elementary_operation}`` on these sub-tensors.

    The loop expression is used only by analogy to define what the output of an einx operation is. einx internally calls backend functions from the respective tensor
    frameworks to efficiently compute the result,
    rather than literally invoking ``{elementary_operation}`` in Python loops (see :ref:`the implementation section <tutorial-compiled-code>` below for details).


Example: Matrix multiplication
******************************

As a more conrete example, we consider the matrix multiplication operation in einx notation:

..  code-block:: python

    z = einx.dot("a [b], [b] c -> a c", x, y)

We follow the three steps described above to understand the meaning of this operation:

1.  The full expression defines the signature of the operation ``einx.dot``: It accepts inputs with shapes (a, b), (b, c), and returns an output with shape (a, c).

2.  The sub-expressions in brackets ``[b], [b] -> []`` define the signature of the elementary operation ``dot``: It accepts two equally sized vectors, and returns a scalar.

3.  The axes not marked with brackets define the vectorization of ``dot`` by analogy with loop notation:

    ..  code-block:: python

        for a in range(...):
            for c in range(...):
                z[a, c] = dot(x[a, :], y[:, c])

Example: Vectorized scalar operations
*************************************

Scalar elementary operations accept and return scalar values, *i.e.* zero-dimensional tensors. For example, the following operation computes the element-wise addition of two matrices:

..  code-block:: python

    z = einx.add("a b, a b -> a b", x, y)

The lack of brackets indicates that the elementary operation is indeed applied to zero-dimensional values. The vectorization
is define by analogy with loop notation as follows:

..  code-block:: python

    for a in range(...):
        for b in range(...):
            z[a, b] = x[a, b] + y[a, b]

The order of the axes may be changed to represent different vectorizations of the same scalar addition. For example:

..  code-block:: python

    z = einx.add("a b, b a -> a b", x, y)
    z = einx.add("a b, a b -> b a", x, y)

More generally, tensors must not necessarily have the same number of dimensions or use the same set of loop indices. For instance, all of the following are valid:

..  code-block:: python

    # Loop notation
    for a in range(...): for b in range(...): for c in range(...):
        z[a, b, c] = x[a, b] + y[b, c]

    # einx notation
    z = einx.add("a b, b c -> a b c", x, y)

..  code-block:: python

    # Example: Outer product between x and y

    # Loop notation
    for a in range(...): for b in range(...):
        z[a, b] = x[a] * y[b]

    # einx notation
    z = einx.multiply("a, b -> a b", x, y)

..  code-block:: python

    # Example: Scalar (i.e. 0-dimensional) arguments

    # Loop notation
    for a in range(...): for b in range(...):
        z[a, b] = x[a, b] + y

    # einx notation
    z = einx.add("a b, -> a b", x, y)

..  note::

    The usage of a subset of the available loop indices for a given tensor corresponds to
    `broadcasting <https://numpy.org/doc/stable/user/basics.broadcasting.html>`__ in Numpy and other frameworks.

    ..  code-block:: python

        # einx notation
        z = einx.add("a b, a -> a b", x, y)

        # Numpy notation follows broadcasting rules
        z = x + y[:, np.newaxis]

Example: Sum reduction
**********************

The elementary sum-reduction operation accepts a single tensor and returns the sum over all its values as a scalar output:

..  code-block:: python

    y = einx.sum("a b [c] -> a b", x)

..  note::

    The above operation is expressed in Numpy notation as follows:

    ..  code-block:: python

        # Numpy notation
        y = np.sum(x, axis=2)

In general, brackets must be placed around the number of axes in an operation that matches the signature of the elementary operation. For example, all of the following are valid

..  code-block:: python

    y = einx.sum("[a] b [c] -> b", x)
    y = einx.sum("[a] [c] b -> b", x)
    y = einx.sum("[a c] b -> b", x)

If the bracket placement does not match the signature of the elementary operation, an exception is raised:

..  code-block:: python

    # Raises an exception: sum returns a scalar, but expression indicates a vector
    y = einx.sum("a b [c] -> a b [c]", x)



.. _tutorial-supported-operations:

Supported operations
********************

einx's API contains many common tensor operations as functions in the namespace ``einx.*``. Additionally, einx provides
options to adapt custom, new operations that are not part of its built-in API to be callable with einx notation.


API
===

The following is an incomplete list of tensor operations in einx's API. For a complete reference, see :doc:`the API documentation </api/operations>`.

*   Identity map (``einx.id``): The elementary operation accepts a scalar (or tuple of scalars) and returns it unchanged.

    .. code-block:: python

        y = einx.id("a b -> b a", x)

    ``einx.id`` may be used, *e.g.*, to rearrange axes of a tensor without changing its values.

*   Reduction functions (*e.g.*, ``einx.{sum|mean|max|count_nonzero}``): The elementary operation accepts a single tensor and returns a scalar.

    .. code-block:: python

        y = einx.sum("a [b] -> a", x)

*   Scalar functions (*e.g.*, ``einx.{add|multiply|divide|maximum|where}``): The elementary operation accepts two or more scalars and returns a single scalar.

    .. code-block:: python

        z = einx.add("a b, b a -> a b", x, y)
        z = einx.where("a b, a, b -> a b", cond, x, y)

*   Dot-product (``einx.dot``): The elementary operation accepts two vectors and returns their dot-product as a scalar:

    .. code-block:: python

        z = einx.dot("a [b], [b] c -> a c", x, y)

    If more than one contraction axis or more than two input tensors are provided, the dot-product is applied sequentially in an
    unspecified order to all pairs of contracted axes with the same name.

*   Gather (``einx.get_at``): The elementary operation accepts a value tensor and a coordinate vector of integers,
    and returns the single scalar value from the value tensor at the specified coordinates.

    .. code-block:: python

        y = einx.get_at("b [h w] c, b p [2] -> b p c", image, indices)

*   Shape-preserving operations(*e.g.*, ``einx.{softmax|sort|argsort|flip``): The elementary operation accepts a tensor and returns a tensor of the same shape.

    .. code-block:: python

        y = einx.softmax("a b [c] -> a b [c]", x)

Non-API
=======

einx provides options to adapt new, custom operations that are not part of its built-in API such that they can be invoked using einx notation.

To adapt a new, custom operation to einx notation, we have to know how vectorization is expressed in the interface of that custom operation.
This allows einx to convert its own representation of vectorization (*i.e.* einx notation) to the representation of vectorization that is expected
by the custom operation.

einx provides different adapters for different types of vectorization interfaces. For a complete list of adapters, see :doc:`the API documentation </api/adapters>`.
In the following, we consider examples of adapting Numpy-like reduction operations and vmappable operations.

**Adapting Numpy-like reduction operations.** Reduction operations in Numpy-like notation express their inherent vectorization using the ``axis`` argument:

..  code-block:: python

    # Numpy-like notation
    y = np.sum(x, axis=1)

    # einx notation
    y = einx.sum("a [b] -> a", x)

einx provides a dedicated adapter for adapting reduction operations in Numpy-like notation that allows invoking these operations using einx notation:

..  code-block:: python

    # Define some custom reduction operation with a Numpy-like interface
    def myfunc(x, axis):
        # x is an array, and we should reduce along the 'axis' dimension
        return 0.5 * np.sum(x ** 2, axis=axis)

    # Adapt the operation to be callable with einx notation
    einmyfunc = einx.numpy.adapt_numpylike_reduce(myfunc)

    # Invoke with einx notation
    y = einmyfunc("a [b] c -> c a", x)

**Adapting vmappable operations.** While adapters such as the above cover particular vectorization interfaces,
they do not represent a universal interface for adapting arbitrary custom operations to einx notation.
Such a universal adapter requires a universal mechanism in the underlying framework to express vectorization of arbitrary operations.

One such universal mechanism is the `vmap transformation <https://docs.jax.dev/en/latest/automatic-vectorization.html>`__ that was first introduced in the Jax framework.
vmap (*i.e.* vectorizing map) allows vectorizing arbitrary operations along single dimensions of input and output tensors. einx provides an adapter
that utilizes vmap to adapt arbitrary custom operations to einx notation. This adapter is only available with frameworks
that support vmap (*e.g.*, Jax and PyTorch, but not Numpy):

..  code-block:: python

    # Define some custom elementary operation
    def myfunc(x):
        # Shape of x contains only the non-vectorized dimensions
        return 0.5 * jnp.sum(x ** 2)

    # Adapt the operation to be callable with einx notation
    einmyfunc = einx.jax.adapt_with_vmap(myfunc)

    # Invoke with einx notation
    z = einmyfunc("a [b] c -> c a", x, y)

The ``adapt_with_vmap`` adapter not only allows adapting arbitrary operations, but also represents a more natural interface
than the specialized adapters described above by forwarding only the non-vectorized axes to the custom operation. For example, when invoking a custom operation with

..  code-block:: python

    y = einmyfunc("a [b c] d -> d a", x)

the vmap-adapter forwards only the non-vectorized dimensions of the inputs to the custom operation

..  code-block:: python

    @einx.jax.adapt_with_vmap
    def einmyfunc(x):
        # Shape of x is (b, c)
        # We don't need to worry about vectorized axes here
        return ...

while the reduction adapter forwards all dimensions and uses the ``axis`` argument to express vectorization:

..  code-block:: python

    @einx.numpy.adapt_numpylike_reduce
    def einmyfunc(x, axis):
        # Shape of x is (a, b, c, d)
        # axis is (1, 2)
        # We have to handle vectorized axes here manually
        return ...

As another example, builtin functions in a vmap-supporting tensor framework for which no operation is provided in ``einx.{operation}`` may be adapted to einx notation by using the vmap-adapter:

..  code-block:: python

    einsolve = einx.jax.adapt_with_vmap(jnp.linalg.solve)
    y = einsolve("a [n n] b, c [n] -> a c [n] b", A, x)

    eindet = einx.jax.adapt_with_vmap(jnp.linalg.det)
    y = eindet("a [n n] -> a", x)

    eineig = einx.jax.adapt_with_vmap(jnp.linalg.eig)
    vals, vecs = eineig("a [n n] b -> a [n] b, a [n n] b", x)



.. _tutorial-compiled-code:

Implementation
**************

The implementation of einx operations using literal ``for``-loops in Python would be highly inefficient. The analogy with loop notation is therefore only
used to define *what* the results of an einx operation will be, but now *how* the operation is implemented on a given backend.

Instead of providing its own low-level function implementations, einx just-in-time compiles operations to function calls in a given backend. For instance,
if an operation is executed with Numpy tensor objects, einx will generate a Python code snippet that imports functions from the Numpy 
namespace and executes them to perform the desired operation.

The Python code snippet that einx generates for a given operation can be inspected by passing ``graph=True`` to the function call. For example:

..  code-block:: python

    >>> x = np.random.rand(3)
    >>> y = np.random.rand(4)
    >>> z = einx.add("a, b -> a b", x, y)
    >>> code = einx.add("a, b -> a b", x, y, graph=True)
    >>> print(code)
    import numpy as np
    def op(a, b):
        a = np.reshape(a, (3, 1))
        b = np.reshape(b, (1, 4))
        c = np.add(a, b)
        return c

The Python function is compiled and cached the first time an einx operation is executed, and reused on subsequent calls with the same signature.
This results in no overhead compared to calling the framework functions directly, other than for initialization and during cache look-up. If used with framework-specific just-in-time compilation such as `jax.jit <https://docs.jax.dev/en/latest/jit-compilation.html>`__, the framework traces only
through its own function calls, and the einx footprint therefore disappears entirely.

einx provides different backends for compiling operations to different kinds of code snippets. These backends allow among others compiling operations to classical Numpy-like notation,
vmap-based notation, or einsum notation (if an operation can be expressed with einsum). See :doc:`this page </gettingstarted/backends>` for more information.

We have covered the definition of einx notation by analogy with loop notation, as well as its implementation in the einx library.
On the next page, we will consider more advanced features of einx notation, such as axis compositions, ellipses and implicit outputs.
Advanced tutorial
#################

Axis constraints
****************

einx tries to infer the lengths of all axes by matching the input expressions against the actual shapes of the input tensors.
In some cases, however, the input shapes do not provide sufficient constraints to infer all axis lengths. This is the case, *e.g.*, when broadcasting
tensors along new output axes:

..  code-block:: python

    # This raises an exception: Too few constraints
    y = einx.id("a -> a b", x)

To address cases like this, additional axis constraints may be passed as keyword arguments to einx functions. The keyword argument name
specifies the axis name, and the value specifies the length:

..  code-block:: python

    # This works
    y = einx.id("a -> a b", x, b=42)

Additional constraints may be given for any axis in an expression, even those that are already constrained by the inputs shapes:

..  code-block:: python

    x = np.random.randn(10, 20)

    # This works
    y = einx.id("a b -> b a", x, a=10, b=20)

    # This raises an exception: Conflicting constraints
    y = einx.id("a b -> b a", x, a=10, b=21)



Numerical axes
**************

For convenience, axes with a numerical name may be used to specify their length inline. Numerical
axes are equivalent to introducing a new, unique axis name with a corresponding constraint:

..  code-block:: python

    y = einx.id("a b -> a b 3", x)
    # same as
    y = einx.id("a b -> a b c", x, c=3)

Multiple numerical axes with the same name refer to *different* axes:

..  code-block:: python

    y = einx.id("a b -> a b 3 3", x)
    # same as
    y = einx.id("a b -> a b c d", x, c=3, d=3)

This may lead to unexpected results in some cases:

..  code-block:: python

    y = einx.id("a b 3 -> b a 3", x)
    # same as
    y = einx.id("a b c -> b a d", x, c=3, d=3)

The above operation may appear like a mere permutation of axes. However, the input ``3`` axis does not appear in the output!
This is an invalid expression for the ``einx.id`` function and raises an exception.



Axis squeezing
**************

Any vectorized axis with a length of 1 may be removed (*i.e.* "squeezed") from an expression:

..  code-block:: python

    x = np.random.randn(10, 1, 20)

    # This works: axis is specified with length 1
    y = einx.id("a 1 c -> a c", x)

    # This works: axis b has length 1 when matching with x
    y = einx.id("a b c -> a c", x)

    # This does not work: axis a is not squeezable
    y = einx.id("a b c -> b c", x)

In loop notation, the corresponding loop is omitted and an index of ``0`` is used for the index variable:

..  code-block:: python

    # einx notation
    y = einx.id("a 1 c -> a c", x)

    # Loop notation
    for a in range(...): for c in range(...):
        y[a, c] = x[a, 0, c]




Implicit outputs
****************

Some einx functions allow omitting the arrow and output expression and infer the output from the input expression instead.
This behavior is indicated in a function's documentation. Implicit outputs are supported among others in the following cases:

*   Functions that do not change the dimensionality of a single input determine the output by replicating the input expression:

    ..  code-block:: python

        y = einx.softmax("a b [c]", x)
        # same as
        y = einx.softmax("a b [c] -> a b [c]", x)

*   Reduction functions determine the output by removing all brackets from the input:

    ..  code-block:: python

        y = einx.sum("a b [c]", x)
        # same as
        y = einx.sum("a b [c] -> a b", x)

*   Arg-operations determine the output by replacing brackets from the input with a single new axis in brackets:

    ..  code-block:: python

        y = einx.argmax("b [h w] c", x)
        # same as
        y = einx.argmax("b [h w] c -> b [2] c", x)

*   Element-wise operations determine the output by choosing one of the input expressions if it contains the axis names of all
    other inputs and if this choice is unique:

    ..  code-block:: python

        y = einx.add("a b c, b c", x, z)
        # same as
        y = einx.add("a b c, b c -> a b c", x, z)

        y = einx.add("a, b", x, z)
        # Raises exception: Cannot determine output expression

        y = einx.add("a b, b a", x, z)
        # Raises exception: Cannot determine output expression



.. _tutorial-axis-compositions:

Axis compositions
*****************

An axis composition refers to any way in which one or more axes may be composed to form a new, "composed" axis. In loop notation,
this corresponds to mapping one or more loop variables to a new index value (*e.g.*, via arithmetic operations) which is then
used to index the respective tensor dimension.

einx supports two types of axis compositions: Flattened axes and concatenated axes.

Flattened axes
==============

A **flattened axis** is defined as multiple axes of a single tensor that are flattened in
`row-major order <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`__. For example,
a flattened axis consisting of the sub-axes ``a`` and ``b`` corresponds ``a`` groups of ``b``
elements stacked along the single, composed axis.

A flattened axis is specified in the einx expression by wrapping the sub-axes in parentheses. For instance, the expression ``a (b c)`` matches a
two-dimensional tensor where the second dimension consists of two axes (``b`` and ``c``) flattened in row-major order.

A flattened axis may be used in an einx operation to flatten (*i.e.* compose) or unflatten (*i.e.* decompose) axes:

..  code-block:: python

    x = np.random.randn(3, 4, 5)

    y = einx.id("a b c -> a (b c)", x)
    # composes axes b and c into a single new axis (b c)

    # y has shape (3, 4 * 5)

    z = einx.id("a (b c) -> a b c", y, b=4)
    # decomposes axis (b c) into separate axes b and c

    # z has shape (3, 4, 5)

When decomposing a flattened axis, the input tensor shape only constrains the length of the flattened axis, but not the lengths of the sub-axes.
Therefore, additional axis constraints have to be provided to fully specify all axis lengths. In the above case, the length of axis ``b`` is specified as a keyword constraint,
and the length of axis ``c`` is then solved for internally using the other constraints.

In loop notation, a flattened axis corresponds to applying the
`row-major formula <https://en.wikipedia.org/wiki/Row-_and_column-major_order#Address_calculation_in_general>`__ to the corresponding loop variables:

..  code-block:: python

    # einx notation
    y = einx.id("a b c -> a (b c)", x)

    # Loop notation
    for a in range(...): for b in range(...): for c in range(...):
        y[a, b * length_c + c] = x[a, b, c]

..  code-block:: python

    # einx notation
    y = einx.id("a (b c) -> a b c", x, b=4)

    # Loop notation
    for a in range(...): for b in range(...): for c in range(...):
        y[a, b, c] = x[a, b * length_c + c]

However, it is often easier to think of a flattened axis simply in terms of groups of elements stacked along a single axis rather than in terms of the index formula.

Flattening and unflattening of axes is implemented in einx by applying a reshape operation to the tensor.
This can be verified by inspecting the Python function generated for the above expressions:

..  code-block:: python

    >>> x = np.random.randn(3, 4, 5)
    >>> print(einx.id("a b c -> a (b c)", x, graph=True))
    import numpy as np
    def op(a):
        a = np.reshape(a, (3, 20))
        return a

..  code-block:: python

    >>> x = np.random.randn(3, 20)
    >>> print(einx.id("a (b c) -> a b c", x, b=4, graph=True))
    import numpy as np
    def op(a):
        a = np.reshape(a, (3, 4, 5))
        return a

..  note::

    `einops <https://einops.rocks/>`__ first introduced the idea to represent flattened axes with parentheses in an expression. It refers to flattened axes as "axis compositions", while we
    use the term "axis composition" to denote the general concept of composing axes, and "flattened axis" to denote the specific case of row-major ordering. See
    :doc:`this page </comparison/ein>` for a comparison of einx with einops notation, and `the einops tutorial <https://einops.rocks/1-einops-basics/#composition-of-axes>`__
    for visual examples of axis (un)flattening.

Concatenated axes
=================

A **concatenated axis** is defined as multiple axes of multiple tensors that are concatenated along a single new axis. It is represented
in the einx expression by using the plus operator ``+`` between the sub-axes and wrapping them in parentheses. For instance, the expression ``(a + b)``
represents a new axis formed by concatenating axis ``a`` of the first tensor and axis ``b`` of the second tensor.

A concatenated axis may be used in an einx operation to concatenate (*i.e.* compose) or split (*i.e.* decompose) axes:

..  code-block:: python

    x = np.random.randn(3, 4)
    y = np.random.randn(5, 4)

    z = einx.id("a c, b c -> (a + b) c", x, y)
    # concatenates axes a and b into a single new axis (a + b)

    # z has shape (3 + 5, 4)

    w1, w2 = einx.id("(a + b) c -> a c, b c", z, a=3)
    # splits axis (a + b) into separate axes a and b

    # w1 has shape (3, 4)
    # w2 has shape (5, 4)

When splitting the axis as above, we have to specify an additional axis constraint to fully determine the lengths of all axes. Since concatenated axes change the number
of tensors in an operation, they are currently only supported in the ``einx.id`` function which does not rely on the order or number of inputs and outputs.

Nesting
=======

Axis compositions may be nested both with each other and with brackets (and with ellipses, see below) to form more complex expressions. All of the following are valid:

..  code-block:: python

    # Compute the sum of groups of subsequent values along the second axis
    y = einx.sum("a (b [c]) -> a b", x, c=4)

    # Compute the sum of values at even and odd positions along the second axis
    y = einx.sum("a ([b] c) -> a c", x, c=2)

    # Mean-pooling of an image with 4x4 patches (if evenly divisible)
    y = einx.mean("(h [dh]) (w [dw]) c -> h w c", x, dh=4, dw=4)

    # Flatten the spatial dimensions of an image and prepend a vector (e.g. class token)
    z = einx.id("c, h w c -> (1 + (h w)) c", vec, image)



.. _tutorial-ellipsis:

Ellipses
********

An ellipsis (``...``) may be used in an einx expression to represent multiple axes of a tensor jointly. The ellipsis is placed
immediately after a sub-expression to indicate that this sub-expression is repeated zero or more times. The number of repetitions
is inferred from the input tensor shapes and additional constraints.

For instance, in the following example the axis ``s`` is repeated such that the overall expression matches the dimensionality of the input tensor:

..  code-block:: python

    x = np.random.randn(10, 20, 30, 40)

    # Compute the sum along the last axis of x:

    y = einx.sum("s... [c] -> s...", x)
    # expands to
    y = einx.sum("s1 s2 s3 [c] -> s1 s2 s3", x)

..  code-block:: python

    x = np.random.randn(10)

    # Compute the sum along the last axis of x:

    y = einx.sum("s... [c] -> s...", x)
    # expands to
    y = einx.sum("[c] ->", x)

Ellipses may be used multiple times with different sub-expressions:

..  code-block:: python

    x = np.random.randn(10, 20)
    y = np.random.randn(30, 40)

    z = einx.add("a..., b... -> a... b...", x, y)
    # expands to
    z = einx.add("a1 a2, b1 b2 -> a1 a2 b1 b2", x)

If the same axis name is expanded with an ellipses multiple times, the number of repetitions must match across all occurrences:

..  code-block:: python

    x = np.random.randn(10, 20)
    y = np.random.randn(10, 20)

    z = einx.add("s..., s... -> s...", x, y)
    # expands to
    z = einx.add("s1 s2, s1 s2 -> s1 s2", x, y)

..  code-block:: python

    x = np.random.randn(10, 20)
    y = np.random.randn(10, 20, 30)

    z = einx.add("s..., s... -> s...", x, y)
    # Raises an exception: Mismatching number of repetitions for axis 's'

Ellipses may be applied to any type of sub-expression, including brackets and axis compositions:

..  code-block:: python

    x = np.random.randn(4, 640, 480, 3)

    # Compute the global spatial mean of a batch of images

    y = einx.mean("b [s]... c -> b c", x)
    # expands to
    y = einx.mean("b [s1] [s2] c -> b c", x)

..  code-block:: python

    x = np.random.randn(640, 480, 3)

    # Divide an image into a (flattened) list of 4x4 patches

    y = einx.id("(s ds)... c -> (s...) ds... c", x, ds=4)
    # expands to
    y = einx.id("(s1 ds1) (s2 ds2) c -> (s1 s2) ds1 ds2 c", x, ds1=4, ds2=4)

    # y has shape (19200, 4, 4)

..  code-block:: python

    x = np.random.randn(640, 480)

    # Perform mean pooling with 4x4 patches (if evenly divisible)

    y = einx.mean("(s [ds])...", x, ds=4)
    # expands to
    y = einx.mean("(s1 [ds1]) (s2 [ds2]) -> s1 s2", x, ds1=4, ds2=4)

Ellipses not only allow writing more concise expressions for multi-dimensional operations, but also indicate similar treatment of some subsequent axes in an operation (*e.g.*, spatial axes).

Additional axis constraints for axes expanded by ellipses may be provided both as lists matching the repetition number, and as simple integers that apply to all repetitions:

..  code-block:: python

    y = einx.id("a -> a b...", x, b=(5, 6))
    # expands to
    y = einx.id("a -> a b1 b2", x, b1=5, b2=6)

    y = einx.id("(a b)... -> a... b...", x, b=2)
    # expands to
    y = einx.id("(a1 b1) (a2 b2) -> a1 a2 b1 b2", x, b1=2, b2=2)

Lastly, einx allows writing anonymous ellipses without a preceding expression. In this case, a new, unique axis name is generated and used for all occurrences of the anonymous ellipsis:

..  code-block:: python

    z = einx.add("..., ... -> ...", x, y)
    # same as
    z = einx.add("s..., s... -> s...", x, y)

..  note::

    Anonymous ellipses in einx align with the behavior of ellipses in `einops <https://einops.rocks/>`__.
    See :doc:`this page </comparison/ein>` for a general comparison of einx with einops notation.

..  note::

    Ellipses in einx are motivated by their role in programming languages such as Java, C++ and Swift: In these languages, an ellipsis
    is placed after a parameter to indicate that the function or template accepts a variable number of
    arguments of that type. The actual number is determined from how many arguments are provided at
    a given call site. For example, in Java:

    .. code-block:: java

        // "String..." accepts a variable number of arguments that match the type "String"
        void printAll(String... values) {
            for (String v : values) {
                System.out.println(v);
            }
        }

        printAll("a", "b", "c");
        printAll("a");

    Similarly, in einx an ellipsis indicates that an operation accepts a variable number of axes of a certain type,
    and the actual number is determined from the constraints provided at a given call site:

    .. code-block:: python

        # "s..." accepts a variable number of axes that match the sub-expression "s"
        y = einx.sum("s... [c] -> s...", np.random.randn(10, 20, 30, 40))
        y = einx.sum("s... [c] -> s...", np.random.randn(10, 20))



Nested ``,`` and ``->``
***********************

The operators ``->`` and ``,`` may be nested with an expression to allow writing complex operations more concisely. If either operator
appears nested within an expression, the expression is expanded by moving these operators to the top level:

..  code-block:: python

    einx.{...}("a [b -> c]", x)
    # expands to
    einx.{...}("a [b] -> a [c]", x)

    einx.{...}("b p [i,->]", x, y)
    # expands to
    einx.{...}("b p [i], b p -> b p", x, y)


.. _tutorial-tensor-factories:

Tensor factories
****************

All einx functions allow passing *tensor factories* as arguments instead of actual tensor objects. A tensor factory is simply a
Python function or callable object that accepts a ``shape`` argument (*i.e.* tuple of integers) and returns a tensor of the specified shape. This allows
deferring the creation of a tensor until its shape has been resolved, and avoids having to manually determine the shape in advance. Tensor
factories provide no axis constraints for the corresponding input expression.

For example:

..  code-block:: python

    noise = lambda shape: np.random.normal(size=shape)

    # Add random noise to x
    z = einx.add("a b c, b c", x, noise)

Inspecting the generated code snippet for the above operation shows that the tensor factory is called with the required shape
to create the tensor before using it in the operation:

..  code-block:: python

    >>> x = np.ones((10, 20, 30))
    >>> y = lambda shape: np.random.rand(*shape)
    >>> print(einx.add("a b c, a b", x, y, graph=True))
    import numpy as np
    def op(a, b):
        b = b((10, 20))
        assert isinstance(b, np.ndarray), "Invalid type as output of tensor factory"
        assert (tuple(b.shape) == (10, 20)), "Expected shape (10, 20) as output of tensor factory"
        b = np.reshape(b, (10, 20, 1))
        c = np.add(a, b)
        return c
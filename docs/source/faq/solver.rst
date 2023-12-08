How does einx parse Einstein expressions?
#########################################

Overview
--------

einx functions accept a operation string that specifies the shapes of input and output tensors and the requested operation in Einstein notation. For example:

..  code::

    einx.mean("b (s [r])... c -> b s... c", x, r=4) # Mean-pooling with stride 4

To identify the backend operations that are required to execute this statement, einx first parses the operation string and determines an *Einstein expression tree*
for each input and output tensor. The tree represents a full description of the tensor's shape and axes marked with brackets. The nodes represent different types of
subexpressions such as axis lists, compositions, ellipses and concatenations. The leaves of the tree are the named and unnamed axes of the tensor. The expression trees
are used to determine the required rearranging steps and axes along which backend operations are applied.

einx uses a multi-step process to convert expression strings into expression trees:

* **Stage 0**: Split the operation string into separate expression strings for each tensor.
* **Stage 1**: Parse the expression string for each tensor and return a (stage-1) tree of nodes representing the nested subexpressions.
* **Stage 2**: Expand all ellipses by repeating the respective subexpression, resulting in a stage-2 tree.
* **Stage 3**: Determine a value for each axis (i.e. the axis length) using the provided constraints, resulting in a stage-3 tree, i.e. the final Einstein expression tree.

Stage 0: Splitting the operation string
---------------------------------------

The operation string is first split into separate expression strings for each tensor. In the above example, this results in ``b (s [r])... c`` and ``b s... c``
for the input and output tensor, respectively. Inputs and outputs are separated by ``->``, and multiple tensors on each side are separated by ``,``. The order of the tensors
matches the order of the parameters and return values of the einx function.

Most functions also accept shorthand operation strings to avoid redundancy and facilitate more concise expressions. For example, in ``einx.mean`` the output expression can
be implicitly determined from the input expression by removing marked axes, and can therefore be omitted (see :func:`einx.reduce`):

..  code::

    einx.mean("b (s [r])... c -> b s... c", x, r=4)
    # same as
    einx.mean("b (s [r])... c", x, r=4)

Another example of shorthand notation in :func:`einx.dot`:

..  code::

    einx.dot("a b, b c -> a c", x, y)
    # same as
    einx.dot("a [b] -> a [c]", x, y)
    # same as
    einx.dot("a [b|c]", x, y)

See :doc:`Tutorial: Tensor manipulation </gettingstarted/tensormanipulation>` and the documentation of the respective functions for allowed shorthand notation.

Stage 1: Parsing the expression string
--------------------------------------

The expression string for each tensor is parsed into a stage-1 tree using a simple lexer and parser. The tree is a nested structure of nodes that represent the different types of
subexpressions:

.. figure:: /images/stage1-tree.png
  :width: 300
  :align: center

  Stage-1 tree for ``b (s [r])... c``.

This includes several semantic checks, e.g. to ensure that axis names do not appear more than once per expression.

Stage 2: Expanding ellipses
---------------------------

To expand the ellipses in a stage-1 expression, einx first determines the *depth* of every axis, i.e. the number of ellipses that the axis is nested in. In the above expression,
``b`` and ``c`` have depth 0, while ``s`` and ``r`` have depth 1. einx ensures that the depth of axes is consistent over different expressions: E.g. an operation
``b s... c -> b s c`` would raise an exception.

In a second step, the *expansion* of all ellipses, i.e. the number of repetitions, is determined using the constraints provided by the input tensors. For example, given a tensor with
rank 4, the ellipsis in ``b (s [r])... c`` has an expansion of 2. einx ensures that the expansion of all axes is consistent over different expressions: E.g. an
operation ``s..., s... -> s...`` would raise an exception if the two input tensors have different rank.

The expression ``b (s [r])... c`` is expanded to ``b (s.0 [r.0]) (s.1 [r.1]) c`` for a 4D input tensor:

.. figure:: /images/stage2-tree.png
  :height: 240
  :align: center

  Stage-2 tree for ``b (s [r])... c`` on input tensor with rank 4.

Parameters that are passed as additional constraints to the einx function, such as ``r=4`` in

..  code::

    einx.mean("b (s [r])... c -> b s... c", x, r=4)

are included when solving for the depth and expansion of all expressions. Unlike the root
expressions describing the input tensors, these parameters can be given both in expanded (``r=(4, 4)``) and unexpanded form (``r=4``). In the first case, the values of ``r.0`` and ``r.1``
are defined explicitly and an additional constraint for the expansion of ``r`` is included. In the second case, the same value is used for the repetitions ``r.0`` and ``r.1``. This
extends to nested ellipsis with depth > 1 analogously.

Stage 3: Determining axis values
--------------------------------

In the last step, the values of all axes (i.e. their lengths) are determined using the constraints provided by the input tensors and additional parameters. For example, the above
expression with an input tensor of shape ``(2, 4, 8, 3)`` and additional constraint ``r=4`` results in the following final Einstein expression tree:

.. figure:: /images/stage3-tree.png
  :height: 240
  :align: center

  Stage-3 tree for ``b (s [r])... c`` for tensor with shape ``(2, 4, 8, 3)`` and constraint ``r=4``.

The value of axis lists and axis concatenations is determined as the product and sum of their children's values, respectively. An unnamed axis (i.e. a number in the expression string such as
``1``, ``16``) is treated as an axis with a new unique name and an additional constraint specifying its value.

Solver
------

einx uses a `SymPy <https://www.sympy.org/en/index.html>`_-based solver to determine the depth and expansion of all expressions in stage 2, and the values of all axes in stage 3 by providing
equations representing the respective constraints.

Instead of directly applying the solver to these equations, einx first determines *equivalence classes* of axes that are known to have
the same value (from equations like ``a = b`` and ``a = 1``) and for each equivalence class passes a single variable to `SymPy <https://www.sympy.org/en/index.html>`_.
This speeds up the solver and allows raising more expressive exceptions when conflicting constraints are found.

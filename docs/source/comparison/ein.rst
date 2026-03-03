ein*-notations
##############

Several notations inspired by einsum's take on Einstein's summation convention have been introduced since einsum's release.
These notations typically cover a specific set of several tensor operations, but are incompatible with each other and do not extend to other tensor operations.
In contrast, einx provides a universal notation and model for tensor operations and explicitly incorporates the concept of vectorization into the notation.

In the following, we consider the ein*-notations of einsum, einops, eindex and einmesh, and compare them to einx notation.
See :doc:`this page in the documentation <../more/isthiseinsteinnotation>` for a general comparison with the concept of Einstein's summation notation.



einsum
******

Short introduction
==================

einsum notation is intended to represent generalized tensor contractions, and provides two ways to specify an operation.
The first, rarely used *Einstein mode* follows
`Einstein's summation convention <https://en.wikipedia.org/wiki/Einstein_notation>`__ by listing
the mathematical indices from an Einstein summation expression as characters in a string. The contracted axes
are determined according to the rule:

    **Einstein rule:** An index is contracted iff it appears exactly twice in the expression.

For example, the index ``j`` appears exactly twice in the following expression, and is therefore contracted:

.. code-block:: python

    z = einsum("ij,jk", x, y)
    # matrix multiplication: j is contracted

The second, more commonly used *non-Einstein mode* extends the string with an arrow and output indices. The contracted axes
are determined according to the rule:

.. _non-einstein-rule:

    **Non-Einstein rule:** An index is contracted iff it appears in the input but not in the output.

For example:

.. code-block:: python

    z = einsum("ij,jk->ik", x, y)
    # matrix multiplication: j is contracted

    z = einsum("bij,bjk->bik", x, y)
    # batched matrix multiplication: j is contracted (not representable in Einstein mode)

    z = einsum("bj->j", x)
    # batched sum-reduction: j is contracted (not representable in Einstein mode)

einsum notation typically refers to the second, non-Einstein mode. For more information, refer to
`einsum's documentation <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`__ or
`one of the many tutorials <https://www.google.com/search?q=einsum+tutorial>`__ available online.

einsum notation allows representing a set of four vectorized elementary operations. This becomes clear when comparing
instances of einsum notation with their equivalent einx notation:

..  code-block:: python

    # sum-of-product operation (= dot-product)
    einsum("ab,bc->ac", x, y)
    einx.dot("a [b], [b] c -> a c", x, y)

    # sum operation
    einsum("ab->a", x)
    einx.sum("a [b] -> a", x)

    # product operation
    einsum("a,b->ab", x, y)
    einx.multiply("a, b -> a b", x, y)

    # identity operation
    einsum("ab->ba", x)
    einx.id("a b -> b a", x)

For the set of these four operations, einsum and einx notation are similar up to bracket placement
(and syntactic sugar such as spaces and multi-letter axis names). The following sections describe the differences
between einsum and einx notation.

.. _einsum-not-universal:

Not universal
=============

einsum supports the set of four elementary operations shown above, but does not generalize to other operations.

In contrast, einx is defined by analogy with loop notation (see :ref:`the tutorial <tutorial-loop-notation>`) rather than by analogy with Einstein's summation convention.
This allows einx to express the vectorization of *any* tensor operation, including other reduction and scalar operations, as well as
more complex operations and arbitrary, custom operations:

..  code-block:: python

    # Reduction operations
    z = einx.sum("a [b] -> a", x)
    z = einx.min("a [b] -> a", x)
    z = einx.max("a [b] -> a", x)
    z = einx.mean("a [b] -> a", x)

    # Scalar operations
    z = einx.multiply("a, b -> a b", x, y)
    z = einx.add("a, b -> a b", x, y)
    z = einx.divide("a, b -> a b", x, y)
    z = einx.subtract("a, b -> a b", x, y)

    # More complex operations
    z = einx.softmax("a [b] -> a [b]", x)
    z = einx.sort("a [b] -> a [b]", x)

    # Any other operation 1
    def myfunc(x, y):
        return jnp.sum(x) * 2 + jnp.flip(y) # Implemented with Jax
    einmyfunc = einx.jax.adapt_with_vmap(myfunc)
    z = einmyfunc("a [b], c [b] -> c [b] a", x, y)

    # Any other operation 2
    einsolve = einx.jax.adapt_with_vmap(jnp.linalg.solve)
    z = einsolve("a [n n], [n] d -> a d [n]", x, y)

Less readable
=============

einx provides different entry-points for the different elementary operations that are also included in einsum notation, and
denotes axes that are forwarded to the elementary operation with brackets.
This improves clarity and readability of the expressions, as shown in the following examples:

..  code-block:: python

    # Which operation is computed here?
    z = einsum("beag,dcf->abcdegf", x, y)

    # This is just an element-wise multiplication!
    z = einx.multiply("b e a g, d c f -> a b c d e g f", x, y)

..  code-block:: python

    # Which axes are contracted?
    z = einsum("b q k h, b k h c -> b q h c", x, y)

    # Only k is contracted!
    z = einx.dot("b q [k] h, b [k] h c -> b q h c", x, y)

Subject to silent failures
==========================

einx avoids potential silent failures that may arise in einsum due to typos in the expression.
For example, consider the following matrix multiplication in einsum notation:

..  code-block:: python

    z = einsum("ij,jk->ik", x, y)
    # Computes dot-product along j

    # -> Now introduce a typo

    z = einsum("ij,ik->ik", x, y)
    # This fails silently!
    # It computes sum-reduction along j
    # followed by element-wise multiplication.

einx catches such errors by checking for the signature of the respective operation:

..  code-block:: python

    z = einx.dot("i [j], [j] k -> i k", x, y)
    # Computes dot-product along j

    # -> Now introduce a typo

    # Option 1
    z = einx.dot("i [j], [i] k -> i k", x, y)
    # This raises an exception
    # due to inconsistent bracket placement:
    # Axis i is given both inside and outside brackets.

    # Option 2
    z = einx.dot("i j, i k -> i k", x, y)
    # This raises an exception
    # since operation does not match
    # the dot-product signature.

No axis compositions
====================

einx notation includes the option to use **axis compositions** (*i.e.* flattened and concatenated axes, see :ref:`the tutorial <tutorial-axis-compositions>`)
which are not supported in einsum:

..  code-block:: python

    # Axis (un)flattening
    z = einx.id("a b c -> a (b c)", x)
    z = einx.id("a (b c) -> a b c", x)

    # Axis (un)concatenation
    z1, z2 = einx.id("a (b1 + b2) -> a b1, a b2", x, b1=4)
    z = einx.id("a b1, a b2 -> a (b1 + b2)", x1, x2)

.. _einsum-ellipsis:

Limited use of ellipses
=======================

einsum notation includes the option to represent multiple axes using the ellipsis specifier ``...``.
All ellipses always refer to the *same* set of axes in all input and output tensors:

.. code-block:: python

    # Same set of axes
    z = einsum("... a, ... a -> ...", x, y)

In contrast, einx's composable ellipses (see :ref:`the tutorial <tutorial-ellipsis>` for more information) allow representing multiple sets of axes:

.. code-block:: python

    # Same set of axes
    z = einx.dot("x... [a], x... [a] -> x...", x, y)

    # Multiple sets of axes
    z = einx.dot("x... [a], y... [a] -> x... y...", x, y)

See :ref:`the below comparison with einops <einops-ellipsis>` for a more detailed comparison.


einops
******

Short introduction
==================

`einops <https://einops.rocks/>`__ notation extends einsum notation to support
multi-letter axis names,

.. code-block:: python

    z = einsum("ij,jk->ik", x, y)
    z = einops.einsum(x, y, "axis1 axis2, axis2 axis3 -> axis1 axis3")

additional reduction operations with the same notation as einsum,

.. code-block:: python

    z = einsum("ab->a", x)                            # sum-reduction
    z = einops.reduce(x, "a b -> a", reduction="sum") # sum-reduction
    z = einops.reduce(x, "a b -> a", reduction="min") # min-reduction

repetition of values along axes in the output expression,

.. code-block:: python

    z = einops.repeat(x, "a b -> a b c", c=3)
    # values are repeated along c

and axis (un)flattening:

.. code-block:: python

    z = einops.rearrange(x, "a (b c) -> a b c", b=3) # unflattening
    z = einops.rearrange(x, "a b c -> a (b c)")      # flattening

More information can be found in `einops's tutorials <https://einops.rocks/1-einops-basics/>`__.

Not universal
=============

einops takes einsum's notation for sum-reductions and uses it to support other reduction operations in addition to einsum's sum-reduction:

..  code-block:: python

    z = einsum("ab->a", x)
    z = einops.reduce(x, "a b -> a", reduction="sum")
    z = einops.reduce(x, "a b -> a", reduction="min")
    z = einops.reduce(x, "a b -> a", reduction="max")
    z = einops.reduce(x, "a b -> a", reduction="mean")

However, einops does not similarly extend einsum's notation for element-wise multiplication to other element-wise operations:

..  code-block:: python

    z = einsum("a,b->ab", x, y)
    # This is not part of einops:
    z = einops.elementwise(x, "a, b -> a b", reduction="subtract")

In contrast, einx supports any tensor operation, including other reduction and element-wise operations. See :ref:`the above list <einsum-not-universal>` for examples.

Ad-hoc treatment of repetition
==============================

einops introduces a *repeat rule* in addition to einsum's :ref:`non-Einstein rule <non-einstein-rule>` that allows repeating inputs along new axes in the output:

    **Repeat rule:** Values are repeated along an output axis iff it appears in the output but not in the input.

This rule is only supported in the ``einops.repeat`` function:

.. _einops-repeat-rule:

..  code-block:: python

    z = einops.repeat(x, "a b -> a b c", c=3)
    # values are repeated along c

In einx, repetition is simply a type of vectorization that is independent of any particular operation, and does not require introduction of new concepts or entry-points:

..  code-block:: python

    z = einx.id("a b -> a b c", x, c=3)
    z = einx.sum("a [b] -> a c", x, c=3)

The equivalent behavior to einops's repeat rule can be illustrated by inspecting the loop notation representation of these examples:

..  code-block:: python

    for a in range(...): for b in range(...): for c in range(...):
        z[a, b, c] = x[a, b]

    for a in range(...): for c in range(...):
        z[a, c] = sum(x[a, :])

Importantly, the representation in loop notation only indicates *what* result the einx operation computes, but now *how*. In practice,
einx invokes operations such as ``np.sum`` for the above case only once and broadcasts the result as required.

Ad-hoc choice of entry-points
=============================

einops chooses the set of entry-points ``einops.{reduce|rearrange|repeat}`` (in addition to ``einops.einsum``) based on how
a tensor operation changes the number of dimensions of its inputs. `The paper <https://openreview.net/pdf?id=oapKSVM2bcj>`__ describes this as follows:

    We made an explicit choice to separate scenarios of “adding dimensions” (repeat), “removing dimensions” (reduce) and “keeping number of elements the same” (rearrange)

This choice leads to some challenges in the notation:

*   How to support operations that keep the number of dimensions the same but modify the tensor elements (*e.g.*, softmax, sort)?
*   How to support operations that introduce new dimensions other than through simple repetition?
*   How to support operations with multiple inputs and/ or outputs where "adding", "removing" and "keeping" is not clearly defined?

Rather than choosing the entry-points based on the change in dimensionality, einx provides *one entry-point per elementary operation* with a name that reflects the respective operation:

*   ``einops.rearrange`` and ``einops.repeat`` both compute a vectorized identity map - in einx, they are represented by the single entry-point ``einx.id``
    with a name reflecting the identity operation.
*   ``einops.einsum`` computes a vectorized sum-of-product operation - in einx, it is represented by the individual entry points ``einx.dot``, ``einx.sum``, ``einx.multiply`` and ``einx.id`` with
    names that reflect the respective operations.
*   ``einops.reduce`` computes various reduction operations - in einx, each reduction operation is represented by its own entry-point, *e.g.*, ``einx.sum``, ``einx.min``, ``einx.max``, ``einx.mean``
    with names that reflect the respective operations.

No support for axis concatenations
==================================

In einx, an axis composition represents any composition of multiple axes into a single new axis (see :ref:`the tutorial <tutorial-axis-compositions>` for more information).
einx includes support for flattened and concatenated axes.

Flattened axes were first introduced by einops and called "axis compositions". `The paper <https://openreview.net/pdf?id=oapKSVM2bcj>`__ describes them as follows:

    The main novelty of our notation is the composition and decomposition of axes denoted by parenthesis.
    (De)composition uses C-ordering convention

We follow the notational choice with parentheses, but use the term "axis composition"
to denote the general concept of composing axes, and "flattened axis" to denote the specific case of row-major ordering (*i.e.* C-ordering) that is also used in einops.
einx notation supports concatenated axes as another type of axis composition:

..  code-block:: python

    z = einx.id("a (b c) -> a b c", x, b=3)
    z = einops.rearrange(x, "a (b c) -> a b c", b=3)

    z = einx.id("a b c -> a (b c)", x)
    z = einops.rearrange(x, "a b c -> a (b c)")

    z1, z2 = einx.id("a (b1 + b2) -> a b1, a b2", x, b1=4)
    # no concatenated axes in einops

    z = einx.id("a b1, a b2 -> a (b1 + b2)", x1, x2)
    # no concatenated axes in einops

.. _einops-ellipsis:

Limited use of ellipses
=======================

Ellipses in einops align with the behavior of einsum: All ellipses in an expression refer to the same set of primitive axes.

In contrast, ellipses in einx are defined *to repeat the preceding expression* rather than a fixed set of primitive axes. This
allows representing complex multi-dimensional operations more concisely (see :ref:`the tutorial <tutorial-ellipsis>` for more information).

For instance, to represent a spatial mean-pooling operation over an n-dimensional tensor, we may compose brackets with flattened axes and ellipsis in einx
as follows:

..  code-block:: python

    y = einx.mean("(s [ds])...", x, ds=4) # n-dim

    # For a 2D input tensor, this expands to:

    y = einx.mean("(s1 [ds1]) (s2 [ds2])", x, ds1=4, ds2=4) # 2-dim

    # Outputs are determined implicitly in einx.mean,
    # so this is equivalent to:

    y = einx.mean("(s1 [ds1]) (s2 [ds2]) -> s1 s2", x, ds1=4, ds2=4) # 2-dim

In contrast, ellipses in einops may only represent a fixed set of primitive axes and do not compose, *e.g.*, with ellipses. The same operation
has to be written without ellipses in einops:

..  code-block:: python

    y = einops.reduce(x, "(h dh) -> h", reduction="mean", dh=4) # 1-dim
    y = einops.reduce(x, "(h dh) (w dw) -> h w", reduction="mean", dh=4, dw=4) # 2-dim
    y = einops.reduce(x, "(h dh) (w dw) (d dd) -> h w d", reduction="mean", dh=4, dw=4, dd=4) # 3-dim

Similarly, the depth-to-space operation is represented in einx with ellipses as follows:

..  code-block:: python

    y = einx.id("b s... (c ds...) -> b (s ds)... c", ds=4)

The same operation is written in einops without ellipses:

..  code-block:: python

    y = einops.rearrange(x, "b h (c dh) -> b (h dh) c", dh=4) # 1-dim
    y = einops.rearrange(x, "b h w (c dh dw) -> b (h dh) (w dw) c", dh=4, dw=4) # 2-dim
    y = einops.rearrange(x, "b h w d (c dh dw dd) -> b (h dh) (w dw) (d dd) c", dh=4, dw=4, dd=4) # 3-dim

Composable ellipses not only improve conciseness and readability, but also indicate to the reader that some subsequent axes in an operation (*e.g.*, spatial axes) are treated similarly.



einops.pack, einops.unpack
**************************

Short introduction
==================

In addition to the four main functions described above, einops also introduces ``einops.pack`` and ``einops.unpack`` for concatenation
and splitting of tensors. The notation uses a single expression to represent the shapes of all input and output tensors, and the ``*`` symbol
to represent the concatenated axis: All axes across inputs tensors at the ``*`` are first flattened and then concatenated along this axis
in the output.

For instance, the following operation flattens all but the first two dimensions of the input tensors and
concatenates them along the third dimension:

..  code-block:: python

    # x has shape (2, 3, 4, 5)
    # y has shape (2, 3, 3)

    z, ps = einops.pack([x, y], "a b *")
    # flattenes x to shape (2, 3, 20)
    # concatenates x and y to shape (2, 3, 23)

The notation differs from the original einsum/einops notation:

* Multiple input tensors are represented using a single expression rather than multiple expressions separated by commas.
* There is no output expression.
* The shapes are not represented fully and explicitly.
* The new ``*`` symbol is introduced to represent the concatenated axis.
* The function returns two outputs: The resulting tensor(s), and an object that allows reversing the operation (*i.e.* ``ps``).

For more information, refer to `einops's documentation <https://einops.rocks/4-pack-and-unpack/>`__.

No universal notation
=====================

``einops.pack`` and ``einops.unpack`` introduce a new notation with new rules (for concatenation and splitting) that is inconsistent
with the existing einsum/einops notation (*e.g.*, for sum-of-product and reductions) and does not apply to other operations.

In contrast, einx represents these operations entirely using its existing notation with the concatenated axis composition.
For instance, a concatenation of tensors along the third dimension
is represented in einx with the existing entry-point ``einx.id`` as follows:

..  code-block:: python

    z = einx.id("a b c1, a b c2 -> a b (c1 + c2)", x, y)

No support for unaligned shapes
===============================

``einops.pack`` and ``einops.unpack`` use a single expression to represent the shape of all inputs and the output. This limits
the expressiveness of the notation and does not allow representing operations where the input and output shapes do not already align.
In contrast, einx notation specifies shapes for each input and the output individually. For example:

..  code-block:: python

    z = einx.id("a b1, a b2 -> a (b1 + b2)", x, y)
    # Axes already align -> works with einops.pack
    z, ps = einops.pack([x, y], "a b *")

    z = einx.id("a b1, b2 a -> a (b1 + b2)", x, y)
    z = einx.id("a b1, b2 -> a (b1 + b2)", x, y)
    z = einx.id("a1 b1, a2 b2 -> a1 a2 (b1 + b2)", x, y)
    z = einx.id("a1, a2 -> a1 a2 (1 + 1)", x, y)
    # Axes do not align -> does not work with einops.pack
    z, ps = einops.pack([x, y], "?")

The expressiveness of einx's notation allows representing many common concatenation operations that are not expressible
with ``einops.pack`` and ``einops.unpack``, such as appending a number to the channel dimension of an image

..  code-block:: python

    img_out = einx.id("b h w c1, -> b h w (c1 + 1)", img, 42.0)

or creating mesh-grids (similar to ``np.meshgrid``):

..  code-block:: python

    x = np.arange(64)
    y = np.arange(48)

    grid = einx.id("x, y -> x y (1 + 1)", x, y)

Less self-documenting
=====================

einx notation (as well as the original einsum and einops notations) follows a self-documenting style where the shapes
of all inputs and outputs are fully specified in the expression. In contrast, ``einops.pack`` and ``einops.unpack`` use the
``*`` symbol to hide the respective axes across the different arguments:

..  code-block:: python

    # Shapes of x and y are not fully documented
    z, ps = einops.pack([x, y], "a b *")

    # All axes are documented
    z = einx.id("a b c1, a b       -> a b (c1 + 1      )", x, y)
    z = einx.id("a b c1, a b c2    -> a b (c1 + c2     )", x, y)
    z = einx.id("a b c1, a b c2 c3 -> a b (c1 + (c2 c3))", x, y)


eindex
******

Short introduction
==================

`eindex <https://github.com/arogozhnikov/eindex>`__ proposes a notation for expressing indexing operations on tensors (*e.g.*, gather-, scatter-, arg-operations).
For example, the following eindex operation gathers pixel colors from a batch of images at the coordinates specified in an index tensor:

..  code-block:: python

    # eindex
    colors = EX.gather(img, idx, "b h w c, [h, w] b -> b c")

The sub-expression ``[h, w]`` denotes an axis with length 2 in the tensor whose values are
used to index the axes ``h`` and ``w`` of the value tensor. For more information, see the
`documentation <https://github.com/arogozhnikov/eindex/blob/main/tutorial/tutorial.ipynb>`__.

The operation is expressed in einx notation as follows:

..  code-block:: python

    # einx
    colors = einx.get_at("b [h w] c, [2] b -> b c", x, idx)

No universal notation
=====================

eindex notation is inconsistent with the existing einops notation: It introduces a new notational concept (*i.e.* an indexing axis such as ``[h, w]``)
that only applies to indexing operations, and does not apply einsum's :ref:`non-Einstein rule <non-einstein-rule>` or einops's :ref:`repeat rule <einops-repeat-rule>`.

In contrast, einx represents these operations entirely using its existing notation:

..  code-block:: python

    # gather
    z = einx.get_at("b [h w] c, b [2] -> b c", x, idx)

    # scatter-set
    z = einx.set_at("b [h w] c, b [2], b c -> b [h w] c", x, idx, updates)

    # scatter-add
    z = einx.add_at("b [h w] c, b [2], b c -> b [h w] c", x, idx, updates)

    # argmax
    z = einx.argmax("b [h w] c -> b c [2]", x)

Only supports index-axis-first
==============================

The indexing axis in eindex must always appear as the first axis in the index tensor:

..  code-block:: python

    # This is supported
    colors = EX.gather(img, idx, "b h w c, [h, w] b -> b c")

    # This is not supported
    colors = EX.gather(img, idx, "b h w c, b [h, w] -> b c")

In contrast, einx does not impose restrictions on the input shapes:

.. code-block:: python

    # This is supported
    colors = einx.get_at("b [h w] c, [2] b -> b c", x, idx)

    # This is also supported
    colors = einx.get_at("b [h w] c, b [2] -> b c", x, idx)

    # This is also supported
    colors = einx.get_at("b [h w] c, b, b -> b c", x, idx_h, idx_w)



einmesh
*******

Short introduction
==================

`einmesh <https://github.com/Niels-Skovgaard-Jensen/einmesh>`__ proposes a notation for expressing mesh-grid operations (similar to ``np.meshgrid``):

..  code-block:: python

    xs, ys = einmesh.LinSpace(0, 1, 10), einmesh.LinSpace(-1, 1, 20)

    # Coordinates returned as separate tensors
    x, y = einmesh.numpy.einmesh("x y", x=xs, y=ys)

    # Coordinates stacked along "*" axis
    xy = einmesh.numpy.einmesh("x y *", x=xs, y=ys)

Already covered by einx notation
================================

Mesh-grid operations (both in einmesh and ``np.meshgrid``) are compositions of broadcasting and concatenation with existing generator
functions such as ``np.linspace``. As such, they are special cases of the vectorized identity map and already
expressible entirely using ``einx.id``:

..  code-block:: python

    xs, ys = np.linspace(0, 1, 10), np.linspace(-1, 1, 20)

    # Coordinates returned as separate tensors
    x, y = einx.id("x, y -> x y, x y", xs, ys)

    # Coordinates stacked along last axis
    xy = einx.id("x, y -> x y (1 + 1)", xs, ys)

While einmesh and ``np.meshgrid`` require knowledge of the concept and meaning of mesh-grids, the einx expression
clearly self-documents the behavior without requiring the introduction of these new abstractions and documentation.



Other ein*-notations
********************

The following is a non-exhaustive list of projects for ein*-notations in chronological order:

.. list-table::
   :widths: 20, 20, 60
   :header-rows: 1

   * - Name
     - First commit
     - Operations

   * - `einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_
     - `Jan 23 2011 <https://github.com/numpy/numpy/commit/a41de3adf9dbbff9d9f2f50fe0ac59d6eabd43cf>`_
     - Dot-product, transpose, trace/diag, sum-reduction.
   * - `einops <https://github.com/arogozhnikov/einops>`_
     - `Sep 22 2018 <https://github.com/arogozhnikov/einops/commit/8e72d792ee88dae177aba3e299179ed478b9a592>`_
     - Dot-product, transpose, reshape, repeat, trace/diag, reduction, stack/concat/split.
   * - `einindex <https://github.com/malmaud/einindex>`_
     - `Dec 3 2018 <https://github.com/malmaud/einindex/commit/5eb212246d6dfa7061cb76545ac1cb8e41c82525>`_
     - Indexing.
   * - `einop <https://github.com/cgarciae/einop>`_
     - `Nov 21 2020 <https://github.com/arogozhnikov/einops/pull/91/commits/b959fff865a534b3f9800024558b24759f3b4002>`_
     - → *einops*
   * - `einshape <https://github.com/google-deepmind/einshape>`_
     - `Jun 22 2021 <https://github.com/google-deepmind/einshape/commit/69d853936d3401c711a723f938e6e20cf3811359>`_
     - Transpose, reshape, repeat.
   * - `jaxtyping <https://github.com/patrick-kidger/jaxtyping>`_
     - `Jul 1 2022 <https://github.com/patrick-kidger/jaxtyping/commit/7ac6ee04a8ec2f1a6b724a1ed2414d438069f2cf>`_
     - Typing.
   * - `eindex <https://github.com/arogozhnikov/eindex>`_
     - `Mar 11 2023 <https://github.com/arogozhnikov/eindex/commit/b787619efd868b7f5100cd69267aa80c4a6c8621>`_
     - Indexing.
   * - `eingather <https://twitter.com/francoisfleuret/status/1661372730241953793>`_
     - `May 24 2023 <https://twitter.com/francoisfleuret/status/1661372730241953793>`_
     - Indexing.
   * - `eins <https://github.com/nicholas-miklaucic/eins>`_
     - `Mar 14 2024 <https://github.com/nicholas-miklaucic/eins/commit/dc5e9a0a3f5bf6fb9e62427b6cedf1ffab1a8873>`_
     - Custom.
   * - `einshard <https://github.com/yixiaoer/einshard>`_
     - `Mar 24 2024 <https://github.com/yixiaoer/mistral-v0.2-jax/commit/b800c054109a14fb04ce72ed1c990c7aa7bba628>`_
     - Sharding.
   * - `shardops <https://github.com/MatX-inc/seqax/tree/main>`_
     - `May 4 2024 <https://github.com/MatX-inc/seqax/commit/db2bd8f8492875d7d09bacfb23b4b76bd5fec220>`_
     - Sharding.
   * - `einmesh <https://github.com/Niels-Skovgaard-Jensen/einmesh>`_
     - `Mar 15 2025 <https://github.com/Niels-Skovgaard-Jensen/einmesh/commit/efdb7e4fdf4a3e6334e09974b4c3c13d783047da>`_
     - Mesh-grid.
Is this Einstein notation?
##########################

What is Einstein notation?
==========================

Einstein introduced what is now known as *Einstein's summation convention* (or: *Einstein notation*) for the mathematical notation of tensor contractions
in `The Foundation of the General Theory of Relativity <https://onlinelibrary.wiley.com/doi/epdf/10.1002/andp.19163540702>`_ (page 781):

    *German original*: "Es ist deshalb moglich, ohne die Klarheit zu beeintrachtigen, die Summenzeichen wegzulassen. Dafür führen wir die Vorschrift ein:
    Tritt ein Index in einem Term eines Ausdruckes zweimal auf, so ist über ihn stets zu summieren"

    *Translated to English*: "It is therefore possible, without compromising clarity, to omit the summation signs. To that end, we introduce the rule:
    If an index appears twice in a term of an expression, it is always to be summed over"

As an example, in the following contraction of :math:`A` and :math:`B` the index :math:`j` appears twice, and the summation sign over :math:`j` may therefore be omitted:

..  math::

    {\sum}_j A_{ij} B_{jk} = A_{ij} B_{jk}

The purpose of Einstein's summation convention thus is to distinguish between summed-over and free indices in tensor contractions *implicitly*, rather than *explicitly* with summation signs.

How is einsum related to Einstein notation?
===========================================

`einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_ is a notation for expressing tensor contractions in Python/ Numpy that is inspired by Einstein notation.
It provides two different modes:

*   **Einstein mode:** In this mode, the mathematical indices of the tensor contraction are simply listed in a comma-delimited string to compute the respective contraction:

    .. code-block:: python

        A = np.random.rand(2, 3)
        B = np.random.rand(3, 4)

        # Einstein-mode
        C = np.einsum("ij,jk", A, B)

    The mode relies on the ability of Einstein's summation convention to implicitly determine the role of the indices: ``j`` appears twice and is thus summed over, while ``i`` and ``k``
    appear only once and are "free" indices.

*   **Non-Einstein mode:** In this mode, the string is extended with an arrow and output indices:

    .. code-block:: python

        A = np.random.rand(2, 3)
        B = np.random.rand(3, 4)

        # Einstein-mode
        C = np.einsum("ij,jk->ik", A, B)

    This mode does *not* follow Einstein notation, but a different convention: An index is contracted iff it appears in the input but not in the output.
    In the above example, the index ``j`` appears in the input but not in the output and is thus summed over, while ``i`` and ``k`` appear in the output and are "free" indices.

While the Einstein mode of einsum follows Einstein's summation convention, it is rarely used in practice. The non-Einstein mode is much more common,
but does not follow Einstein's summation convention.

..  note::

    Searching for ``np.einsum`` with ``->`` on Github (``language:Python "np.einsum" "->"``) yields 77k results, while ``np.einsum`` without ``->``
    (``language:Python "np.einsum" NOT "->"``) yields only 6k results. For ``torch.einsum`` the difference is 199k to 2k.

The main commonality between einsum's most widely used mode and Einstein notation is that both rely on an implicit convention to determine which
indices in a tensor contraction are summed over and which are not. This is similarly the case with the popular package `einops <https://einops.rocks/>`_
that extends the non-Einstein mode of einsum.

Nevertheless, the *ein* terminology has become associated in the community with this general style of representing tensor operations
where a string of axis names represents tensor dimensions and their usage in a tensor operation.
Many subsequent packages have adopted the naming convention for notations of other tensor operations (*e.g.*
`einmesh <https://github.com/Niels-Skovgaard-Jensen/einmesh>`_,
`eindex <https://github.com/arogozhnikov/eindex>`_, `eingather <https://twitter.com/francoisfleuret/status/1661372730241953793>`_).
See :doc:`this page in the documentation <../comparison/ein>` for a more detailed description of einsum, einops and and other ein*-notations.



How is einx related to Einstein notation?
=========================================

einx notation does not follow Einstein notation. It is instead defined to represent the vectorization of arbitrary operations by analogy with loop notation.

einx also differs fundamentally from einsum/einops which are defined to represent mathematical indices of tensor contractions and related operations,
and do not incorporate the concept of vectorization into the notation. einx also does not rely on implicit conventions to distinguish the role of
axes, but explicitly uses brackets for this purpose.

However, the notation of einx leads to similar expressions as einsum/einops for operations that are also representable by them. For example:

..  code-block:: python

    z = einx.dot("a [b], [b] c -> a c", x, y)
    z = einsum("ab,bc->ac", x, y)

    z = einx.sum("a [b] -> a", x)
    z = einsum("ab->a", x)

    z = einx.multiply("a, b -> a b", x, y)
    z = einsum("a,b->ab", x, y)

    z = einx.id("a b -> b a", x)
    z = einsum("ab->ba", x)

Due to the similarity with einsum/einops and the general idea of using a string of axis names to represent tensor operations,
we name our notation *einx*, but avoid claims of it being "Einstein-like" or "Einstein-inspired".
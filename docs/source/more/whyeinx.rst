Why einx?
#########

einx is interpretable
=====================

To read and understand an einx operation such as 

..  code-block:: python

    z = einx.dot("b q [k] h, b [k] h c -> b q h c", x, y)

we may follow this simple approach:

1.  First identify the elementary operation and its signature in the operation, *i.e.* axes marked with brackets:

    ..  code-block:: python

        "[k], [k] -> []"

    This tells us we got two vectors and return a scalar (by computing their ``dot`` product).

2.  Any additional information in the expression only tells us that this elementary operation is somehow repeated over different parts
    of the input and output tensors. This repeated application (*i.e.* vectorization) is analogous to the corresponding loop notation:

    ..  code-block:: python

        # Write one loop for each remaining axis (i.e. those not marked with brackets)
        for b in range(...):
            for q in range(...):
                for h in range(...):
                    for c in range(...):
                        # 1. Retrieve the specified parts of the input tensors
                        # 2. Apply the elementary operation
                        # 3. Write to the specified parts of the output tensor
                        z[b, q, h, c] = dot(x[b, q, :, h], y[b, :, h, c])

------------------------

As another example, consider the following indexing operation:

..  code-block:: python

    z = einx.get_at("b [h w] c, b p [2] -> b p c", x, y)

1.  First identify the elementary operation and its signature in the operation:

    ..  code-block:: python

        "[h w], [2] -> []"

    According to the function documentation (:func:`einx.get_at`), the second argument is a coordinate and it points to a pixel in the first argument. The elementary operation just
    returns the value at that coordinate.

2.  Now repeat this elementary operation over the remaining axes:

    ..  code-block:: python

        for b in range(...):
            for p in range(...):
                for c in range(...):
                    z[b, p, c] = get_at(x[b, :, :, c], y[b, p, :])

einx is universal
=================

*Any* tensor operation may be vectorized with einx notation, and *any* vectorization representable in loop notation may also be expressed with
einx notation. For example, any operation may be used for ``SOME_OPERATION`` in the following example, as long as the argument shapes that are marked
with brackets match the signature expected by the operation:

..  code-block:: python

    z = einx.SOME_OPERATION("a [b], [b] c -> a c", x, y)

    for a in range(...): for c in range(...):
        z[a, c] = SOME_OPERATION(x[a, :], y[:, c])

The einx library already contains implementations for many commonly used tensor operations such as ``einx.sum`` and ``einx.dot``. To vectorize operations that
are not included in the namespace ``einx.*``, einx provides adapters that allow invoking custom operations with einx notation. The most versatile one is
``adapt_with_vmap`` which may be used as follows. For example,

..  code-block:: python

    def myoperation(x, y):
        return 2 * x + torch.sum(y)
    einmyoperation = einx.torch.adapt_with_vmap(myoperation)

defines a new einx operation which can be invoked with einx notation

..  code-block:: python

    z = einmyoperation("a [c], b [c] -> a b [c]", x, y)

yielding the same output as the analogous loop representation:

..  code-block:: python

    for a in range(...): for b in range(...):
        z[a, b, :] = myoperation(x[a, :], y[b, :])

einx is self-documenting
========================

einx notation explicitly documents the shapes of all input and output arguments in each operation. In contrast, shapes are hidden in classical operations,
and must be documented separately or inferred from other parts of the code:

..  code-block:: python

    z = einx.dot("b1 i [j], b2 [j] k -> b1 i b2 k", x, y)
    z = np.dot(x, y) # What are the shapes of x, y, and z?

einx is declarative
===================

einx notation is declarative: The user only declares *what* the inputs and outputs look like, and the system determines how to compute the operation.
In contrast, Numpy-like notation is *imperative*: It requires the user to express *how* to achieve the desired result, *e.g.* involving reshaping, broadcasting, and
transposing dimensions:

..  code-block:: python

    # einx: "This is what the inputs and outputs look like, please rearrange as required"
    y = einx.id("a b c -> b a c", x)

    # Numpy-like: "Swap the first and second axis, and keep the third axis in place"
    y = np.transpose(x, (1, 0, 2))

..  code-block:: python

    # einx: "This is what the inputs and outputs look like, please rearrange and add as required"
    y = einx.add("a d e, c b e -> a b c d e", x, y) 

    # Numpy-like: "First swap the first and second axis of y while keeping the third in place,
    # then introduce two unitary axes at these positions in x, and two unitary axes at these
    # positions in y, then apply the addition operation while broadcasting the unitary axes
    # accordingly"
    x[:, np.newaxis, np.newaxis] + np.transpose(y, (1, 0, 2))[np.newaxis, :, :, np.newaxis]

The user may still inspect the compiled code snippet for a given einx operation by passing ``graph=True`` (see :ref:`this <tutorial-compiled-code>` for more information)
to verify *how* the operation is actually computed on a given backend.

einx requires learning fewer abstractions
=========================================

Many operations and corresponding rules that have to be learned in Numpy-like notation represent the
same elementary operation, but with different function interfaces to represent different vectorization cases.
Since einx represents vectorization entirely using einx notation, these operations reduce to fewer abstractions that
have to be learned with a simpler interface. The following shows examples of such cases.

Matmul or dot product or inner product?
---------------------------------------

Numpy-like frameworks represent the operations *matrix multiplication*, *dot product*, and *inner product*
using different abstractions:

..  code-block:: python

    z = np.matmul(x, y)             # Matrix multiplication
    z = np.dot(x, y)                # Dot product
    z = np.tensordot(x, y, axes=1)  # Also dot product
    z = np.inner(x, y)              # Inner product

The names do not communicate the difference in their behavior: ``np.dot`` reduces along the last axis of ``x`` and the second-to-last axis of ``y``, while
``np.inner`` reduces along the last axis of both ``x`` and ``y``. ``np.matmul`` applies Numpy's broadcasting rules, while the other three operations do not.
These different abstractions and rules must be learnt from documentation.

In contrast, einx uses the same abstraction to represent all these operations. They are simply cases of the vectorized dot-product:

..  code-block:: python

    z = einx.dot("b i [j], b [j] k -> b i k", x, y)         # Example of np.matmul
    z = einx.dot("b1 i [j], b2 [j] k -> b1 i b2 k", x, y)   # Example of np.dot
    z = einx.dot("b1 [j], b2 [k] -> b1 b2", x, y)           # Example of np.inner
    z = einx.dot("[j] b1, [j] b2 -> b1 b2", x, y)           # Example of np.tensordot with axes=(0, 0)

Learning the abstraction ``einx.dot`` only requires learning that it computes a dot product with the signature

..  code-block:: python

    "[j], [j] -> []"

(as well as multiple subsequent dot-products if requested). Any additional functionality is expressed using einx notation which applies consistently
across all operations.

Gather or take or index?
------------------------

Numpy-like frameworks provide different functions for gathering values from tensors at specified indices, such as the following:

..  code-block:: python

    z = torch.take(x, indices)
    z = torch.gather(x, dim, indices)
    z = torch.index_select(x, dim, indices)
    z = tf.gather(x, indices, axis=...)
    z = tf.gather_nd(x, indices, batch_dims=1)

The names alone do not communicate the difference in their behavior: ``torch.take`` does not support batch dimensions in the value tensor, while the others do.
In ``torch.gather`` the value and index tensors may have common batch dimensions, while in ``torch.index_select`` and ``tf.gather`` the vectorized dimensions are
disjunct. ``tf.gather`` and ``torch.gather`` have the same name, but follow different behavior. These different abstractions and rules must be learnt from documentation.

In contrast, einx uses the same abstraction to represent all these operations. They are simply different vectorizations of the same elementary gather operation:

..  code-block:: python

    z = einx.get_at("[x], ... -> ...", x, y)            # Example of torch.take
    z = einx.get_at("[x] b c, i b c -> i b c", x, y)    # Example of torch.gather with dim=0
    z = einx.get_at("a [x] c, i -> a i c", x, y)        # Example of torch.index_select and tf.gather with axis=dim=1
    z = einx.get_at("a [...], a b [i] -> a b", x, y)    # Example of tf.gather_nd with batch_dims=1

Learning the abstraction ``einx.get_at`` only requires learning the possible signatures of the elementary ``get_at`` operation as well as the simple index retrieval that it performs.
Any additional functionality is expressed using einx notation which applies consistently across all operations.

Stacking or concatenating?
--------------------------

Numpy-like frameworks provide the two different abstractions of *stacking* and *concatenating* tensors along an axis:

..  code-block:: python

    z = np.stack([x, y], axis=0)                        # Stack along a new axis
    z = np.concatenate([x, y], axis=0)                  # Concatenate along an existing axis

The names alone do not communicate the difference in their behavior: ``np.stack``
stacks/ concatenates along a new axis, while ``np.concatenate`` stacks/ concatenates along existing axes. The different abstractions must be learnt from documentation.

In contrast, einx uses the same abstraction to represent both these operations. They are simply cases of
vectorized identity maps using the ``+`` axis composition:

..  code-block:: python

    z = einx.id("..., ... -> (1 + 1) ...", x, y)        # What np.stack does
    z = einx.id("a ... , b ... -> (a + b) ...", x, y)   # What np.concatenate does

Mesh-grids?
-----------

A mesh-grid operation (such as `np.meshgrid <https://numpy.org/devdocs/reference/generated/numpy.meshgrid.html>`__) creates coordinate tensors
from individual coordinate vectors by computing the Cartesian product of the vectors. This operation simply broadcasts the input vectors to a common shape
and can thus be expressed purely in terms of a vectorized identity map:

..  code-block:: python

    x = np.arange(10)
    y = np.arange(20)

    xn, yn = np.meshgrid(x, y, indexing="ij")
    xn, yn = einx.id("a, b -> a b, a b", x, y)

..  code-block:: python

    xn, yn = np.meshgrid(x, y, indexing="xy")
    xn, yn = einx.id("a, b -> b a, b a", x, y)

einx self-documents what the operation is doing and does not require introducing the new "mesh-grid" abstraction. In contrast, dedicated mesh-grid operations
require learning a new abstraction, new function interface and meaning of parameters such as ``indexing`` in the above example.

einx allows varying shape and operation independently
=====================================================

einx decouples the representation of an elementary operation from the representation of its vectorization. This allows varying
either representation independently from the other.

Changing the shape
------------------

For example, consider a simple indexing operation in einx and Numpy-like notation
where elements in the argument ``x`` are retrieved at positions stored in the argument ``y``:

..  code-block:: python

    einx.get_at("[x] a, b -> b a", x, y)
    torch.index_select(x, 0, y)

We now change the input and output shapes of this operation. einx allows varying the vectorization
term to reflect these changes and keeps the entry-point fixed. In contrast, changing the shapes in
Numpy-like notation necessitates switching to a different entry-point with a different signature and
vectorization rules, or is not representable using a single entry-point at all:

..  code-block:: python

    # 1. Introduce axis a in 2nd parameter y
    einx.get_at("[x] a, b a -> b a", x, y) 
    torch.take_along_dim(x, y, dim=0) # Have to switch to torch.take_along_dim

    # 2. Introduce axis c
    einx.get_at("[x] b, c b a -> c b a", x, y)
    # No single entry-point in torch -> would require additional rearrange steps

    # 3. Replace 1D indexing with 2D indexing
    einx.get_at("[x y] b, c b a [2] -> c b a", x, y)
    # No single entry-point in torch -> would require additional rearrange steps

Changing the operation
----------------------

In Numpy-like notation, some functions (*e.g.* ``np.kron``) are provided
for particular vectorization cases of an elementary operation (*e.g.* scalar multiplication), but similar
specializations are not available for other elementary operations (*e.g.* scalar addition). In contrast,
einx allows using analogous vectorization patterns across different operations:

..  code-block:: python

    einx.multiply("a..., b... -> (a b)...", x, y)         # Same as np.kron
    einx.add     ("a..., b... -> (a b)...", x, y)         # kron-like add
    einx.less    ("a..., b... -> (a b)...", x, y)         # kron-like less
    einx.id      ("a..., b... -> (a b)... (1 + 1)", x, y) # kron-like stack

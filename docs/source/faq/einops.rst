How does einx compare with einops?
##################################

einx uses Einstein notation that is inspired by and compatible with the notation used in `einops <https://github.com/arogozhnikov/einops>`_,
but follows a novel design:

* Full composability of Einstein expressions: Axis lists, compositions, ellipses and concatenations can be nested arbitrarily (e.g. ``(a b)...`` or
  ``b (1 + (s...)) c``).
* Introduction of ``[]``-notation that allows expressing vectorization in an intuitive and concise way, similar to the ``axis`` argument in Numpy functions (see :ref:`Bracket notation <bracketnotation>`).
* Introduction of concatenations as first-class expressions in Einstein notation.

When combined, these features allow for a concise and expressive formulation of a large variety of tensor operations.

The einx library provides the following additional features:

* Full support for rearranging expressions in all operations (see :doc:`How does einx handle input and output tensors? </faq/flatten>`).
* ``einx.vmap`` and ``einx.vmap_with_axis`` allow applying arbitrary operations using Einstein notation.
* Specializations provide ease-of-use for main abstractions using Numpy naming convention, e.g. ``einx.sum`` and ``einx.multiply``.
* Several generalized deep learning modules in the ``einx.nn.*`` namespace (see :doc:`Tutorial: Neural networks </gettingstarted/neuralnetworks>`).
* Support for inspecting the backend calls made by einx in index-based notation (see :ref:`Inspecting operations <inspectingoperations>`).

A non-exhaustive comparison of operations expressed in einx-notation and einops-notation:

.. list-table::
   :widths: 50 60
   :header-rows: 0

   * - **einx**
     - **einops**
   * - .. code-block:: python

          einx.mean("b [...] c", x)
     - .. code-block:: python

          einops.reduce(x, "b ... c -> b c", reduction="mean")
   * - .. code-block:: python

          einx.mean("b [...] c", x, keepdims=True)
     - .. code-block:: python

          # For 2D case:
          einops.reduce(x, "b h w c -> b 1 1 c", reduction="mean")
   * - .. code-block:: python

          einx.mean("b (s [s2])... c", x, s2=2)
     - .. code-block:: python
      
          # For 2D case:
          einops.reduce(x, "b (h h2) (w w2) c -> b h w c", reduction="mean", h2=2, w2=2)
   * - .. code-block:: python
          
          einx.dot("... [c1|c2]", x, w)
     - .. code-block:: python
          
          einops.einsum(x, w, "... c1, c1 c2 -> ... c2")
   * - .. code-block:: python
          
          einx.rearrange("h a, h -> h (a + 1)", x, y)
     - .. code-block:: python
          
          einops.pack([x, y], "h *")
   * - .. code-block:: python
          
          einx.rearrange("h (a + 1) -> h a, h 1 ", x)
     - .. code-block:: python
      
          einops.unpack(x, [[3], [1]], "h *")
   * - .. code-block:: python
    
          einx.rearrange("a c, 1 -> a (c + 1)", x, [42])
     - Rearranging and broadcasting not supported in ``einops.pack``
   * - .. code-block:: python
          
          einx.dot("... (g [c1|c2])", x, w)
     - Shape rearrangement not supported in ``einops.einsum``
   * - .. code-block:: python
    
          einx.add("... [c]", x, b)
     - Elementwise operations not supported
   * - .. code-block:: python
    
          einx.rearrange("(a b) c -> c (a b)", x)
     - Fails, since values for ``a`` and ``b`` cannot be determined
   * - .. code-block:: python
    
          einx.vmap("b [...] c -> b c", x, op=my_func)
     - vmap not supported

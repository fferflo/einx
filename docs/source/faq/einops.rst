How is einx different from einops?
##################################

einx uses Einstein-inspired notation that is based on and compatible with the notation used in `einops <https://github.com/arogozhnikov/einops>`_,
but introduces several novel concepts that allow using it as a universal language for tensor operations:

* Introduction of ``[]``-notation to express vectorization of elementary operations (see :ref:`Bracket notation <bracketnotation>`).
* Ellipses repeat the preceding expression rather than an anonymous axis. This allows expressing multi-dimensional operations more concisely
  (e.g. ``(a b)...`` or ``b (s [ds])... c``)
* Full composability of expressions: Axis lists, compositions, ellipses, brackets and concatenations can be nested arbitrarily (e.g. ``(a b)...`` or
  ``b (1 + (s...)) c``).
* Introduction of concatenations as first-class expressions.

The library provides the following additional features based on the einx notation:

* Support for many more tensor operations, for example:

  .. code::

     einx.flip("... (g [c])", x, c=2) # Flip pairs of values
     einx.add("a, b -> a b", x, y) # Outer sum
     einx.get_at("b [h w] c, b i [2] -> b i c", x, indices) # Gather values
     einx.softmax("b q [k] h", attn) # Part of attention operation

* Simpler notation for existing tensor operations:

  .. code::

     einx.sum("a [b]", x)
     # same op as
     einops.reduce(x, "a b -> a", reduction="sum")

     einx.mean("b (s [ds])... c", x, ds=2)
     # einops does not support named ellipses. Alternative for 2D case:
     einops.reduce(x, "b (h h2) (w w2) c -> b h w c", reduction="mean", h2=2, w2=2)

* Full support for rearranging expressions in all operations (see :doc:`How does einx handle input and output tensors? </faq/flatten>`).

  .. code::

     einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=16)
     # Axis composition not supported e.g. in einops.einsum.

* ``einx.vmap`` and ``einx.vmap_with_axis`` allow applying arbitrary operations using einx notation.
* Several generalized deep learning modules in the ``einx.nn.*`` namespace (see :doc:`Tutorial: Neural networks </gettingstarted/tutorial_neuralnetworks>`).
* Support for inspecting the backend calls made by einx in index-based notation (see :doc:`Just-in-time compilation </more/jit>`).

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
          
          einx.dot("... [c1->c2]", x, w)
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
          
          einx.dot("... (g [c1->c2])", x, w)
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

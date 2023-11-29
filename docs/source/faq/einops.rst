How does einx compare with einops?
##################################

einx is fully compatible with einops-style notation used in ``einops.{rearrange|repeat|reduce|einsum}``. Beyond that, einx
is characterized by the following features:

* ``[]``-notation allows expressing vectorization in an intuitive and concise way.
* Ellipses can repeat any type of expression, e.g. ``(a b)...``.
* Expressions can be composed arbitrarily, e.g. by nesting ellipses, concatenations and axis-compositions.
* Concatenations are represented as first-class expressions in Einstein notation.
* Specializations provide ease-of-use for some main abstractions using numpy naming, e.g. ``einx.mean`` and ``einx.where``.
* Full support for rearranging expressions in all operations.
* ``einx.vmap`` function for vectorizing arbitrary operations in Einstein notation.
* ``einx.elementwise`` function for element-by-element operations in Einstein notation.
* Several generalized deep learning modules in the ``einx.nn.*`` namespace.

A comparison of operations expressed in einx-notation and einops-notation:

.. list-table:: 
   :widths: 50 60
   :header-rows: 0

   * - **einx**
     - **einops**
   * - ``einx.mean("b [...] c", x)``
     - ``einops.reduce(x, "b ... c -> b c", reduction="mean")``
   * - ``einx.mean("b [...] c", x, keepdims=True)``
     - | For 2D case:
       | ``einops.reduce(x, "b h w c -> b 1 1 c", reduction="mean")``
   * - ``einx.mean("b (s [s2])... c", x, s2=2)``
     - | For 2D case:
       | ``einops.reduce(x, "b (h h2) (w w2) c -> b h w c", reduction="mean", h2=2, w2=2)``
   * - ``einx.dot("... [c1|c2]", x, w)``
     - ``einops.einsum(x, w, "... c1, c1 c2 -> ... c2")``
   * - ``einx.rearrange("h a, h -> h (a + 1)", x, y)``
     - ``einops.pack([x, y], "h *")``
   * - ``einx.rearrange("h (a + 1) -> h a, h 1 ", x)``
     - ``einops.unpack(x, [[3], [1]], "h *")``
   * - ``einx.rearrange("a c, 1 -> a (c + 1)", x, [42])``
     - Rearranging and broadcasting not supported in ``einops.pack``
   * - ``einx.dot("... (g [c1|c2])", x, w)``
     - Shape rearrangement not supported in ``einops.einsum``
   * - ``einx.add("... [c]", x, b)``
     - Elementwise operations not supported
   * - ``einx.rearrange("(a b) c -> c (a b)", x)``
     - Fails, since values for ``a`` and ``b`` cannot be determined
   * - ``einx.vmap("b [...] c -> b c", x, op=my_func)``
     - vmap not supported

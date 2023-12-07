Cheatsheet
##########

**Simple tensor operations in Einstein notation and index-based notation**. More complex operations often require only slight modifications in the Einstein
expression, while the complexity of index-based notation grows rapidly, making it less expressive and more error-prone.

.. list-table:: 
   :widths: 10 45 45
   :header-rows: 1

   * -
     - einx
     - Numpy
   * - Transpose
     - ``einx.rearrange("a b c -> a c b", x)``
     - ``np.transpose(x, (0, 2, 1))``
   * - Compose
     - ``einx.rearrange("a b c -> a (b c)", x)``
     - ``np.reshape(x, (2, -1))``
   * - Decompose
     - ``einx.rearrange("a (b c) -> a b c", x, b=3)``
     - ``np.reshape(x, (2, 3, 4))``
   * - Concatenate
     - ``einx.rearrange("a, b -> (a + b)", x, y)``
     - ``np.concatenate([x, y], axis=0)``
   * - Split
     - ``einx.rearrange("(a + b) -> a, b", x, a=5)``
     - ``np.split(x, [5], axis=0)``
   * - Reduce
     - ``einx.sum("[a] ...", x)``
     - ``np.sum(x, axis=0)``
   * -
     - ``einx.sum("... [a]", x)``
     - ``np.sum(x, axis=-1)``
   * -
     - ``einx.sum("a [...]", x)``
     - ``np.sum(x, axis=tuple(range(1, x.ndim)))``
   * -
     - ``einx.sum("[...] a", x)``
     - ``np.sum(x, axis=tuple(range(0, x.ndim - 1)))``
   * - Elementwise
     - | ``einx.add("a b, b -> a b", x, y)``
       | ``einx.add("a b, b", x, y)``
       | ``einx.add("a [b]", x, y)``
     - ``x + y[np.newaxis, :]``
   * -
     - | ``einx.add("a b, a -> a b", x, y)``
       | ``einx.add("a b, a", x, y)``
       | ``einx.add("[a] b", x, y)``
     - ``x + y[:, np.newaxis]``
   * - Dot
     - | ``einx.dot("a b, b c -> a c", x, y)``
       | ``einx.dot("a [b] -> a [c]", x, y)``
       | ``einx.dot("a [b|c]", x, y)``
     - ``np.einsum("ab,bc->ac", x, y)``
   * -
     - | ``einx.dot("a b, a b c -> a c", x, y)``
       | ``einx.dot("[a b] -> [a c]", x, y)``
       | ``einx.dot("[a b|a c]", x, y)``
     - ``np.einsum("ab,abc->ac", x, y)``
   * - Indexing
     - ``einx.get_at("[h w] c, p [2] -> p c", x, y)``
     - ``x[y[:, 0], y[:, 1]]``
   * -
     - ``einx.set_at("[h] c, p [1], p c -> [h] c", x, y, z)``
     - ``x[y[:, 0]] = z``

**Deep learning modules**. The layer types can be used to implement a wide variety of neural network modules and provide an intuitive and concise description of the
underlying operation.

``import einx.nn.{torch|flax|haiku} as einn``

.. list-table::
   :widths: 40 60
   :header-rows: 0

   * - `Layer Norm <https://arxiv.org/abs/1607.06450v1>`_
     - ``einn.Norm("b... [c]")``

   * - `Instance Norm <https://arxiv.org/abs/1607.08022v3>`_
     - ``einn.Norm("b [s...] c")``

   * - `Group Norm <https://arxiv.org/abs/1803.08494>`_
     - ``einn.Norm("b [s...] (g [c])", g=8)``

   * - `Batch Norm <https://arxiv.org/abs/1502.03167v3>`_
     - ``einn.Norm("[b...] c", decay_rate=0.9)``

   * - `RMS Norm <https://arxiv.org/abs/1910.07467v1>`_
     - ``einn.Norm("b... [c]", mean=False, bias=False)``

   * - Channel mixing
     - ``einn.Linear("b... [c1|c2]", c2=64)``

   * - | Grouped channel mixing
       | (same weights per group)
     - ``einn.Linear("b... (g [c1|c2])", c2=64)``
   * - | Grouped channel mixing
       | (different weights per group)
     - ``einn.Linear("b... ([g c1|g c2])", c2=64)``

   * - Spatial mixing (as in `MLP-Mixer <https://arxiv.org/abs/2105.01601>`_)
     - | ``einn.Linear("b [s...|s2] c", s2=64)``
       | ``einn.Linear("b [s2|s...] c", s=(64, 64))``

   * - Patch embedding (if evenly divisible)
     - ``einn.Linear("b (s [s2|])... [c1|c2]", s2=4, c2=64)``

   * - Dropout
     - ``einn.Dropout("[...]", drop_rate=0.2)``

   * - Spatial dropout
     - ``einn.Dropout("[b] ... [c]", drop_rate=0.2)``

   * - Droppath
     - ``einn.Dropout("[b] ...", drop_rate=0.2)``

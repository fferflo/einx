Cheatsheet
##########

**Einstein-notation in einx and index-based notation in numpy**

.. list-table:: 
   :widths: 10 45 45
   :header-rows: 1

   * -
     - einx
     - numpy
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


**Common neural network operations using einx.***

.. list-table::
   :widths: 40 60
   :header-rows: 0

   * - Linear layer: x * w + b
     - | ``x = einx.dot("b... [c1|c2]", x, w, c2=64)``
       | ``x = einx.add("b... [c]", x, b)``

   * - | Grouped linear layer
       | (same weights per group)
     - | ``x = einx.dot("b... (g [c1|c2])", x, w, g=8, c2=64)``
       | ``x = einx.add("b... (g [c])", x, b, g=8)``

   * - | Grouped linear layer
       | (different weights per group)
     - | ``x = einx.dot("b... ([g c1|g c2])", x, w, g=8, c2=64)``
       | ``x = einx.add("b... [c]", x, b)``

   * - | Spatial mixing (as in `MLP-Mixer <https://arxiv.org/abs/2105.01601>`_)
     - | ``einx.dot("b [s...|s2] c", x, w, s2=64)``
       | ``einx.dot("b [s2|s...] c", x, w, s=(16, 16))``

   * - Global spatial mean pooling
     - ``einx.mean("b [...] c", x)``
   * - | Sum-pooling with kernel_size=stride=2
       | (if evenly divisible)
     - ``einx.sum("b (s [s2])... c", x, s2=2)``

**Common neural network operations using einx.nn.***

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

Cheatsheet
##########

**Simple tensor operations in Einstein notation and index-based notation**.

.. list-table:: 
   :widths: 10 48 45
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
     - ``einx.set_at("[h] c, p, p c -> [h] c", x, y, z)``
     - ``x[y] = z``

Supported tensor frameworks
###########################

**Numpy**: `https://numpy.org/ <https://numpy.org/>`_

Numpy does not support automatic vectorization (``vmap``). einx implements a custom ``vmap`` for Numpy instead using a Python for-loop
for testing and debugging purposes. This affects ``einx.vmap`` and ``einx.{index|get_at|set_at|...}``.

----

**Torch**: `https://pytorch.org/ <https://pytorch.org/>`_

einx disables ``torch.compile`` (using ``torch.compiler.disable``) when JIT-compiling a call into a Python function and reenables it when
executing the function.

----

**Jax**: `https://jax.readthedocs.io/ <https://jax.readthedocs.io/>`_

----

**Tensorflow**: `https://www.tensorflow.org/ <https://www.tensorflow.org/>`_

einx does not support tensors with dynamic shapes (i.e. ``None`` in the shape).

----

**MLX**: `https://ml-explore.github.io/mlx <https://ml-explore.github.io/mlx>`_

``einx.vmap`` and ``einx.{index|get_at|set_at|...}`` are currently not supported (``mx.vmap`` does not support all required primitives yet).

----

**Tinygrad**: `https://tinygrad.org/ <https://tinygrad.org/>`_

``einx.vmap`` and ``einx.{index|get_at|set_at|...}`` are currently not supported.

----

**Dask**: `https://docs.dask.org/en/stable/array.html <https://docs.dask.org/en/stable/array.html>`_

``einx.vmap`` and ``einx.{index|get_at|set_at|...}`` are currently not supported.
# Changelog

## [0.4.1]

### Fixed

- In ``einx.get_at`` with PyTorch, fix support for indexing tensors with a dtype other than int64/long.



## [0.4.0] Fully embrace vectorization!

### Summary

*Vectorization.* This release fully embraces vectorization by analogy with loop notation as the core abstraction of einx: Any einx expression

```python
# einx notation
z = einx.{OP}("a [i j], b -> a b [j]", x, y)
```

will yield the same output as invoking the underlying elementary operation in an analogous loop expression:

```python
# Loop notation
for a in range(...):
    for b in range(...):
        z[a, b, :] = {OP}(x[a, :, :], y[b])
         "a  b [j]"        "a [i  j]"  "b"
```

See the [new documentation](https://einx.readthedocs.io/en/latest/gettingstarted/basics.html) for more information. This definition was already adhered to
almost entirely, but is now strictly enforced through smaller changes in the interface such as renaming `einx.rearrange` to `einx.id` and
removing some specialized behavior in the notation (see details below).

*Backends.* This release introduces major updates to how tensor operations are implemented in einx. This allows adapting arbitrary functions to einx notation

```python
# Define some custom operation
def op(x, y):
    return torch.sum(x, dim=0) * torch.flip(y)

# Adapt to einx notation
einop = einx.torch.adapt_with_vmap(op)

# Invoke using einx notation
result = einop("a [b c], a [c] -> a [c]", x, y)
```

and choosing [different backend implementations for operations](https://einx.readthedocs.io/en/latest/gettingstarted/backends.html) (e.g., Numpy-like notation, vmap-based notation, or einsum notation).

*Clarity.* The release improves clarity through better error reporting among others for syntax and shape errors, [a new documentation](https://einx.readthedocs.io/), and by removing special behavior and edge cases from the einx notation (see details below).

### Added

- **Allow adapting arbitrary functions to einx notation.** einx provides different adapters based on the signature of the wrapped function in the namespace ``einx.{framework}.adapt_*``. The simplest is ``einx.{framework}.adapt_with_vmap`` which uses a framework's ``vmap`` transformation internally, but is only supported for frameworks that provide ``vmap`` (e.g., Jax, PyTorch, MLX, but not Numpy). Other adapters are provided for functions that follow Numpy-like signatures (e.g. reduction operation with ``axis`` parameter). See [the documentation](https://einx.readthedocs.io/en/latest/api/adapters.html) for more information.

  The functions ``einx.{reduce|elementwise|vmap|vmap_with_axis}`` that partially provided this functionality in previous versions have been removed in favor of the new adapters.

- **Add different backend implementations for operations.** Each einx operation can now be invoked using different backend implementations by specifying the ``backend`` argument. For example, passing ``backend="torch.numpylike"`` uses only Numpy-like operations from PyTorch, while ``backend="torch.vmap"`` uses [torch.vmap](https://docs.pytorch.org/docs/latest/generated/torch.vmap.html), and ``backend="torch.einsum"`` uses [torch.einsum](https://docs.pytorch.org/docs/latest/generated/torch.einsum.html) internally (if the operation is expressible using ``torch.einsum``). The default backend ``backend="torch"`` uses a combination of the above. See [the documentation](https://einx.readthedocs.io/en/latest/gettingstarted/backends.html) for more information and examples of the compiled code with different backends.

  Indexing functions (``einx.{get_at|set_at|...}``) were previously implemented only using ``vmap`` which lead to some problems with frameworks that have limited support for ``vmap`` (e.g., PyTorch) or no support for ``vmap`` (e.g., Numpy). The default backend for all frameworks now uses a purely Numpy-like implementation of indexing functions which avoids these issues.

- **Add support for new operations:** ``einx.{argmin|argmax|sort|argsort|logaddexp}``.

- **Support multiple vectorized axes with the same name in input expressions.** In this case, the diagonal of the input tensor is extracted along the specified axes
  before applying the operation. This adheres to the loop notation analogy. For example:
  ```python
  einx.id("a b b c -> a b c", x) # Extracts diagonal along the 'b' axes
  einx.sum("[a] b b c", x) # Extracts diagonal along the 'b' axes, and computes sum along 1st axis
  einx.sum("a [b b] c", x) # 'b' is not vectorized, so the behavior does not apply here. Still computes sum along 2nd and 3rd axis.
  ```

- **Add support for Array API backend.** As a result, einx now supports all tensor frameworks that implement the [Array API standard](https://data-apis.org/array-api/latest/). This requires the [array-api-compat](https://pypi.org/project/array-api-compat/) package to be installed.

- Add ``einx.solve_axes`` and ``einx.solve_shapes``.

### Changed

- **Improve error reporting to improve clarity.** Most errors should be a lot easier to fix now. For example:
  ```python
  x = np.zeros((10, 5))
  einx.id("(a b) c -> a b c", x)
  ```
  raises
  ```
  einx.errors.AxisSizeError: Failed to uniquely determine the size of the axes a, b. Please provide more constraints.
  Expression: "(a b) c -> a b c"
                ^ ^       ^ ^
  The operation was called with the following arguments:
    - Positional argument #1: Tensor with shape (10, 5)
  ```

- **Simplify einx notation by removing special behavior and edge cases:**
  - Deprecate ``keepdims`` argument in reduction functions:
    ```python
    einx.sum("a [b]", x, keepdims=True) # version < 0.4.0
    ```
    The behavior can be equally achieved using a flattened axis:
    ```python
    einx.sum("a ([b])", x) # version >= 0.4.0
    ```
  - Remove ``cse`` argument from einx functions which previously allowed disabling common subexpression elimination.
  - Remove special shorthand notation in dot-product and elementwise operations where two tensors are passed, but the second input expression is determined implicitly:
    ```python
    einx.dot("b [c_in] -> b [c_out]", x, weight) # version < 0.4.0
    einx.add("b [c]", x, bias) # version < 0.4.0
    ```
    The behavior can be equally achieved by explicitly specifying the second input:
    ```python
    einx.dot("b [c_in], [c_in] c_out -> b c_out", x, weight) # version >= 0.4.0
    einx.add("b c, c", x, bias) # version >= 0.4.0
    ```
  - Remove ``einx.arange``:
    ```python
    einx.arange("a b [2]", a=5, b=10) # version < 0.4.0
    ```
    The behavior can be equally achieved using ``einx.id`` with ``np.arange``:
    ```python
    einx.id("a, b -> a b (1 + 1)", np.arange(5), np.arange(10)) # version >= 0.4.0
    ```
  - Deprecate ``einx.check``:
    ```python
    einx.check("a b", x) # version < 0.4.0
    ```
    The behavior can be equally achieved using ``einx.id``:
    ```python
    einx.id("a b", x) # version >= 0.4.0
    ```
  - Change named axes (``"a"``) and unnamed axes (``"1"``) to have identical behavior now. Among others, this now allows squeezing named axes:
    ```python
    einx.id("a b c -> a b", x, c=1) # version >= 0.4.0
    ```
  - Remove automatic reordering of arguments in ``einx.id``:
    ```python
    einx.id("a, b -> (b + a)", x, y) # version < 0.4.0
    ```
    The behavior can be equally achieved by switching the order of the arguments:
    ```python
    einx.id("b, a -> (b + a)", y, x) # version >= 0.4.0
    ```

- Rename ``einx.rearrange`` to ``einx.id`` to reflect that it computes a vectorized identity map. This follows the general naming convention of einx where function names reflect the elementary operation that is computed.
- Clean up public API by moving implementation into ``einx._src`` namespace.
- Remove ``einx.experimental.shard``.
- Remove ``einx.nn``. This namespace contained implementations of neural net layers for different frameworks in einx notation. Supporting many different neural net libraries created an overhead that is not warranted by the benefit. Rather than provide special einx layers, einx may be used internally by layer implementations.
- Remove support for passing lists or tuples as tensor arguments:
  ```python
  einx.add("a b, a", x, [1.0, 2.0, 4.0]) # version < 0.4.0
  ```
  The behavior can be equally achieved by using a Numpy array instead:
  ```python
  einx.add("a b, a", x, np.asarray([1.0, 2.0, 4.0])) # version >= 0.4.0
  ```
- Bump required Python version to 3.10 since [3.8 and 3.9 have reached end-of-life](https://devguide.python.org/versions/).
- Remove all usages of ``tensorflow.experimental.numpy`` in the Tensorflow backend, and instead rely only on standard Tensorflow operations.
- Remove dedicated support for the Dask framework. Dask is now instead supported using the Array API backend.
- Disallow changing order of non-vectorized axes in some einx functions:
  ```python
  einx.softmax("a [b c] -> a [c b]", x) # version < 0.4.0
  ```
  This avoids confusion of vectorized axes (where axis ordering indicates permutation) and non-vectorized axes (where axis ordering only indicates the signature of the elementary operation).
- Disallow using ``|`` as an alternative to ``->`` in einx notation which was previously supported.
- ``einx.dot`` now only supports dot-product operations, and no longer supports other operation signatures also supported by ``einsum``.

### Fixed

- When initializing a backend, delay raising an exception until the backend is used in an operation. This avoids problems where the import of a framework failed, even though it is not actually used with einx.
- Use ``torch.{amin|amax}`` instead of ``torch.{min|max}`` since in some configurations the latter returns a tuple rather than only the reduced tensor (see https://github.com/fferflo/einx/issues/24 and https://github.com/fferflo/einx/issues/26).



## [0.3.0]

### Added

- Add partial support for [tinygrad](https://github.com/tinygrad/tinygrad).
  - Supported:
    - `einx.rearrange`
    - `einx.{elementwise|add|multiply|where|...}`
    - `einx.{reduce|sum|mean|...}`
    - `einx.{vmap_with_axis|flip|softmax|...}`
    - `einx.dot`
  - Not supported:
    - `einx.vmap` (no `vmap` in tinygrad)
    - `einx.{index|get_at|set_at|...}` (due to relying on `einx.vmap`)

### Changed

- Use `tf.gather_nd` instead of `x[y]` to implement `einx.get_at` for Tensorflow.

### Fixed

- Allow empty tuples and lists as constraints for ellipsis parameters.
- Fix shorthand notation in `einx.dot`.



## [0.2.2]

### Added

- Add `einx.experimental.shard`.

### Fixed

- Fix bug when calling einx from multiple threads. (Run unit tests also in multi-threaded context.)



## [0.2.1]

### Changed

- **Remove einx dependency in compiled code:** The code for a traced function now directly imports and uses the namespace
  of the backend (e.g. `import torch`). For example:
  ```python
  >>> print(einx.dot("b q (h c), b k (h c) -> b q k h", x, y, h=16, graph=True))
  import torch
  def op0(i0, i1):
      x0 = torch.reshape(i0, (16, 768, 16, 64))
      x1 = torch.reshape(i1, (16, 768, 16, 64))
      x2 = torch.einsum("abcd,aecd->abec", x0, x1)
      return x2
  ```
  In most cases, compiled functions now contain no reference to other einx code.
- **Improve handling of Python scalars:** (see https://github.com/fferflo/einx/issues/7) einx now only converts `int`, `float` and `bool` to tensor
  objects (e.g. via `torch.asarray`) if the backend function that is called does not support Python scalars (previously all inputs were converted
  to tensor objects). When using PyTorch, the `device` argument will be used to place the constructed tensor on the correct
  device.<br>For example, `torch.add` supports Python scalars
  ```python
  >>> print(einx.add("a,", x, 1, graph=True))
  import torch
  def op0(i0, i1):
      x0 = torch.add(i0, i1)
      return x0
  ```
  while `torch.maximum` does not:
  ```python
  >>> print(einx.maximum("a,", x, 1, graph=True))
  import torch
  def op0(i0, i1):
      x0 = torch.asarray(i1, device=i0.device)
      x1 = torch.maximum(i0, x0)
      return x1
  ```
- Run unit tests for PyTorch and Jax also on the GPU (if it is available).
- Run unit tests also with `jax.jit` and `torch.compile`.

### Fixed

- Add workarounds for issues with `torch.compile`: https://github.com/pytorch/pytorch/issues/94674 and https://github.com/pytorch/pytorch/issues/124269



## [0.2.0]

### Added

- Add partial support for Apple's [mlx](https://github.com/ml-explore/mlx).
  - Supported:
    - `einx.rearrange`
    - `einx.{elementwise|add|multiply|where|...}`
    - `einx.{reduce|sum|mean|...}`
    - `einx.{vmap_with_axis|flip|softmax|...}`
  - Not supported yet:
    - `einx.dot` (`mx.einsum` is not implemented yet)
    - `einx.vmap` (`mx.vmap` does not fully support all primitives yet)
    - `einx.{index|get_at|set_at|...}` (due to relying on `einx.vmap`)
- Add partial support for [dask.array](https://docs.dask.org/en/stable/array.html).
  - Supported:
    - `einx.rearrange`
    - `einx.{elementwise|add|multiply|where|...}`
    - `einx.{reduce|sum|mean|...}`
    - `einx.{vmap_with_axis|flip|softmax|...}`
    - `einx.dot`
  - Not supported:
    - `einx.vmap` (`vmap` not implemented in dask)
    - `einx.{index|get_at|set_at|...}` (due to relying on `einx.vmap`)
- Add environment variable `EINX_WARN_ON_RETRACE` to warn when excessive retracing is detected.

### Changed

- Allow `->` and `,` to be composed with other operators. (This deprecates the existing `[|]` notation which should instead be implemented with
  composable `->`. The feature is still maintained for backwards compatibility). For example:
    - `einx.dot("b [c1->c2]", ...)` expands to `einx.dot("b [c1] -> b [c2]", ...)`
    - `einx.get_at("b p [i,->]", ...)` expands to `einx.get_at("b p [i], b p -> b p", ...)`
- Allow `einx.{set_at|add_at|...}` to be called with zero-sized updates or coordinates (in which case the input tensor is returned as-is).
- Remove `backend.dot` which was not used anywhere but in the unit tests.
- Improve error reporting:
  - Drop internal stack frames when raising exceptions.
  - Better error when passing invalid shape constraints to einx functions.
- Reduce overhead of einx when using the PyTorch backend.

### Fixed

- Fix compatibility of `einx.nn.torch.Norm` with PyTorch 2.2.
- Fix parameters in `einn.param` being ignored.
- Fix bug when using concatenations in `einx.rearrange`. See: https://github.com/fferflo/einx/issues/6
- Fix broadcasting new axes in `einx.vmap_with_axis`.
- Disable `torch.compile` during graph construction using [torch.compiler.disable](https://pytorch.org/docs/stable/generated/torch.compiler.disable.html).


## [0.1.3]

### Added

- Add option to install einx via `pip install einx[torch]` or `pip install einx[keras]` to enforce version requirements on PyTorch or Keras.

### Changed

- Fail gracefully and report error when run with incompatible version of PyTorch and Keras.

### Fixed

- Fix compatibility with 2.0 <= PyTorch < 2.1.



## [0.1.2]

### Added

- Add type annotations to public API.
- Allow passing multiple coordinate tensors in `einx.{get_at|set_at|...}`.
- Allow implicit output shape in `einx.{set_at|add_at|...}`.
- Allow passing backend with string argument to `einx.nn.norm`.
- Make backends accessible as `einx.backend.{NAME}` once they are loaded.

### Changed

- Refactor tracing:
    - Trace vmapped functions (previously kept a pointer to an untraced function).
    - Add shape assertion when calling unsafe functions.
    - Add comments for better inspection.
    - Remove `pass_backend` argument from `einx.vmap`.
    - Cache different functions for different backends.
    - Don't call `backend.to_tensor` if input already has correct type.

  For example, tracing `einx.get_at` now gives the following jit-compiled code:
    ```python
    >>> print(einx.get_at("b [h w] c, b p [2] -> b p c", x, y, graph=True))
    # backend: einx.backend.numpy
    def op1(i0, i1):
        x1 = i1[:, 0]
        x2 = i1[:, 1]
        x0 = backend.get_at(i0, (x1, x2))
        return (x0,)
    def op0(i0, i1, op1=op1):
        op2 = backend.vmap(op1, in_axes=(0, 0), out_axes=(0,))
        op3 = backend.vmap(op2, in_axes=(3, None), out_axes=(2,))
        x0 = op3(i0, i1)
        return x0[0]
    ```

### Fixed

- Fix bug when using "1" as coordinate axis in einx.index.
- Add workaround for scalar indexing operations with torch.vmap (see https://github.com/pytorch/functorch/issues/747).
- Fix support for list/ tuple arguments as tensors with non-trivial shape.
- Change einx.reduce to accept only single tensors as arguments (API allowed multiple arguments, but was not implemented).
- Don't trace and jit functions if EINX_CACHE_SIZE=0.
- Fix bug where some static code analysis tools fail to recognize function specializations.
# Changelog

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
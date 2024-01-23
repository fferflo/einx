# Changelog

## [Unreleased]

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

### Fixed

- Fix bug when using "1" as coordinate axis in einx.index.
- Add workaround for scalar indexing operations with torch.vmap (see https://github.com/pytorch/functorch/issues/747).
- Fix support for list/ tuple arguments as tensors with non-trivial shape.
- Change einx.reduce to accept only single tensors as arguments (API allowed multiple arguments, but was not implemented).
- Don't trace and jit functions if EINX_CACHE_SIZE=0.
- Fix bug where some static code analysis tools fail to recognize function specializations.
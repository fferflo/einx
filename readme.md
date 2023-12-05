# *einx* - Tensor Operations in Einstein-Inspired Notation

einx is a Python library that allows formulating many tensor operations as concise expressions using few powerful abstractions. It is inspired by [einops](https://github.com/arogozhnikov/einops) and [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).

### tl;dr:

- Introduces composable Einstein expressions that are compatible with einops-notation (see [Comparison with einops](https://einx.readthedocs.io/en/latest/faq/einops.html)).
- Integrates easily into existing code using Numpy, PyTorch, Tensorflow and Jax.
- Uses Numpy-like naming conventions:
  - `einx.{sum|mean|any|max|count_nonzero|where|add|logical_and|flip|...}`
  - `[]`-notation similar to `axis` argument (see [Bracket notation](https://einx.readthedocs.io/en/latest/gettingstarted/overview.html#bracket-notation))
- Incurs zero overhead when used with just-in-time compilation like [`jax.jit`](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) or [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), small overhead in eager mode where operations are cached on the first call (see [Performance](https://einx.readthedocs.io/en/latest/gettingstarted/overview.html#performance)).
- Allows inspecting the backend calls in index-based notation that are made for a given einx operation (see [Inspection](https://einx.readthedocs.io/en/latest/gettingstarted/overview.html#inspecting-operations)).
- Provides generalized neural network layers formulated in einx notation (see [Neural networks](https://einx.readthedocs.io/en/latest/gettingstarted/neuralnetworks.html)).

:warning: **This library is currently experimental and may undergo breaking changes.** :warning:

### Getting started

[Go to overview and documentation](https://einx.readthedocs.io/en/latest/gettingstarted/overview.html)

### What does einx look like?

```python
import einx
x = {np.asarray|torch.as_tensor|jnp.asarray|tf.convert_to_tensor}(...)

einx.sum("a [b]", x)                              # Sum-reduction along columns
einx.flip("([a] b)", x)                           # Reverse values along sub-axis

einx.mean("b [s...] c", x)                        # Global mean-pooling
einx.sum("b (s [s2])... c", x, s2=2)              # Sum-pooling with kernel_size=stride=2

einx.dot("b... [c1|c2]", x, w)                    # Linear layer: x * w
einx.add("b... [c]", x, b)                        # Linear layer: x + b

# Grouped linear layer
einx.dot("b... ( g  [c1|c2])", x, w) # Same weights per group
einx.dot("b... ([g c1|g c2])", x, w) # Different weights per group

einx.dot("b [s...|s2] c", x, w)                   # Spatial mixing as in MLP-mixer

einx.vmap("b [s...] c -> b c", x, op=np.mean)     # Global mean-pooling using vectorized map
einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot) # Matmul using vectorized map

einx.rearrange("a, b -> (a + b)", x, y)           # Concatenate
einx.rearrange("b (q + k) -> b q, b k", x, q=2)   # Split
einx.rearrange("b c, 1 -> b (c + 1)", x, [42])    # Append number to each channel

# Layer normalization
mean = einx.mean("b... [c]", x, keepdims=True)
var = einx.var("b... [c]", x, keepdims=True)
x = (x - mean) * torch.rsqrt(var + epsilon)

# Lazy tensor construction
w = torch.nn.parameter.UninitializedParameter()
einx.dot("b... [c1|c2]", x, w, c2=32) # Calls w.materialize(shape)
```

```python
import einx.nn.{torch|flax|haiku} as einn

layernorm       = einn.Norm("b... [c]")
instancenorm    = einn.Norm("b [s...] c")
groupnorm       = einn.Norm("b [s...] (g [c])", g=8)
batchnorm       = einn.Norm("[b...] c", decay_rate=0.9)
rmsnorm         = einn.Norm("b... [c]", mean=False, bias=False)

channel_mix     = einn.Linear("b... [c1|c2]", c2=64)
spatial_mix1    = einn.Linear("b [s...|s2] c", s2=64)
spatial_mix2    = einn.Linear("b [s2|s...] c", s=(64, 64))
patch_embed     = einn.Linear("b (s [s2|])... [c1|c2]", s2=4, c2=64)

dropout         = einn.Dropout("[...]",       drop_rate=0.2)
spatial_dropout = einn.Dropout("[b] ... [c]", drop_rate=0.2)
droppath        = einn.Dropout("[b] ...",     drop_rate=0.2)

# See scripts/train_{torch|flax|haiku}.py for example trainings on CIFAR10
```

### Installation

```python
pip install git+https://github.com/fferflo/einx.git
```
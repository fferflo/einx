# *einx* - Tensor Operations in Einstein-Inspired Notation

einx is a Python library that allows formulating many tensor operations as concise expressions using few powerful abstractions. It is inspired by [einops](https://github.com/arogozhnikov/einops) and [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).

- *Seamless integration* with tensor frameworks like Numpy, PyTorch, Tensorflow, Jax.
- *Zero-overhead* when used with just-in-time compilation like [`jax.jit`](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) or [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). (See [Performance](#performance))
- *Compatible with einops expressions.* Any einops shape expression is also a valid einx shape expression.

If you are new to Einstein-notation, see [this great einops tutorial](https://nbviewer.org/github/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb) for an introduction and many useful examples.<br>If you have already worked with einops, see the following for an introduction to the features of einx.

:warning: **This library is currently experimental and may undergo breaking changes.** :warning:

[Go to Overview](#overview)

### Examples (tl;dr)

```python
import einx

einx.mean("b [s...] c", x)                        # Global mean-pooling (dimension-agnostic)
einx.sum("b (s [s2])... c", x, s2=2)              # Sum-pooling with kernel_size=stride=2 (dimension-agnostic)

einx.dot("b... [c1|c2]", x, w)                    # Linear layer: x * w
einx.add("b... [c]", x, b)                        # Linear layer: x + b

einx.dot("b... ( g  [c1|c2])", x, w)              # Grouped linear layer w/o bias: Same weights per group
einx.dot("b... ([g c1|g c2])", x, w)              # Grouped linear layer w/o bias: Different weights per group

einx.dot("b [s...|s2] c", x, w)                   # Spatial mixing as in MLP-mixer (dimension-agnostic)

mean = einx.mean("b... [c]", x, keepdims=True)    # Layer norm: Normalize mean-variance
var = einx.var("b... [c]", x, keepdims=True)      # Layer norm: Normalize mean-variance
x = (x - mean) * torch.rsqrt(var + epsilon)       # Layer norm: Normalize mean-variance

w = torch.nn.parameter.UninitializedParameter()   # Lazily construct weight
einx.dot("b... [c1|c2]", x, w, c2=32)             # Lazily construct weight: Calls w.materialize(shape)

einx.vmap("b [s...] c -> b c", x, op=np.mean)     # Global mean-pooling using vectorized map
einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot) # Matmul using vectorized map

einx.rearrange("a, b -> (a + b)", x, y)           # Concatenate
einx.rearrange("b (q + k) -> b q, b k", x, q=2)   # Split
einx.rearrange("b c, 1 -> b (c + 1)", x, [42])    # Append number to each channel
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

# *einx* - Tensor Operations in Einstein-Inspired Notation

[![pytest](https://github.com/fferflo/einx/actions/workflows/run_pytest.yml/badge.svg)](https://github.com/fferflo/einx/actions/workflows/run_pytest.yml)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://einx.readthedocs.io)
[![PyPI version](https://badge.fury.io/py/einx.svg)](https://badge.fury.io/py/einx)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

einx is a Python library that allows formulating many tensor operations as concise expressions using Einstein notation. It is inspired by [einops](https://github.com/arogozhnikov/einops), but follows a novel and unique design:

- Fully composable and powerful Einstein expressions with `[]`-notation.
- Support for many tensor operations (`einx.{sum|max|where|add|dot|flip|get_at|...}`) with Numpy-like naming.
- Easy integration and mixing with existing code. Supports tensor frameworks Numpy, PyTorch, Tensorflow and Jax.
- Just-in-time compilation of all operations into regular Python functions using Python's [`exec()`](https://docs.python.org/3/library/functions.html#exec).

*Optional:*

- Generalized neural network layers in Einstein notation. Supports PyTorch, Flax, Haiku, Equinox and Keras.

**Getting started:**

* [Tutorial](https://einx.readthedocs.io/en/latest/gettingstarted/einsteinnotation.html)
* [Example: GPT-2/ Mamba with einx](https://einx.readthedocs.io/en/latest/gettingstarted/gpt2.html)
* [How does einx compare with einops?](https://einx.readthedocs.io/en/latest/faq/einops.html)
* [API reference](https://einx.readthedocs.io/en/latest/api.html)

## Installation

```python
pip install einx
```

See [Installation](https://einx.readthedocs.io/en/latest/gettingstarted/installation.html) for more information.

## What does einx look like?

#### Tensor manipulation

```python
import einx
x = {np.asarray|torch.as_tensor|jnp.asarray|tf.convert_to_tensor}(...) # Create some tensor

einx.sum("a [b]", x)                              # Sum-reduction along columns
einx.flip("... (g [c])", x, c=2)                  # Flip pairs of values along the last axis
einx.mean("b [s...] c", x)                        # Global mean-pooling
einx.sum("b (s [s2])... c", x, s2=2)              # Sum-pooling with kernel_size=stride=2
einx.add("b... [c]", x, b)                        # Add bias

einx.get_at("b [h w] c, b i [2] -> b i c", x, y)  # Gather values at coordinates

einx.rearrange("b (q + k) -> b q, b k", x, q=2)   # Split
einx.rearrange("b c, 1 -> b (c + 1)", x, [42])    # Append number to each channel

einx.dot("... [c1|c2]", x, y)                     # Matmul = linear map from c1 to c2 channels

# Vectorizing map
einx.vmap("b [s...] c -> b c", x, op=np.mean)     # Global mean-pooling
einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot) # Matmul
```

All einx functions simply forward computation to the respective backend, e.g. by internally calling `np.reshape`, `np.transpose`, `np.sum` with the appropriate arguments.

#### Common neural network operations

```python
# Layer normalization
mean = einx.mean("b... [c]", x, keepdims=True)
var = einx.var("b... [c]", x, keepdims=True)
x = (x - mean) * torch.rsqrt(var + epsilon)

# Prepend class token
einx.rearrange("b s... c, c -> b (1 + (s...)) c", x, cls_token)

# Multi-head attention
attn = einx.dot("b q (h c), b k (h c) -> b q k h", q, k, h=8)
attn = einx.softmax("b q [k] h", attn)
x = einx.dot("b q k h, b k (h c) -> b q (h c)", attn, v)

# Matmul in linear layers
einx.dot("b...      [c1|c2]",  x, w)              # - Regular
einx.dot("b...   (g [c1|c2])", x, w)              # - Grouped: Same weights per group
einx.dot("b... ([g c1|g c2])", x, w)              # - Grouped: Different weights per group
einx.dot("b  [s...|s2]  c",    x, w)              # - Spatial mixing as in MLP-mixer
```

See [Common neural network ops](https://einx.readthedocs.io/en/latest/gettingstarted/commonnnops.html) for more examples.

#### Deep learning modules

```python
import einx.nn.{torch|flax|haiku|equinox|keras} as einn

batchnorm       = einn.Norm("[b...] c", decay_rate=0.9)
layernorm       = einn.Norm("b... [c]") # as used in transformers
instancenorm    = einn.Norm("b [s...] c")
groupnorm       = einn.Norm("b [s...] (g [c])", g=8)
rmsnorm         = einn.Norm("b... [c]", mean=False, bias=False)

channel_mix     = einn.Linear("b... [c1|c2]", c2=64)
spatial_mix1    = einn.Linear("b [s...|s2] c", s2=64)
spatial_mix2    = einn.Linear("b [s2|s...] c", s=(64, 64))
patch_embed     = einn.Linear("b (s [s2|])... [c1|c2]", s2=4, c2=64)

dropout         = einn.Dropout("[...]",       drop_rate=0.2)
spatial_dropout = einn.Dropout("[b] ... [c]", drop_rate=0.2)
droppath        = einn.Dropout("[b] ...",     drop_rate=0.2)
```

See `examples/train_{torch|flax|haiku|equinox|keras}.py` for example trainings on CIFAR10, [GPT-2](https://einx.readthedocs.io/en/latest/gettingstarted/gpt2.html) and [Mamba](https://github.com/fferflo/weightbridge/blob/master/examples/mamba2flax.py) for working example implementations of language models using einx, and [Tutorial: Neural networks](https://einx.readthedocs.io/en/latest/gettingstarted/neuralnetworks.html) for more details.

#### Just-in-time compilation

einx traces the required backend operations for a given call into graph representation and just-in-time compiles them into a regular Python function using Python's [`exec()`](https://docs.python.org/3/library/functions.html#exec). This reduces overhead to a single cache lookup and allows inspecting the generated function. For example:

```python
>>> x = np.zeros((3, 10, 10))
>>> graph = einx.sum("... (g [c])", x, g=2, graph=True)
>>> print(graph)
# backend: einx.backend.numpy
def op0(i0):
    x1 = backend.reshape(i0, (3, 10, 2, 5))
    x0 = backend.sum(x1, axis=3)
    return x0
```

See [Just-in-time compilation](https://einx.readthedocs.io/en/latest/gettingstarted/jit.html) for more details.
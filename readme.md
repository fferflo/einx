# *einx* - Tensor Operations in Einstein-inspired Notation

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

### Overview

1. [Installation](#installation)
2. [Basics](#basics)
3. [Ellipses](#ellipses)
4. [Brief notation](#brief-notation)
5. [Vectorized map](#vectorized-map)
6. [Compatibility with tensor frameworks](#compatibility-with-tensor-frameworks)
7. [Compatibility with einops expressions](#compatibility-with-einops-expressions)
8. [Lazy tensor construction](#lazy-tensor-construction)
9. [Examples: einx-einops](#examples-einx-einops)

## Installation

```python
pip install git+https://github.com/fferflo/einx.git
```

## Introduction

If you are new to Einstein-notation, see [this great einops tutorial](https://nbviewer.org/github/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb) for an introduction and many useful examples.

### Basics

einx has a familiar interface:

```python
import numpy as np
import einx
x = np.ones((4, 3, 128, 128))

# Transpose channels-first to channels-last
x = einx.rearrange("batch channels height width -> batch height width channels", x)

# Mean-pooling over 2x2 patches
x = einx.reduce("b (h h2) (w w2) c -> b h w c", x, h2=2, w2=2, op=np.mean)
```

The following main abstractions are provided:

* `einx.rearrange`: Permute axes, insert new broadcasted axes, concatenate and split tensors (similar to `einops.{rearrange|repeat|pack|unpack}`)
* `einx.reduce`: Reduction operations along axes like `np.sum`, `np.mean`, `np.any` (similar to `einops.reduce`)
* `einx.dot`: General tensor dot-products (similar to `einops.einsum`)
* `einx.elementwise`: Element-wise operations like `np.add`, `np.multiply` or `np.where`
* `einx.vmap`: Apply a function over batched inputs

Additionally, many specializations are provided as top-level functions in the `einx.*` namespace following [Numpy](https://numpy.org/doc/stable/reference/routines.math.html) naming:

* Reduction operations: `einx.{sum|prod|mean|any|all|max|min|count_nonzero|...}`
* Element-wise operations: `einx.{add|multiply|logical_and|where|equal|...}`

einx solves expression shapes using symbolic equations with [SymPy](https://www.sympy.org/en/index.html).

### Ellipses

An ellipsis repeats the expression that appears directly in front of it. The number of repetitions is determined from the shapes of the passed arguments. For example, the following operations are equivalent:

```python
einx.rearrange("b c h w  -> b h w  c", x)
einx.rearrange("b c s... -> b s... c", x)
```

Ellipses can appear multiple times per expression and be composed with other expressions arbitrarily:

```python
# Mean pooling over 2x2 patches
einx.mean("b (h h2) (w w2) c -> b h w  c", x, h2=2, w2=2)
einx.mean("b (s s2)...     c -> b s... c", x, s2=2) # or s2=(2, 2)

# Divide image into patches (space-to-depth)
einx.rearrange("b (h h2) (w w2) c -> b h w  (h2 w2 c)", x, h2=2, w2=2)
einx.rearrange("b (s s2)...     c -> b s... (s2... c)", x, s2=2) # or s2=(2, 2)
```

This facilitates writing dimension-agnostic code even for more complex operations.

### Brief notation

To improve the readability of Einstein-inspired notation, einx adds optional brief notations for different types of operations.

#### Reduction

In reduction operations, reduced axes can be specified using `[]`-brackets:

```python
einx.sum("b s... [c]", x)      # Same as: b s... c -> b s...
einx.sum("b (s [s2])... c", x) # Same as: b (s s2)... c -> b s... c
```

This allows for flexible `keepdims=True` behavior out-of-the-box:

```python
einx.sum("b... [c]", x)                # Shape: b...
einx.sum("b... ([c])", x)              # Shape: b... 1
einx.sum("b... [c]", x, keepdims=True) # Shape: b... 1
```

In the second example, `c` is reduced within the axis group `(c)`, resulting in an empty group `()`, i.e. a trivial axis with size 1.

#### Element-wise operations

In element-wise operations, the output expression is determined implicitly as follows: Use one of the input expressions if it contains the axis names of all other input expressions. Otherwise, use all axes in the order they appear in the input expressions.

```python
einx.add("a b, a", x, y)         # Same as: a b, a -> a b
einx.where("a b, a, b", x, y, z) # Same as: a b, a, b -> a b
einx.add("a, b", x, y)           # Same as: a, b -> a b
```

This can further be abbreviated using `[]`-brackets if the operation is binary and the second input is a subexpression of the first:

```python
einx.add("a [b]", x, y) # Same as: a b, b -> a b
```

#### Tensor dot-product

:warning: **This will likely be changed in a future version** :warning:

For 2-place tensor dot-products, the expression for the right input can be determined implicitly:

```python
einx.dot("b c1 -> b c2", x, y) # Same as: b c1, c1 c2 -> b c2

# []-brackets in left input indicate batch axes for right input
einx.dot("[b] c1 -> b c2", x, y) # Same as: b c1, b c1 c2 -> b c2
```

This can be simplified further by using `[input1|output]`-notation that represents both expressions as one:

```python
einx.dot("b [c1|c2]", x, y) # Same as: b c1 -> b c2
```

The left choice represents the first input tensor and the right choice represents the output tensor. The second input tensor's shape is then determined implicitly as shown above.

### Vectorized map

A vectorized map can be used to apply a function over a batch of inputs. For example, consider the following function that computes the mean and max over two tensors:

```python
def func(x, y): # c, d -> 2
    return np.stack([np.mean(x), np.max(y)])
```

The function can be applied over a batch of inputs without modifying the original source code using `einx.vmap`:

```python
einx.vmap("b [c], b [d] -> b [2]", x, y, op=func)
```

Here, `[]`-brackets indicate the axes that `func` is applied over, while all other axes are considered batch axes. `einx.vmap` also accepts multiple batch axes in arbitrary order:

```python
einx.vmap("b1 [c] b2, b2 [d] -> b2 [2] b1", x, y, op=func) # Second input is repeated for b1
```

Other `einx` functions also support batch axes that can similarly be expressed using `einx.vmap`, for example:

```python
einx.mean("a b [c]", x)
einx.vmap("a b [c] -> a b", x, op=np.mean)

einx.add("a b, b", x, y)
einx.vmap("a b, b -> a b", x, y, op=np.add) # Function is applied on scalars

einx.dot("a b, b c -> a c", x, y)
einx.vmap("a [b], [b] c -> a c", x, y, op=np.dot)
```

While using the option without `einx.vmap` is often faster, `einx.vmap` also allows vectorizing functions that do not support batch axes (e.g. [`map_coordinates`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.ndimage.map_coordinates.html)).

`einx.vmap` is implemented using optimized backend methods (e.g. [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html), [`torch.vmap`](https://pytorch.org/docs/stable/generated/torch.vmap.html), [`tf.vectorized_map`](https://www.tensorflow.org/api_docs/python/tf/vectorized_map)).

### Compatibility with tensor frameworks

einx integrates seamlessly with the following tensor frameworks: Numpy, PyTorch, Tensorflow, Jax. The framework is determined from the type of the input tensors and used for all underlying tensor operations.

```python
x = np.zeros((10, 20))
einx.sum("a [b]", x)                        # Use numpy
einx.sum("a [b]", torch.from_numpy(x))      # Use torch
einx.sum("a [b]", tf.convert_to_tensor(x))  # Use tensorflow
einx.sum("a [b]", jnp.asarray(x))           # Use jax
```

Numpy tensors can be mixed with other frameworks in the same operation, in which case the latter backend is used for computations. Frameworks other than Numpy cannot be mixed in the same operation.

```python
x = np.zeros((10, 20))
y = np.zeros((20, 30))
einx.dot("a [c1|c2]", x, torch.from_numpy(y))              # Use torch
einx.dot("a [c1|c2]", x, jnp.asarray(y))                   # Use jax
einx.dot("a [c1|c2]", torch.from_numpy(x), jnp.asarray(y)) # Fails
```

Unkown tensor objects and python sequences are converted to numpy via `np.asarray`.

### Compatibility with einops expressions

For full compatibility with einops shape expressions, einx implicitly converts anonymous ellipses (that do not have a preceeding expression) by adding a name in front:

```python
einx.rearrange("b ... -> ... b", x) # Same as: "b anonymous_ellipsis... -> anonymous_ellipsis... b"
```

This behavior can be turned off:

```python
einx.anonymous_ellipsis_name = None
einx.rearrange("b ... -> ... b", x) # Fails
```

einx currently includes equivalent operations for `einops.rearrange`, `einops.repeat`, `einops.reduce` and `einops.einsum`. Support for other operations (`einops.pack` and `einops.unpack`) might be added in the future.

### Lazy tensor construction

Instead of passing tensors, all operations also accept tensor factories (e.g. a function `lambda shape: tensor`) that are called to create the corresponding tensor when the shape is resolved.

```python
einx.dot("b s... [c1|c2]", x, np.ones, c2=32) # Second input is constructed using np.ones
```

This can for example be used to determine the shape of weight tensors when implementing deep learning modules. For example, the following represents a linear layer in PyTorch:

```python
import torch, einx
class Linear(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weight = torch.nn.parameter.UninitializedParameter()
        self.bias = torch.nn.parameter.UninitializedParameter()

    def forward(self, x):
        x = einx.dot("b... [c1|c2]", x, self.weight, c2=self.channels) # Calls self.weight.materialize
        x = einx.add("b... [c]", x, self.bias)
        return x
```

This uses [PyTorch's lazy modules](https://pytorch.org/docs/stable/generated/torch.nn.modules.lazy.LazyModuleMixin.html#torch.nn.modules.lazy.LazyModuleMixin) since the shape of the weight is only determined the first time the module is called.

## Performance

When using just-in-time compilation like [`jax.jit`](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) or [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), the einx functions for parsing and executing expressions are only run during initialization and result in zero overhead after the code is compiled. To reduce the overhead in eager mode, einx caches parsed operations and reuses them if the input expressions and shapes match. The size of the cache per function can be adjusted by setting the environment variable `EINX_CACHE_SIZE` before `import einx`. The environment variable `EINX_PRINT_CACHE_MISS=true` can be set to indicate when cache misses occur.

## Examples: einx-einops

```python
# ---------------- EINX --------------------   -------------------------- EINOPS -----------------------------
# Global average pooling
einx.mean("b [s...] c", x)                     einops.reduce(x, "b ... c -> b c", reduction="mean")
einx.mean("b [s...] c", x, keepdims=True)      einops.reduce(x, "b h w c -> b 1 1 c", reduction="mean")

# Sum-pooling over 2x2 patches
einx.sum("b (s [s2])... c", x, s2=2)           einops.reduce(x, "b (h h2) (w w2) c -> b h w c",
                                                                            reduction="sum", h2=2, w2=2)

# Linear layer w/o bias
einx.dot("b... [c1|c2]", x, w)                 einops.einsum(x, w, "... c1, c1 c2 -> ... c2")

# Linear layer w/o bias for spatial-mixing (as in MLP-Mixer)
einx.dot("b [s...|s2] c", x, w)                einops.einsum(x, w, "b h w c, h w s2 -> b s2 c")

# Grouped linear layer w/o bias
einx.dot("b... (g [c1|c2])", x, w)             # Shape rearrangement not supported in einops.einsum

# Add bias
einx.add("b... [c]", x, b)                     # Element-wise operations not supported

# Layer scale
einx.multiply("b... [c]", x, scale)            # Element-wise operations not supported

# Elimination of common sub-expressions
einx.rearrange("(a b) c -> c (a b)", x)        # Fails, since values for {a, b} cannot be inferred

# Vectorized map
einx.vmap("b [s...] c -> b c", x, op=my_func)  # Vectorized map not supported
```

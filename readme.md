# *einx* - einsum, einmean, eindot, einadd, ...

einx is a library for **tensor operations in Einstein-inspired notation** that is **inspired by [einops](https://github.com/arogozhnikov/einops) and [einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)**, and offers extended features like a concise notation, implicit tensor shapes, arbitrary composition of expressions, additional tensor operations, lazy tensor construction, a [SymPy](https://www.sympy.org/en/index.html)-based solver and more.

- *Seamless integration* with tensor frameworks like Numpy, PyTorch, Tensorflow, Jax.
- *Zero-overhead* when used with just-in-time compilation like [`jax.jit`](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) or [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html). (See [Performance](#performance))
- *Compatible with einops expressions.* Any einops shape expression is also a valid einx shape expression.

If you are new to Einstein-notation, see [this great einops tutorial](https://nbviewer.org/github/arogozhnikov/einops/blob/master/docs/1-einops-basics.ipynb) for an introduction and many useful examples.<br>If you have already worked with einops, see the following for an introduction to the features of einx.

:warning: **This library is currently experimental and may undergo breaking changes.** :warning:

### Examples (tl;dr)

```python
einx.mean("b [s...] c", x)                      # Global mean-pooling (dimension-agnostic)
einx.sum("b (s [s2])... c", x, s2=2)            # Sum-pooling with kernel_size=stride=2 (dimension-agnostic)

einx.dot("b... [c1|c2]", x, w)                  # Linear layer: x * w
einx.add("b... [c]", x, b)                      # Linear layer: x + b

einx.dot("b... ( g  [c1|c2])", x, w)            # Grouped linear layer w/o bias: Same weights per group
einx.dot("b... ([g] [c1|c2])", x, w)            # Grouped linear layer w/o bias: Different weights per group

einx.dot("b [s...|s2] c", x, w)                 # Spatial mixing as in MLP-mixer (dimension-agnostic)

mean = einx.mean("b... [c]", x, keepdims=True)  # Layer norm: Normalize mean-variance
var = einx.var("b... [c]", x, keepdims=True)    # Layer norm: Normalize mean-variance
x = (x - mean) * torch.rsqrt(var + epsilon)     # Layer norm: Normalize mean-variance

w = torch.nn.parameter.UninitializedParameter() # Lazily construct weight
einx.dot("b... [c1|c2]", x, w, c2=32)           # Lazily construct weight: Calls w.materialize(shape)

einx.vmap("b [s...] c -> b c", x, op=np.mean)   # Global mean-pooling using vectorized map
einx.vmap("a, b c -> a b c", x, y, op=np.add)   # Element-wise addition using vectorized map
```

```python
layernorm    = einx.{torch|flax|...}.Norm("b... [c]")
instancenorm = einx.{torch|flax|...}.Norm("b [s...] c")
groupnorm    = einx.{torch|flax|...}.Norm("b [s...] (g [c])")
batchnorm    = einx.{torch|flax|...}.Norm("[b...] c", decay_rate=0.9)
rmsnorm      = einx.{torch|flax|...}.Norm("b... [c]", mean=False, bias=False)

channel_mix  = einx.{torch|flax|...}.Linear("b... [c1|c2]", c2=64)
spatial_mix1 = einx.{torch|flax|...}.Linear("b [s...|s2] c", s2=64)
spatial_mix2 = einx.{torch|flax|...}.Linear("b [s2|s...] c", s=(64, 64))
patch_embed  = einx.{torch|flax|...}.Linear("b (s [s2|])... [c1|c2]", s2=4, c2=64)

# See scripts/train_{torch|flax}.py for example trainings on CIFAR10
```

### Overview

1. [Installation](#installation)
2. [Short Introduction](#short-introduction)
3. [Long Introduction](#long-introduction)
    1. [Basics](#basics)
    2. [Ellipses](#ellipses)
    3. [Brief notation](#brief-notation-1)
    4. [Vectorized map](#vectorized-map-1)
    5. [Compatibility with tensor frameworks](#compatibility-with-tensor-frameworks)
    6. [Compatibility with einops expressions](#compatibility-with-einops-expressions)
    7. [Lazy tensor construction](#lazy-tensor-construction-1)
4. [Examples: einx-einops](#examples-einx-einops)

## Installation

```python
pip install git+https://github.com/fferflo/einx.git
```

## Short Introduction

#### Main functions

```python
einx.{rearrange|reduce|dot|elementwise}
```
Top-level overloads following [Numpy](https://numpy.org/doc/stable/reference/routines.math.html) naming:
```python
einx.{sum|prod|mean|any|all|max|min|count_nonzero|...}         # specialize einx.reduce
einx.{add|multiply|logical_and|where|equal|...}  # specialize einx.elementwise
```

#### Expressions

Ellipses and axis groups are composable:
```python
einx.rearrange("b (s s2)... c -> b s... s2... c", x, s2=4) # Create 4x4 patches
```

#### Brief notation

Denote reduced axes in `einx.reduce` with `[]`-brackets, e.g.:
```python
einx.mean("b [s...] c", x) # Same as: b s... c -> b c
```

Denote 2nd input shape in `einx.elementwise` with `[]`-brackets, e.g.:
```python
einx.add("b... [c]", x, b) # Same as: b... c, c -> b... c
```

Implicitly determine shape in `einx.dot`, combine expressions with `[|]`-brackets, e.g.:
```python
einx.dot("b... c1, c1 c2 -> b... c2", x, w)
einx.dot("b... c1 -> b... c2", x, w) # Same as above
einx.dot("b... [c1|c2]", x, w) # Same as above
```
:warning: **This will likely be changed in a future version** :warning:

#### Vectorized map

Map a function over batched inputs (see e.g. [`jax.vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html)):
```python
einx.vmap("b [s...] c -> b c", x, op=np.mean) # np.mean is called on tensor with shape "s..." and repeated over b and c
```

#### Lazy tensor construction

Construct a tensor lazily after the shape has been determined:
```python
einx.dot("b... [c1|c2]", x, np.zeros, c2=32)
```

Determine weight shapes in deep learning frameworks:
```python
weight = torch.nn.parameter.UninitializedParameter()
einx.dot("b... [c1|c2]", x, weight, c2=32) # Calls weight.materialize()
```

## Long Introduction

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

The following general operations are supported:

* `einx.rearrange`: Permute axes and insert new broadcasted axes (similar to `einops.rearrange` and `einops.repeat`)
* `einx.reduce`: Reduction operations along axes like `np.sum`, `np.mean`, `np.any` (similar to `einops.reduce`)
* `einx.dot`: General tensor dot-products (similar to `einops.einsum`)
* `einx.elementwise`: Element-wise operations like `np.add`, `np.multiply` or `np.where` (no `einops` equivalent)
* `einx.vmap`: Apply a function over batched inputs (no `einops` equivalent)

Additionally, many specializations are provided as top-level functions in the `einx.*` namespace following [Numpy](https://numpy.org/doc/stable/reference/routines.math.html) naming:

* Reduction operations: `einx.{sum|prod|mean|any|all|max|min|count_nonzero|...}`
* Element-wise operations: `einx.{add|multiply|logical_and|where|equal|...}`

einx solves expression shapes using symbolic equations with [SymPy](https://www.sympy.org/en/index.html).

### Ellipses

An ellipsis is used to repeat the expression that appears directly in front of it. The number of repetitions is determined from the shapes of the passed arguments. For example, the following operations are equivalent:

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

Here, `[]`-brackets indicate the axes that `func` is applied on, while all other axes are considered batch axes. `einx.vmap` also accepts multiple batch axes in arbitrary order:

```python
einx.vmap("b1 [c] b2, b2 [d] -> b2 [2] b1", x, y, op=func) # Second input is repeated for b1
```

Other `einx` functions also support batch axes that can similarly be expressed using `einx.vmap`, for example:

```python
einx.mean("a b [c]", x)
einx.vmap("a b [c] -> a b", x, op=np.mean)

einx.add("a b, b", x, y)
einx.vmap("a b, b -> a b", x, y, op=np.add) # Function applied on scalars

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
einx.dot("b s... [c1|c2]", x, np.ones, c2=32)
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

### Benchmark: Overhead

Overhead of simple operations in einx and einops compared to index-based notation, benchmarked with [scripts/benchmark1.py](scripts/benchmark1.py):

<details>
<summary><i>Benchmark on RTX A6000 and Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz</i></summary>

```
| Method                     |   einx overhead (us) | einops overhead (us)   | einx (ms)        | einops (ms)      | index-based (ms)   |
|----------------------------|----------------------|------------------------|------------------|------------------|--------------------|
| numpy rearrange            |           25.277     | 2.963005875547727      | 0.028 +-   0.000 | 0.006 +-   0.000 | 0.003 +-   0.000   |
| numpy spatial_mean         |           49.045     | 9.344632116456722      | 1.462 +-   0.009 | 1.422 +-   0.007 | 1.413 +-   0.006   |
| numpy channel_mean         |           46.2927    | 7.380074200530893      | 1.360 +-   0.021 | 1.321 +-   0.021 | 1.314 +-   0.021   |
| numpy spatial_add          |           81.1689    |                        | 0.759 +-   0.011 |                  | 0.678 +-   0.010   |
| numpy channel_add          |           78.8445    |                        | 0.736 +-   0.009 |                  | 0.657 +-   0.008   |
| numpy matmul               |           57.7192    | 70.28647139668463      | 0.155 +-   0.003 | 0.168 +-   0.003 | 0.098 +-   0.002   |
| torch-eager rearrange      |           37.5321    | 6.753067175547281      | 0.055 +-   0.001 | 0.024 +-   0.000 | 0.017 +-   0.000   |
| torch-eager spatial_mean   |           37.8233    | 15.910295148690484     | 1.605 +-   0.002 | 1.583 +-   0.002 | 1.567 +-   0.002   |
| torch-eager channel_mean   |           38.9183    | 10.35996495435637      | 1.709 +-   0.004 | 1.681 +-   0.004 | 1.670 +-   0.004   |
| torch-eager spatial_add    |           65.3814    |                        | 3.238 +-   0.002 |                  | 3.173 +-   0.001   |
| torch-eager channel_add    |           82.5337    |                        | 3.243 +-   0.084 |                  | 3.160 +-   0.080   |
| torch-eager matmul         |           69.3549    | 5.083528968195111      | 0.234 +-   0.022 | 0.170 +-   0.020 | 0.165 +-   0.020   |
| torch-compile rearrange    |           -0.02421   | -0.011160969734188859  | 0.049 +-   0.002 | 0.049 +-   0.002 | 0.049 +-   0.002   |
| torch-compile spatial_mean |            0.118609  | -0.06167807926719948   | 1.637 +-   0.001 | 1.636 +-   0.001 | 1.636 +-   0.001   |
| torch-compile channel_mean |           -0.0413706 | -0.1392870520552144    | 1.610 +-   0.001 | 1.610 +-   0.001 | 1.610 +-   0.001   |
| torch-compile spatial_add  |            0.0653779 |                        | 3.232 +-   0.002 |                  | 3.232 +-   0.002   |
| torch-compile channel_add  |            0.0296018 |                        | 3.200 +-   0.002 |                  | 3.200 +-   0.002   |
| torch-compile matmul       |            0.187771  | 0.1698241879542718     | 0.172 +-   0.001 | 0.172 +-   0.001 | 0.171 +-   0.001   |
| jax-jit rearrange          |           -0.99082   | -1.0315198451281773    | 3.516 +-   0.035 | 3.516 +-   0.035 | 3.517 +-   0.036   |
| jax-jit spatial_mean       |            0.451091  | 0.3436785191298589     | 1.902 +-   0.035 | 1.902 +-   0.036 | 1.901 +-   0.035   |
| jax-jit channel_mean       |           -1.28048   | -1.0852500175436222    | 1.982 +-   0.036 | 1.982 +-   0.036 | 1.983 +-   0.035   |
| jax-jit spatial_add        |            0.169596  |                        | 3.585 +-   0.019 |                  | 3.585 +-   0.019   |
| jax-jit channel_add        |           -0.282265  |                        | 3.556 +-   0.024 |                  | 3.556 +-   0.024   |
| jax-jit matmul             |            0.190998  | 0.5361990382274243     | 0.177 +-   0.001 | 0.177 +-   0.001 | 0.176 +-   0.001   |
```

</details>

**Summary:**<br>
   - In *jit-compiled* code, einx and einops have zero overhead compared to index-based notation.
   - In *eager* mode, einx overhead is < 100us and einops overhead is < 20us (except numpy matmul).

### Benchmark: Deep learning modules

Performance of einx deep learning modules compared with native/ canonical versions in PyTorch and Jax, benchmarked with [scripts/benchmark2.py](scripts/benchmark2.py):

<details>
<summary><i>Benchmark on RTX A6000 and Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz</i></summary>

```
| Method                          |   einx overhead (us) | einx (ms)         | native (ms)       | index-based (ms)   |
|---------------------------------|----------------------|-------------------|-------------------|--------------------|
| torch-compile layernorm         |            -2.89867  | 3.608 +-   0.002  | 3.611 +-   0.002  | 3.606 +-   0.002   |
| torch-compile layernorm_fastvar |          -118.207    | 3.491 +-   0.001  | 3.609 +-   0.001  | 3.491 +-   0.001   |
| torch-compile batchnorm         |           257.416    | 6.493 +-   0.001  | 6.235 +-   0.001  | 6.378 +-   0.001   |
| torch-compile batchnorm_fastvar |         -1366.06     | 4.871 +-   0.002  | 6.237 +-   0.002  | 4.795 +-   0.002   |
| torch-compile channel_linear    |            18.4558   | 11.787 +-   0.005 | 11.768 +-   0.006 | 11.783 +-   0.005  |
| torch-compile spatial_mlp       |        -24352.9      | 17.174 +-   0.900 |                   | 41.527 +-   2.105  |
| jax-jit layernorm               |            -0.58402  | 6.760 +-   0.014  |                   | 6.761 +-   0.015   |
| jax-jit layernorm_fastvar       |            -0.206963 | 5.212 +-   0.018  |                   | 5.212 +-   0.018   |
| jax-jit batchnorm               |            -7.04752  | 6.603 +-   0.028  |                   | 6.610 +-   0.023   |
| jax-jit batchnorm_fastvar       |            -4.74965  | 5.013 +-   0.027  |                   | 5.018 +-   0.027   |
| jax-jit channel_linear          |          -126.612    | 12.325 +-   1.180 |                   | 12.452 +-   1.322  |
| jax-jit spatial_mlp             |         -5015.3      | 14.791 +-   0.864 |                   | 19.806 +-   1.370  |
```

</details>

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

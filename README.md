# *einx* - Universal Notation for Tensor Operations

[![pytest](https://github.com/fferflo/einx/actions/workflows/run_pytest.yml/badge.svg)](https://github.com/fferflo/einx/actions/workflows/run_pytest.yml)
[![Documentation](https://img.shields.io/badge/documentation-link-blue.svg)](https://einx.readthedocs.io)
[![PyPI version](https://img.shields.io/pypi/v/einx.svg?color=blue)](https://pypi.org/project/einx/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

einx is a notation and Python library that provides a universal interface to formulate tensor operations in frameworks such as Numpy, PyTorch, Jax, Tensorflow, and MLX.

* [Quickstart](#quickstart)
* [What does einx look like?](#what-does-einx-look-like)
* [How does the notation work?](#how-does-the-notation-work)
* [How are einx operations implemented?](#how-are-einx-operations-implemented)
* [Which operations are supported?](#which-operations-are-supported)

## Quickstart

**Installation:**

```python
pip install einx
```

**Example code:**

```python
import einx
import numpy as np

x = np.ones((10, 20, 30)) # Create some tensor (numpy/torch/jax/tensorflow/mlx/...)

y = einx.sum("a [b] c", x) # Call an einx operation

print(y.shape)
```

**Documentation and tutorials:** [https://einx.readthedocs.io](https://einx.readthedocs.io)

## What does einx look like?

```python
z = einx.id("a (b c) -> (b a) c", x, b=2)             # Permute and (un)flatten axes
z = einx.sum("a [b]", x)                              # Sum-reduction along second axis
z = einx.flip("... (g [c])", x, c=2)                  # Flip pairs of values along the last axis
z = einx.mean("b [...] c", x)                         # Spatial mean-pooling
z = einx.multiply("a..., b... -> (a b)...", x, y)     # Kronecker product
z = einx.sum("b (s [ds])... c", x, ds=(2, 2))         # Sum-pooling with 2x2 kernel
z = einx.add("a, b -> a b", x, y)                     # Outer sum
z = einx.dot("a [b], [b] c -> a c", x, y)             # Matrix multiplication
z = einx.get_at("b [h w] c, b i [2] -> b i c", x, y)  # Gather values at coordinates
z = einx.id("b (q + k) -> b q, b k", x, q=2)          # Split
z = einx.id("b c, -> b (c + 1)", x, 42)               # Append number to each channel
```

See the [documentation](https://einx.readthedocs.io/en/latest/more/operations.html) for more examples.

## How does the notation work?

An einx operation consists of (1) an elementary operation and (2) an einx expression that describes how the elementary operation is vectorized. For example,
the code

```python
z = einx.{OP}("[c d] a, b -> a [e] b", x, y)
```

vectorizes the elementary operation ``{OP}`` according to the expression ``"[c d] a, b -> a [e] b"``.

The meaning of the string expression is defined by analogy with loop notation as follows. The full operation ``einx.{OP}`` will yield the same output as if the elementary
operation ``{OP}`` were invoked in an analogous loop expression:

```python
for a in range(...):
    for b in range(...):
        z[a, :, b] = {OP}(x[:, :, a], y[b])
```

See the [tutorial](https://einx.readthedocs.io/en/latest/gettingstarted/basics.html) for how an einx expression is mapped to the analogous loop expression.

## How are einx operations implemented?

The analogy with loop notation is used only to define what the output of an operation will be. Internally, einx operations are compiled to Python code snippets that invoke operations from the respective tensor framework, rather than using for loops.

The compiled code snippet can be inspected by passing `graph=True` to the einx operation. For example:

```python
>>> x = np.zeros((2, 3))
>>> y = np.zeros((3, 4))
>>> code = einx.add("a b, b c -> c b a", x, y, graph=True)
>>> print(code)

import numpy as np
def op(a, b):
    a = np.transpose(a, (1, 0))
    a = np.reshape(a, (1, 3, 2))
    b = np.transpose(b, (1, 0))
    b = np.reshape(b, (4, 3, 1))
    c = np.add(a, b)
    return c
```

Different [backends](https://einx.readthedocs.io/en/latest/gettingstarted/backends.html) may be used to compile an operation to different implementations, for example following Numpy-like notation, vmap-based notation, or einsum notation.

## Which operations are supported?

**Operations in the API:** einx supports a large set of tensor operations in the namespace ``einx.*``, including reduction, scalar, indexing, some shape-preserving operations, identity map and dot-product. See the [documentation](https://einx.readthedocs.io/en/latest/api/operations.html) for a complete list.

**Operations *not* in the API:** einx additionally allows adapting custom Python functions to einx notation using einx adapters. For example:

```python
# Define a custom elementary operation
def myoperation(x, y):
    x = 2 * x
    z = x + torch.sum(y)
    return z

# Adapt the operation to einx notation
einmyoperation = einx.torch.adapt_with_vmap(myoperation)

# Invoke as einx operation
z = einmyoperation("a [c], b [c] -> a b [c]", x, y)
```

This will yield the same output as if ``myoperation`` were invoked in loop notation:

```python
for a in range(...):
    for b in range(...):
        z[a, b, :] = myoperation(x[a, :], y[b, :])
```

The interface of ``einmyoperation`` matches that of other einx operations. For example, the compiled code snippet can be inspected using ``graph=True``:

```python
>>> code = einmyoperation("a [c], b [c] -> a b [c]", x, y, graph=True)
>>> print(code)
# Constant const1: <function myoperation at 0x49e9aa3cd8a0>
import torch
def op(a, b):
    def c(d, e):
        f = const1(d, e)
        assert isinstance(f, torch.Tensor), "Expected 1st return value of the adapted function to be a tensor"
        assert (tuple(f.shape) == (3,)), "Expected 1st return value of the adapted function to be a tensor with shape (3,)"
        return f
    c = torch.vmap(c, in_dims=(None, 0), out_dims=0)
    c = torch.vmap(c, in_dims=(0, None), out_dims=0)
    g = c(a, b)
    return g
```

See the [documentation](https://einx.readthedocs.io/en/latest/api/adapters.html) for a list of supported adapters.
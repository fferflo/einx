Tutorial: Overview
##################

einx provides a universal interface to formulate tensor operations as concise expressions in frameworks such as
Numpy, PyTorch, Tensorflow and Jax. This tutorial will introduce the main concepts of Einstein-inspired notation
(or *einx notation*) and how it is used as a universal language for expressing tensor operations.

An einx expression is a string that represents the axis names of a tensor. For example, given the tensor

>>> import numpy as np
>>> x = np.ones((2, 3, 4))

we can name its dimensions ``a``, ``b`` and ``c``:

>>> import einx
>>> einx.matches("a b c", x) # Check whether expression matches the tensor's shape
True
>>> einx.matches("a b", x)
False

The purpose of einx expressions is to specify how tensor operations will be applied to the input tensors:

>>> np.sum(x, axis=1)
>>> # same as
>>> einx.sum("a [b] c", x)

Here, ``einx.sum`` represents the elementary *sum-reduction* operation that is computed. The expression ``a [b] c`` specifies
that it is applied to sub-tensors
spanning the ``b`` axis, and vectorized over axes ``a`` and ``c``. This is an example of the general paradigm
for formulating complex tensor operations with einx:

1. Provide a set of elementary tensor operations such as ``einx.{sum|max|where|add|dot|flip|get_at|...}``.
2. Use einx notation as a universal language to express vectorization of the elementary ops.

The following tutorials will give a deeper dive into einx expressions and how they are used to express a large variety of tensor operations.
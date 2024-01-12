.. toctree::
   :caption: Introduction
   :maxdepth: 3

Introduction
############

einx is a Python library that allows formulating many tensor operations as concise expressions using few powerful abstractions. It is inspired by
`einops <https://github.com/arogozhnikov/einops>`_ and `einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_.

*Main features:*

- Fully composable Einstein expressions with ``[]``-notation. Compatible with einops-notation.
- Powerful abstractions: :func:`einx.rearrange`, :func:`einx.vmap`, :func:`einx.vmap_with_axis`
- Ease of use with numpy-like specializations ``einx.{sum|any|max|where|add|flip|get_at|...}`` and shorthand Einstein notation.
- Easy integration with existing code. Supports tensor frameworks Numpy, PyTorch, Tensorflow and Jax.
- No overhead when used with just-in-time compilation (e.g. `jax.jit <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_). Marginal overhead in eager mode due to tracing and caching operations (see :doc:`Performance </gettingstarted/performance>`).

*Optional:*

- Generalized neural network layers using Einstein notation. Supports PyTorch, Flax, Haiku and Equinox.
- Inspecting backend operations that are made for a given einx call (See :ref:`Inspection <inspectingoperations>`).

**Next steps:**

- :doc:`Installation </gettingstarted/installation>`
- :doc:`Tutorial: Einstein notation </gettingstarted/einsteinnotation>`
- :doc:`Tutorial: Tensor manipulation </gettingstarted/tensormanipulation>`
- :doc:`Tutorial: Neural networks </gettingstarted/neuralnetworks>`

Related libraries
-----------------

* `einops <https://github.com/arogozhnikov/einops>`_
* `einsum in Numpy <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_
* `eindex <https://github.com/arogozhnikov/eindex>`_
* `torchdim <https://github.com/facebookresearch/torchdim>`_
* `einindex <https://github.com/malmaud/einindex>`_
* `einshape <https://github.com/google-deepmind/einshape>`_
* `einop <https://github.com/cgarciae/einop>`_
* `eingather <https://twitter.com/francoisfleuret/status/1661372730241953793>`_
* `Named tensors in PyTorch <https://pytorch.org/docs/stable/named_tensor.html>`_
* `Named axes in Jax <https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html>`_
.. toctree::
   :caption: Introduction
   :maxdepth: 3

Introduction
############

einx is a Python library that allows formulating many tensor operations as concise expressions using few powerful abstractions. It is inspired by
`einops <https://github.com/arogozhnikov/einops>`_.

*Main features:*

- Fully composable and powerful Einstein expressions with ``[]``-notation.
- Support for many tensor operations (``einx.{sum|max|where|add|dot|flip|get_at|...}``) with Numpy-like naming.
- Easy integration and mixing with existing code. Supports tensor frameworks Numpy, PyTorch, Tensorflow, Jax and others.
- Just-in-time compilation of all operations into regular Python functions using Python's `exec() <https://docs.python.org/3/library/functions.html#exec>`_.

*Optional:*

- Generalized neural network layers in Einstein notation. Supports PyTorch, Flax, Haiku, Equinox and Keras.

**Next steps:**

- :doc:`Installation </gettingstarted/installation>`
- :doc:`Tutorial: Einstein notation </gettingstarted/einsteinnotation>`
- :doc:`Tutorial: Tensor manipulation </gettingstarted/tensormanipulation>`
- :doc:`Tutorial: Neural networks </gettingstarted/neuralnetworks>`

Related resources
-----------------

* `einops <https://github.com/arogozhnikov/einops>`_
* `einsum in Numpy <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_
* `eindex <https://github.com/arogozhnikov/eindex>`_
* `torchdim <https://github.com/facebookresearch/torchdim>`_
* `einindex <https://github.com/malmaud/einindex>`_
* `einshape <https://github.com/google-deepmind/einshape>`_
* `einop <https://github.com/cgarciae/einop>`_
* `eingather <https://twitter.com/francoisfleuret/status/1661372730241953793>`_
* `einshard <https://github.com/ayaka14732/einshard>`_
* `eins <https://github.com/nicholas-miklaucic/eins>`_
* `Named axes in PyTorch <https://pytorch.org/docs/stable/named_tensor.html>`_
* `Named axes in Jax <https://jax.readthedocs.io/en/latest/notebooks/xmap_tutorial.html>`_
* `Named axes in Penzai <https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html>`_
* `Dex <https://github.com/google-research/dex-lang>`_
* `Named Tensor Notation <https://namedtensor.github.io/>`_
* `Tensor Considered Harmful <https://nlp.seas.harvard.edu/NamedTensor>`_
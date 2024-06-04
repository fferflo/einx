.. toctree::
   :caption: Introduction
   :maxdepth: 3

Introduction
############

einx is a Python library that provides a universal interface to formulate tensor operations in frameworks such as Numpy, PyTorch, Jax and Tensorflow.
The design is based on the following principles:

1. **Provide a set of elementary tensor operations** following Numpy-like naming: ``einx.{sum|max|where|add|dot|flip|get_at|...}``
2. **Use einx notation to express vectorization of the elementary operations.** The notation is inspired by `einops <https://github.com/arogozhnikov/einops>`_,
   but introduces several novel concepts such as ``[]``-bracket notation and full composability that allow using it as a universal language for tensor operations.

einx can be integrated and mixed with existing code seamlessly. All operations are :doc:`just-in-time compiled </more/jit>`
into regular Python functions using Python's `exec() <https://docs.python.org/3/library/functions.html#exec>`_ and invoke operations from the respective framework.

**Next steps:**

- :doc:`Installation </gettingstarted/installation>`
- :doc:`Tutorial </gettingstarted/tutorial_overview>`

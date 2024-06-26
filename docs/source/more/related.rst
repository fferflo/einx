Related projects
################

**Einstein-inspired notation:** Tensor operations are invoked by specifying a string of axis names. All axes are explicitly named.
The *lifetime* of an axis name is delimited by a single operation.

* `einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_: General dot-product.
* `einops <https://github.com/arogozhnikov/einops>`_: General dot-product, rearranging, reduction.
* `eindex <https://github.com/arogozhnikov/eindex>`_: Indexing.
* `einindex <https://github.com/malmaud/einindex>`_: Indexing.
* `einshape <https://github.com/google-deepmind/einshape>`_: Rearranging.
* `einop <https://github.com/cgarciae/einop>`_: See *einops*.
* `eingather <https://twitter.com/francoisfleuret/status/1661372730241953793>`_: Indexing.
* `einshard <https://github.com/ayaka14732/einshard>`_: Sharding.
* `shardops <https://github.com/MatX-inc/seqax/tree/main>`_: Sharding.
* `eins <https://github.com/nicholas-miklaucic/eins>`_: Misc.

**Named axis notation:** Tensor objects are annotated with axis names. The *lifetime* of the axis name is the combined lifetime
of all tensor objects that use it. Tensor operations are invoked by naming the axes corresponding to the elementary operation.
Vectorized axes are implicit.

* `torchdim <https://github.com/facebookresearch/torchdim>`_
* `Tensor Considered Harmful <https://nlp.seas.harvard.edu/NamedTensor>`_
* `Named axes in PyTorch <https://pytorch.org/docs/stable/named_tensor.html>`_
* `Named axes in Penzai <https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html>`_
* `Named axes in xarray <https://docs.xarray.dev/en/stable/>`_
* `Named Tensor Notation <https://namedtensor.github.io/>`_

**Other:**

* `Dex <https://github.com/google-research/dex-lang>`_
* `Shape suffixes <https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd>`_

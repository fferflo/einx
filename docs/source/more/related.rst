Related projects
################

Einstein-inspired notation
==========================

**Summary:** Tensor operations are invoked by specifying a string of axis names. All axes are explicitly named.
The *lifetime* of an axis name is delimited by a single operation.

A non-exhaustive list of projects that use Einstein-inspired notation in chronological order:

.. list-table::
   :widths: 20, 20, 60
   :header-rows: 1

   * - Name
     - First commit
     - Operations

   * - `einsum <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>`_
     - `Jan 23 2011 <https://github.com/numpy/numpy/commit/a41de3adf9dbbff9d9f2f50fe0ac59d6eabd43cf>`_
     - Dot-product, rearranging, trace/diag, sum-reduction.
   * - `einops <https://github.com/arogozhnikov/einops>`_
     - `Sep 22 2018 <https://github.com/arogozhnikov/einops/commit/8e72d792ee88dae177aba3e299179ed478b9a592>`_
     - Dot-product, rearranging, trace/diag, reduction.
   * - `einindex <https://github.com/malmaud/einindex>`_
     - `Dec 3 2018 <https://github.com/malmaud/einindex/commit/5eb212246d6dfa7061cb76545ac1cb8e41c82525>`_
     - Indexing.
   * - `einop <https://github.com/cgarciae/einop>`_
     - `Nov 21 2020 <https://github.com/arogozhnikov/einops/pull/91/commits/b959fff865a534b3f9800024558b24759f3b4002>`_
     - → *einops*
   * - `einshape <https://github.com/google-deepmind/einshape>`_
     - `Jun 22 2021 <https://github.com/google-deepmind/einshape/commit/69d853936d3401c711a723f938e6e20cf3811359>`_
     - Rearranging.
   * - `eindex <https://github.com/arogozhnikov/eindex>`_
     - `Mar 11 2023 <https://github.com/arogozhnikov/eindex/commit/b787619efd868b7f5100cd69267aa80c4a6c8621>`_
     - Indexing.
   * - `eingather <https://twitter.com/francoisfleuret/status/1661372730241953793>`_
     - `May 24 2023 <https://twitter.com/francoisfleuret/status/1661372730241953793>`_
     - Indexing.
   * - `eins <https://github.com/nicholas-miklaucic/eins>`_
     - `Mar 14 2024 <https://github.com/nicholas-miklaucic/eins/commit/dc5e9a0a3f5bf6fb9e62427b6cedf1ffab1a8873>`_
     - ?
   * - `einshard <https://github.com/ayaka14732/einshard>`_
     - `Mar 24 2024 <https://github.com/yixiaoer/mistral-v0.2-jax/commit/b800c054109a14fb04ce72ed1c990c7aa7bba628>`_
     - Sharding.
   * - `shardops <https://github.com/MatX-inc/seqax/tree/main>`_
     - `May 4 2024 <https://github.com/MatX-inc/seqax/commit/db2bd8f8492875d7d09bacfb23b4b76bd5fec220>`_
     - Sharding.

Named axes
==========

**Summary:** Tensor objects are annotated with axis names. The *lifetime* of the axis name is the combined lifetime
of all tensor objects that use it. Tensor operations are invoked by naming the axes corresponding to the elementary operation.
Vectorized axes are implicit.

A non-exhaustive list of libraries that support named axes:

* `torchdim <https://github.com/facebookresearch/torchdim>`_
* `Named axes in PyTorch <https://pytorch.org/docs/stable/named_tensor.html>`_
* `Named axes in Penzai <https://penzai.readthedocs.io/en/stable/notebooks/named_axes.html>`_
* `Named axes in xarray <https://docs.xarray.dev/en/stable/>`_

Other resources on named axes:

* `Tensor Considered Harmful <https://nlp.seas.harvard.edu/NamedTensor>`_
* `Named Tensor Notation <https://namedtensor.github.io/>`_

Other resources
===============

* `Dex <https://github.com/google-research/dex-lang>`_
* `Shape suffixes <https://medium.com/@NoamShazeer/shape-suffixes-good-coding-style-f836e72e24fd>`_

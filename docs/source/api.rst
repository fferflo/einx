########
einx API
########

Main abstractions
=================

.. autofunction:: einx.rearrange
.. autofunction:: einx.vmap_with_axis
.. autofunction:: einx.vmap
.. autofunction:: einx.dot

.. autofunction:: einx.reduce
.. autofunction:: einx.elementwise

Numpy-like functions
====================

Reduction operations
--------------------

.. autofunction:: einx.sum
.. autofunction:: einx.mean
.. autofunction:: einx.var
.. autofunction:: einx.std
.. autofunction:: einx.prod
.. autofunction:: einx.count_nonzero
.. autofunction:: einx.any
.. autofunction:: einx.all
.. autofunction:: einx.max
.. autofunction:: einx.min

Element-by-element operations
-----------------------------

.. autofunction:: einx.add
.. autofunction:: einx.subtract
.. autofunction:: einx.multiply
.. autofunction:: einx.true_divide
.. autofunction:: einx.floor_divide
.. autofunction:: einx.divide
.. autofunction:: einx.logical_and
.. autofunction:: einx.logical_or
.. autofunction:: einx.where
.. autofunction:: einx.less
.. autofunction:: einx.less_equal
.. autofunction:: einx.greater
.. autofunction:: einx.greater_equal
.. autofunction:: einx.equal
.. autofunction:: einx.not_equal
.. autofunction:: einx.maximum
.. autofunction:: einx.minimum

Miscellaneous operations
------------------------

.. autofunction:: einx.flip
.. autofunction:: einx.roll

Deep Learning Modules
=====================

Haiku
-----

.. autoclass:: einx.nn.haiku.Linear
.. autoclass:: einx.nn.haiku.Norm
.. autoclass:: einx.nn.haiku.Dropout

Flax
----

.. autofunction:: einx.nn.flax.Linear
.. autofunction:: einx.nn.flax.Norm
.. autofunction:: einx.nn.flax.Dropout

Torch
-----

.. autoclass:: einx.nn.torch.Linear
.. autoclass:: einx.nn.torch.Norm
.. autoclass:: einx.nn.torch.Dropout
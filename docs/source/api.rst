########
einx API
########

Main
----

.. autofunction:: einx.rearrange
.. autofunction:: einx.vmap_with_axis
.. autofunction:: einx.vmap
.. autofunction:: einx.reduce
.. autofunction:: einx.elementwise
.. autofunction:: einx.index

.. _apireductionops:

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
.. autofunction:: einx.logsumexp

.. _apielementwiseops:

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

.. _apiindexingops:

Indexing operations
-------------------

.. autofunction:: einx.get_at
.. autofunction:: einx.set_at
.. autofunction:: einx.add_at
.. autofunction:: einx.subtract_at

Miscellaneous operations
------------------------

.. autofunction:: einx.flip
.. autofunction:: einx.roll
.. autofunction:: einx.softmax
.. autofunction:: einx.log_softmax
.. autofunction:: einx.arange

General dot-product
-------------------

.. autofunction:: einx.dot

Deep Learning Modules
=====================

Haiku
-----

.. autoclass:: einx.nn.haiku.Linear
.. autoclass:: einx.nn.haiku.Norm
.. autoclass:: einx.nn.haiku.Dropout

.. autofunction:: einx.nn.haiku.param

Flax
----

.. autofunction:: einx.nn.flax.Linear
.. autofunction:: einx.nn.flax.Norm
.. autofunction:: einx.nn.flax.Dropout

.. autofunction:: einx.nn.flax.param

Torch
-----

.. autoclass:: einx.nn.torch.Linear
.. autoclass:: einx.nn.torch.Norm
.. autoclass:: einx.nn.torch.Dropout

.. autofunction:: einx.nn.torch.param

Equinox
-------

.. autoclass:: einx.nn.equinox.Linear
.. autoclass:: einx.nn.equinox.Norm
.. autoclass:: einx.nn.equinox.Dropout

.. autofunction:: einx.nn.equinox.param

Keras
-----

.. autoclass:: einx.nn.keras.Linear
.. autoclass:: einx.nn.keras.Norm
.. autoclass:: einx.nn.keras.Dropout

.. autofunction:: einx.nn.keras.param

Experimental
============

.. autofunction:: einx.experimental.shard
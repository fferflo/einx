########
einx API
########

Main abstractions
=================

.. autofunction:: einx.rearrange
.. autofunction:: einx.reduce
.. autofunction:: einx.elementwise
.. autofunction:: einx.dot
.. autofunction:: einx.vmap


Layers
======

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
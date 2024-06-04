How does einx support different tensor frameworks?
##################################################

einx provides interfaces for tensor frameworks in the ``einx.backend.*`` namespace. einx functions accept a ``backend`` argument
that defines which backend to use for the computation. For ``backend=None`` (the default case), the backend is implicitly determined from the input tensors.

..  code:: python

    x = np.ones((2, 3))
    einx.sum("a [b]", x, backend=einx.backend.get("numpy")) # Uses numpy backend
    einx.sum("a [b]", x)                                    # Implicitly uses numpy backend

Numpy tensors can be mixed with other frameworks in the same operation, in which case the latter backend is used for computations. Frameworks other than
Numpy cannot be mixed in the same operation.

..  code:: python

    x = np.zeros((10, 20))
    y = np.zeros((20, 30))
    einx.dot("a [c1->c2]", x, torch.from_numpy(y))              # Uses torch
    einx.dot("a [c1->c2]", x, jnp.asarray(y))                   # Uses jax
    einx.dot("a [c1->c2]", torch.from_numpy(x), jnp.asarray(y)) # Raises exception

Unkown tensor objects and python sequences are converted to tensors using calls from the respective backend if possible (e.g. ``np.asarray``, ``torch.asarray``).

..  code:: python

    x = np.zeros((10, 20))
    einx.add("a b, 1", x, [42.0])
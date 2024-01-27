import numpy as np
import einx, sys, inspect, importlib

def get_shape(x):
    if isinstance(x, (tuple, list)):
        subshapes = set(get_shape(y) for y in x)
        if len(subshapes) != 1:
            raise ValueError(f"Failed to determine shape in input tensor")
        subshape = subshapes.pop()
        if subshape is None:
            raise ValueError(f"Failed to determine shape in input tensor")
        return (len(x),) + subshape
    elif isinstance(x, (float, int, np.floating, np.integer)):
        # Scalar
        return ()

    try:
        # Concrete tensor
        return tuple(int(i) for i in x.shape)
    except:
        # Cannot determine shape (e.g. tensor factory)
        return None

def instantiate(x, shape, backend, **kwargs):
    if x is None:
        raise TypeError("instantiate cannot be called on None")
    if backend == einx.backend.tracer:
        if x.shape is None:
            return einx.backend.tracer.apply(instantiate, args=[x], kwargs={**{"shape": shape}, **kwargs}, output_shapes=shape)
        else:
            return backend.to_tensor(x)
    else:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return backend.to_tensor(x)

        for einn in einx.nn.get_frameworks():
            x2 = einn.to_tensor_factory(x)
            if not x2 is None:
                x = x2
                break

        if callable(x):
            # Try to find keyword parameters of the tensor factory and forward all kwargs that are accepted. Pass no keyword parameters if this fails.
            try:
                params = inspect.signature(x).parameters
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                    pass
                else:
                    kwargs = {k: v for k, v in kwargs.items() if k in params}
            except:
                kwargs = {}

            x = x(shape, **kwargs)
            if x is None:
                raise ValueError("Tensor factory returned None")
            if not hasattr(x, "shape"):
                raise ValueError("Tensor factory returned an object without a shape attribute")
            if x.shape != shape:
                raise ValueError(f"Tensor factory returned a tensor of shape {x.shape}, but expected {shape}")
        x = backend.to_tensor(x)

        assert x.shape == shape, f"Shape mismatch: {x.shape} != {shape} for {type(x)}"
        return x
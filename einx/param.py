import numpy as np
import einx, sys, inspect, importlib

def get_shape(x):
    try:
        # Concrete tensor
        return tuple(int(i) for i in x.shape)
    except:
        if isinstance(x, (float, int, np.floating, np.integer)):
            # Single number
            return ()
        else:
            # Tensor factory
            return None

def is_tensor_factory(x):
    if isinstance(x, einx.backend.tracer.Input):
        return x.shape is None

    for name in ["torch", "haiku", "flax", "equinox"]:
        if name in sys.modules:
            einn = importlib.import_module(f"einx.nn.{name}")
            if not einn.to_tensor_factory(x) is None:
                return True

    return callable(x)

def instantiate(x, shape, backend, **kwargs):
    if x is None:
        raise TypeError("instantiate cannot be called on None")
    if backend == einx.backend.tracer:
        if is_tensor_factory(x):
            return einx.backend.tracer.Op(instantiate, [x], {**{"shape": shape}, **kwargs}, output_shapes=np.asarray(shape), pass_backend=True).output_tracers
        else:
            return einx.backend.tracer.Op("to_tensor", [x], output_shapes=np.asarray(shape)).output_tracers
    else:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return backend.to_tensor(x)

        for name in ["torch", "haiku", "flax", "equinox"]:
            if name in sys.modules:
                einn = importlib.import_module(f"einx.nn.{name}")
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
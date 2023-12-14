import numpy as np
import einx, sys, inspect

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

def instantiate(x, shape, backend, **kwargs):
    if x is None:
        raise TypeError("instantiate cannot be called on None")
    if backend == einx.backend.tracer:
        return einx.backend.tracer.Op(instantiate, [x], {"shape": shape} | kwargs, output_shapes=np.asarray(shape), pass_backend=True).output_tracers
    else:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return backend.to_tensor(x)

        if "torch" in sys.modules:
            import torch
            if not callable(x) and isinstance(x, (torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer)) and not isinstance(x, torch._subclasses.FakeTensor):
                if backend.name != "torch":
                    raise ValueError("Cannot instantiate a torch tensor using a non-torch backend")
                def x(shape, x=x, **kwargs):
                    x.materialize(shape)
                    return x

        if callable(x):
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
            if x.shape != shape:
                raise ValueError(f"Tensor factory returned a tensor of shape {x.shape}, but expected {shape}")
        x = backend.to_tensor(x)

        assert x.shape == shape, f"Shape mismatch: {x.shape} != {shape} for {type(x)}"
        return x
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
    if backend == einx.backend.tracer:
        return einx.backend.tracer.Op(lambda x, backend: instantiate(x, shape, backend=backend, **kwargs), [x], shape=shape, pass_backend=True)
    else:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return backend.to_tensor(x)

        if "torch" in sys.modules:
            import torch
            if not callable(x) and isinstance(x, (torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer)) and not isinstance(x, torch._subclasses.FakeTensor):
                if backend != einx.backend.torch:
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
        x = backend.to_tensor(x)

        assert x.shape == shape, f"Shape mismatch: {x.shape} != {shape} for {type(x)}"
        return x
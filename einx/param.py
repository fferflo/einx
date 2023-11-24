import numpy as np
import einx, sys

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

def instantiate(x, shape, backend):
    if backend == einx.backend.tracer:
        return einx.backend.tracer.Op(lambda x, backend: instantiate(x, shape, backend=backend), [x], shape=shape, pass_backend=True)
    else:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return backend.to_tensor(x)

        done = False
        if "torch" in sys.modules:
            import torch
            if isinstance(x, (torch.nn.parameter.UninitializedParameter, torch.nn.parameter.UninitializedBuffer)) and not isinstance(x, torch._subclasses.FakeTensor):
                if backend != einx.backend.torch:
                    raise ValueError("Cannot instantiate a torch tensor using a non-torch backend")
                x.materialize(shape)
                done = True

        if not done:
            if callable(x):
                x = x(shape)
            x = backend.to_tensor(x)

        assert x.shape == shape, f"Shape mismatch: {x.shape} != {shape} for {type(x)}"
        return x
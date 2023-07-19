import sys, einx

def is_tensor_factory(x):
    if "torch" in sys.modules:
        import torch
        if isinstance(x, torch.nn.parameter.UninitializedParameter) or isinstance(x, torch.nn.parameter.UninitializedBuffer):
            return True
    return callable(x)

def get_shape(x):
    if isinstance(x, einx.op.Tensor):
        return x.value.shape
    elif is_tensor_factory(x):
        return None
    else:
        return x.shape

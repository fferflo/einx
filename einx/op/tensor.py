import einx, sys

def _instantiate(tensor, expr):
    if "torch" in sys.modules:
        import torch
        if (isinstance(tensor, torch.nn.parameter.UninitializedParameter) or isinstance(tensor, torch.nn.parameter.UninitializedBuffer)) \
                and not isinstance(tensor.data, torch._subclasses.FakeTensor):
            tensor.materialize(expr.shape)
            return tensor
    if callable(tensor):
        return tensor(shape=expr.shape)
    else:
        return tensor

class Tensor:
    def __init__(self, value, expr, backend=None):
        self.value = _instantiate(value, expr)
        self.expr = expr
        self.backend = backend if not backend is None else einx.backend.get([backend])

    @property
    def dtype(self):
        return self.value.dtype

    @property
    def shape(self):
        return self.value.shape

    def __add__(self, other):
        return einx.op.add([self, other])

    def __sub__(self, other):
        return einx.op.subtract([self, other])

    def __mul__(self, other):
        return einx.op.multiply([self, other])

    def __truediv__(self, other):
        return einx.op.true_divide([self, other])

    def __floordiv__(self, other):
        return einx.op.floor_divide([self, other])

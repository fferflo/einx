import einx._src.tracer as tracer


class NamedTensor:
    def __init__(self, value, expr):
        if not isinstance(value, tracer.signature.classical.Tensor | tracer.signature.classical.ConvertibleTensor):
            raise TypeError(f"Value must be a Tensor/ConvertibleTensor, but got {type(value)}")
        if value.shape is None:
            raise ValueError("Value must have a shape")
        if tuple(value.shape) != tuple(expr.shape):
            raise ValueError(f"Shape mismatch: {tuple(value.shape)} vs {tuple(expr.shape)}")
        self.value = value
        self.expr = expr

    @property
    def ndim(self):
        return len(self.expr.shape)

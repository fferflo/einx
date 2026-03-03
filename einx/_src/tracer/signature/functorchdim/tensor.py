import einx._src.tracer as tracer
from functools import partial


class Tensor(tracer.Tracer):
    def __init__(self, origin, shape):
        super().__init__(origin=origin)
        self.shape = shape

    @property
    def _tracer_type(self):
        return partial(Tensor, shape=self.shape)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.origin == other.origin and self.shape == other.shape
        return False

    def __hash__(self):
        return 234 + hash(self.shape)

    def dims(self):
        return tracer.signature.python.call(tracer.signature.python.getattr(self, "dims"))

    def order(self, *dims):
        return tracer.signature.python.call(tracer.signature.python.getattr(self, "order"), dims)

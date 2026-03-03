import einx._src.tracer as tracer
from functools import partial
from einx._src.util.lru_cache import _freeze_value


class ConvertibleTensor(tracer.Tracer):
    def __init__(self, origin, concrete, shape):
        tracer.Tracer.__init__(self, origin=origin)
        if not hasattr(concrete, "type"):
            raise ValueError("Concrete must have a 'type' attribute.")
        self.concrete = concrete
        self.shape = tuple(int(s) for s in shape) if shape is not None else None

    @property
    def _tracer_type(self):
        return partial(ConvertibleTensor, concrete=self.concrete, shape=self.shape)

    def __eq__(self, other):
        if isinstance(other, ConvertibleTensor):
            return self.origin == other.origin and self.concrete == other.concrete and self.shape == other.shape
        return False

    def __hash__(self):
        if self.origin is not None:
            raise ValueError("Can only hash a tracer without an origin.")
        return hash(self.shape) + hash(_freeze_value(self.concrete))

    @property
    def ndim(self):
        return len(self.shape)


class Tensor(tracer.Tracer):
    def __init__(self, origin, shape):
        super().__init__(origin=origin)
        self.shape = tuple(int(s) for s in shape)

    @property
    def _tracer_type(self):
        return partial(Tensor, shape=self.shape)

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.origin == other.origin and self.shape == other.shape
        return False

    def __hash__(self):
        return 1 + hash(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

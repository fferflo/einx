import numpy as np
import einx._src.tracer as tracer
from einx._src.util import pytree


class Optimizer:
    def __init__(self, optimizations):
        self.id_to_newobj = {}
        self.changed = False
        self.optimizations = optimizations

    def _set(self, oldobj, newobj):
        self.id_to_newobj[id(oldobj)] = newobj

    def _optimize(self, x):
        if id(x) in self.id_to_newobj:
            return self.id_to_newobj[id(x)]

        for pattern in self.optimizations:
            changed, newobj = pattern(x, self._optimize)
            if changed:
                pytree.map(self._set, x, newobj)
                self.changed = True
                return newobj

        # No pattern matched -> apply optimization to all predecessors
        if isinstance(x, str | int | float | np.integer | np.floating | np.ndarray) or x is None:
            return x
        elif isinstance(x, tracer.Graph):
            # Create inputs
            new_inputs = []
            for old_input in x.inputs:
                new_input = old_input._tracer_type(None)
                pytree.map(self._set, old_input, new_input)
                new_inputs.append(new_input)

            # Optimize to output
            new_output = self._optimize(x.output)

            return tracer.Graph(new_inputs, new_output, x.name)
        elif isinstance(x, tracer.Tracer):
            if x.origin is None:
                return x._tracer_type(None)
            else:
                new_origin = x.origin._tracer_transform(self._optimize)
                assert type(new_origin) == type(x.origin)
                assert len(new_origin.inputs) == len(x.origin.inputs)
                assert type(new_origin.output) == type(x.origin.output)

                pytree.map(self._set, x.origin.output, new_origin.output)

                return self.id_to_newobj[id(x)]
        elif isinstance(x, list):
            return [self._optimize(i) for i in x]
        elif isinstance(x, tuple):
            return tuple(self._optimize(i) for i in x)
        elif isinstance(x, dict):
            return {self._optimize(k): self._optimize(v) for k, v in x.items()}
        elif isinstance(x, slice):
            return slice(self._optimize(x.start), self._optimize(x.stop), self._optimize(x.step))
        else:
            raise NotImplementedError(f"Unsupported type: {type(x)}")


def optimize(x, optimizations):
    if len(optimizations) > 0:
        while True:
            optimizer = Optimizer(optimizations)
            x = optimizer._optimize(x)
            if not optimizer.changed:
                break
    return x

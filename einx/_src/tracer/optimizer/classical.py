import numpy as np
import einx._src.tracer as tracer
from einx._src.util import pytree
from ._util import _skip_id


class SkipReshape:
    def __init__(self, reshape):
        self.reshape = reshape

    def _is_result_of_call(self, x):
        return isinstance(x, tracer.Tracer) and isinstance(x.origin, tracer.signature.python.Call) and x.origin.function == self.reshape

    def __call__(self, x, transform):
        if self._is_result_of_call(x):
            input = x.origin.args[0]
            if isinstance(input, tracer.signature.python.Value):
                return False, None

            shape = x.origin.args[1]

            if isinstance(shape, tuple | list | np.ndarray) and tuple(shape) == tuple(input.shape):
                # Skip nop reshape
                return True, transform(input)
            input = _skip_id(input)
            if self._is_result_of_call(input):
                # Merge consecutive reshapes
                input_of_input = input.origin.args[0]
                return True, tracer.signature.python.call(transform(x.origin.function), [transform(input_of_input), shape])
        return False, None


class SkipTranspose:
    def __init__(self, transpose):
        self.transpose = transpose

    def _is_result_of_call(self, x):
        return isinstance(x, tracer.Tracer) and isinstance(x.origin, tracer.signature.python.Call) and x.origin.function == self.transpose

    def __call__(self, x, transform):
        if self._is_result_of_call(x):
            input = x.origin.args[0]
            if isinstance(input, tracer.signature.python.Value):
                return False, None

            perm = x.origin.args[1]

            if isinstance(perm, tuple | list | np.ndarray) and tuple(perm) == tuple(range(input.ndim)):
                # Skip nop transpose
                return True, transform(input)
            input = _skip_id(input)
            if self._is_result_of_call(input):
                # Merge consecutive transposes
                input_of_input = input.origin.args[0]
                perm1 = input.origin.args[1]
                perm2 = perm
                assert all(isinstance(p, int) for p in perm1) and all(isinstance(p, int) for p in perm2)
                new_perm = tuple(perm1[p] for p in perm2)
                return True, tracer.signature.python.call(transform(x.origin.function), [transform(input_of_input), new_perm])
        return False, None


class SkipConcatenate:
    def __init__(self, concatenate):
        self.concatenate = concatenate

    def _is_result_of_call(self, x):
        return isinstance(x, tracer.Tracer) and isinstance(x.origin, tracer.signature.python.Call) and x.origin.function == self.concatenate

    def __call__(self, x, transform):
        if self._is_result_of_call(x):
            tensors = x.origin.args[0]
            if isinstance(tensors, list | tuple) and len(tensors) == 1:
                # Skip concatenate
                return True, transform(tensors[0])
        return False, None


class SkipBroadcastTo:
    def __init__(self, broadcast_to):
        self.broadcast_to = broadcast_to

    def _is_result_of_call(self, x):
        return isinstance(x, tracer.Tracer) and isinstance(x.origin, tracer.signature.python.Call) and x.origin.function == self.broadcast_to

    def __call__(self, x, transform):
        if self._is_result_of_call(x):
            input = x.origin.args[0]
            if isinstance(input, tracer.signature.python.Value):
                return False, None

            shape = x.origin.args[1]

            if isinstance(shape, tuple | list | np.ndarray) and tuple(shape) == tuple(input.shape):
                # Skip broadcast_to
                return True, transform(input)
        return False, None

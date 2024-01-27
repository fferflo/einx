import einx, functools
import numpy as np

def associative_binary_to_nary(binary_op):
    @functools.wraps(binary_op)
    def nary_op(*args):
        x = args[0]
        for y in args[1:]:
            x = binary_op(x, y)
        return x
    return nary_op

class Backend:
    @classmethod
    def op(backend, op, tracable=None):
        if isinstance(op, str):
            x = backend
            for name in op.split("."):
                x = getattr(x, name)
            op = x
        return op

    @classmethod
    def apply(backend, op, args, kwargs, output_shapes):
        return backend.op(op)(*args, **kwargs)

class ErrorBackend:
    def __init__(self, message):
        self.message = message

    def __getattr__(self, name):
        raise RuntimeError(self.message)
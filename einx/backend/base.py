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

class base_backend:
    @classmethod
    def apply(backend, op, args, kwargs, output_shapes):
        if isinstance(op, str):
            x = backend
            for name in op.split("."):
                x = getattr(x, name)
            op = x
        result = op(*args, **kwargs)
        def assert_shape(tensor, out_shape):
            in_shape = np.asarray(tensor.shape)
            out_shape = np.asarray(out_shape)
            assert in_shape.shape == out_shape.shape and np.all(in_shape == out_shape), f"Expected shape {out_shape}, got {in_shape}"
        einx.tree_util.tree_map(assert_shape, result, output_shapes)
        return result
import einx
import numpy as np

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
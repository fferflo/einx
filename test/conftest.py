import importlib
import numpy as np
import types
import einx

tests = []


class WrappedEinx:
    def __init__(self, wrap, name, inline_args):
        import einx

        self.einx = einx
        self.in_func = False
        self.wrap = wrap
        self.name = name
        self.inline_args = inline_args

    def __enter__(self):
        assert not self.in_func
        self.in_func = True

    def __exit__(self, *args):
        assert self.in_func
        self.in_func = False

    def __getattr__(self, attr):
        op = getattr(self.einx, attr)
        if self.in_func or attr in {"matches", "solve", "check", "trace"}:
            return op

        if self.inline_args:

            def op3(*args, **kwargs):
                with self:

                    def op2():
                        return op(*args, **kwargs)

                    op2 = self.wrap(op2)
                    op2()
                    return op2()

            return op3
        else:

            def op3(*args, **kwargs):
                with self:
                    op2 = self.wrap(op)
                    op2(*args, **kwargs)
                    return op2(*args, **kwargs)

            return op3


if importlib.util.find_spec("jax"):
    import jax
    import jax.numpy as jnp

    einx_jit = WrappedEinx(jax.jit, "jax.jit", inline_args=True)

    backend = einx.backend.jax.create()

    test_cpu = types.SimpleNamespace(
        full=lambda shape, value=0.0, dtype="float32": jax.device_put(
            jnp.full(shape, value, dtype=dtype), device=jax.devices("cpu")[0]
        ),
        to_tensor=lambda x: jax.device_put(jnp.asarray(x), device=jax.devices("cpu")[0]),
        to_numpy=np.asarray,
    )

    tests.append((einx, backend, test_cpu))
    tests.append((einx_jit, backend, test_cpu))

    try:
        jax.devices("gpu")
        has_gpu = True
    except:
        has_gpu = False

    if has_gpu:
        test_gpu = types.SimpleNamespace(
            full=lambda shape, value=0.0, dtype="float32": jax.device_put(
                jnp.full(shape, value, dtype=dtype), device=jax.devices("gpu")[0]
            ),
            to_tensor=lambda x: jax.device_put(jnp.asarray(x), device=jax.devices("gpu")[0]),
            to_numpy=np.asarray,
        )

        tests.append((einx, backend, test_gpu))
        tests.append((einx_jit, backend, test_gpu))

if importlib.util.find_spec("torch"):
    import torch

    version = tuple(int(i) for i in torch.__version__.split(".")[:2])

    def wrap(op):
        torch.compiler.reset()
        return torch.compile(op)

    einx_torchcompile = WrappedEinx(wrap, "torch.compile", inline_args=False)

    backend = einx.backend.torch.create()

    dtypes = {
        "float32": torch.float32,
        "long": torch.long,
        "bool": torch.bool,
    }

    test_cpu = types.SimpleNamespace(
        full=lambda shape, value=0.0, dtype="float32", backend=backend: torch.full(
            backend.to_tuple(shape), value, dtype=dtypes[dtype]
        ),
        to_tensor=lambda tensor: torch.asarray(tensor, device=torch.device("cpu")),
        to_numpy=lambda tensor: tensor.numpy(),
    )

    tests.append((einx, backend, test_cpu))
    if version >= (2, 1):
        tests.append((einx_torchcompile, backend, test_cpu))

    if torch.cuda.is_available():
        test_gpu = types.SimpleNamespace(
            full=lambda shape, value=1.0, dtype="float32", backend=backend: torch.full(
                backend.to_tuple(shape), value, dtype=dtypes[dtype], device=torch.device("cuda")
            ),
            to_tensor=lambda tensor: torch.asarray(tensor, device=torch.device("cuda")),
            to_numpy=lambda tensor: tensor.cpu().numpy(),
        )

        tests.append((einx, backend, test_gpu))
        if version >= (2, 1):
            tests.append((einx_torchcompile, backend, test_gpu))

if importlib.util.find_spec("tensorflow"):
    import os

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp

    tnp.experimental_enable_numpy_behavior()

    backend = einx.backend.tensorflow.create()

    test = types.SimpleNamespace(
        full=lambda shape, value=0.0, dtype="float32": tnp.full(shape, value, dtype=dtype),
        to_tensor=tf.convert_to_tensor,
        to_numpy=lambda x: x.numpy(),
    )

    tests.append((einx, backend, test))

if importlib.util.find_spec("mlx"):
    import mlx.core as mx

    backend = einx.backend.mlx.create()

    test = types.SimpleNamespace(
        full=lambda shape, value=0.0, dtype="float32", backend=backend: mx.full(
            shape, value, dtype=backend.to_dtype(dtype)
        ),
        to_tensor=mx.array,
        to_numpy=np.asarray,
    )

    tests.append((einx, backend, test))

if importlib.util.find_spec("dask"):
    import dask.array as da

    backend = einx.backend.dask.create()

    test = types.SimpleNamespace(
        full=lambda shape, value=0.0, dtype="float32": da.full(shape, value, dtype=dtype),
        to_tensor=np.asarray,
        to_numpy=np.asarray,
    )

    tests.append((einx, backend, test))

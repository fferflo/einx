import importlib
import numpy as np
import types
import einx
import threading
import multiprocessing
import os
import pytest


class DYNAMIC:
    pass


def default_is_static_arg(x):
    return "shape" not in dir(x)


def default_is_static_kwargs(k, v):
    return True


class WrappedEinx:
    def __init__(
        self,
        wrap,
        name,
        is_static_arg=default_is_static_arg,
        is_static_kwarg=default_is_static_kwargs,
    ):
        import einx

        self.einx = einx
        self.in_func = False
        self.wrap = wrap
        self.name = name
        self.is_static_arg = is_static_arg
        self.is_static_kwarg = is_static_kwarg

    def __enter__(self):
        assert not self.in_func
        self.in_func = True

    def __exit__(self, *args):
        assert self.in_func
        self.in_func = False

    def __getattr__(self, attr):
        op_no = getattr(self.einx, attr)
        if self.in_func or attr in {"matches", "solve", "check", "trace"}:
            return op_no

        def op_all(*all_args, **all_kwargs):
            with self:
                dynamic_args = [a for a in all_args if not self.is_static_arg(a)]
                dynamic_kwargs = {
                    k: v for k, v in all_kwargs.items() if not self.is_static_kwarg(k, v)
                }
                all_args = [a if self.is_static_arg(a) else DYNAMIC for a in all_args]
                all_kwargs = {
                    k: v if self.is_static_kwarg(k, v) else DYNAMIC for k, v in all_kwargs.items()
                }

                def op_some(*dynamic_args, **dynamic_kwargs):
                    dynamic_args = list(dynamic_args)
                    all_args2 = [dynamic_args.pop(0) if a == DYNAMIC else a for a in all_args]
                    all_kwargs2 = {**all_kwargs, **dynamic_kwargs}

                    op_no(*all_args2, **all_kwargs2)
                    return op_no(*all_args2, **all_kwargs2)

                op_some = self.wrap(op_some)

                op_some(*dynamic_args, **dynamic_kwargs)  # TODO: why twice?
                return op_some(*dynamic_args, **dynamic_kwargs)

        return op_all


def in_new_thread(op):
    def inner(*args, **kwargs):
        result = [None, None]

        def run(result):
            try:
                result[0] = op(*args, **kwargs)
            except Exception as e:
                result[1] = e

        thread = threading.Thread(target=run, args=(result,))
        thread.start()
        thread.join()
        if result[1] is not None:
            raise result[1]
        else:
            return result[0]

    return inner


einx_multithread = WrappedEinx(in_new_thread, "multithreading")


def in_new_process(op):
    def inner(*args, **kwargs):
        result = multiprocessing.Queue()
        exception = multiprocessing.Queue()

        def run(result, exception):
            try:
                result.put(op(*args, **kwargs))
            except Exception as e:
                exception.put(e)

        process = multiprocessing.Process(target=run, args=(result, exception))
        process.start()
        process.join()
        if not exception.empty():
            raise exception.get()
        else:
            return result.get()

    return inner


einx_multiprocess = WrappedEinx(in_new_process, "multiprocessing")


test_configs = {}


def numpy_config():
    tests = []

    backend = einx.backend.numpy.create()

    test = types.SimpleNamespace(
        full=lambda shape, value=0.0, dtype="float32": np.full(shape, value, dtype=dtype),
        to_tensor=np.asarray,
        to_numpy=np.asarray,
    )

    tests.append((einx, backend, test))
    tests.append((einx_multithread, backend, test))
    # tests.append((einx_multiprocess, backend, test)) # too slow

    return tests


test_configs["numpy"] = numpy_config


def jax_config():
    tests = []
    if importlib.util.find_spec("jax"):
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
        )

        import jax
        import jax.numpy as jnp

        einx_jit = WrappedEinx(jax.jit, "jax.jit")

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

    return tests


test_configs["jax"] = jax_config


def torch_config():
    tests = []
    if importlib.util.find_spec("torch"):
        import torch

        version = tuple(int(i) for i in torch.__version__.split(".")[:2])

        def wrap(op):
            torch.compiler.reset()
            return torch.compile(op)

        einx_torchcompile = WrappedEinx(wrap, "torch.compile")

        def wrap(op):
            torch.compiler.reset()
            return torch.compile(op, dynamic=False)

        einx_torchstaticcompile = WrappedEinx(wrap, "torch.compilestatic")

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
            tests.append((einx_torchstaticcompile, backend, test_cpu))
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
                tests.append((einx_torchstaticcompile, backend, test_gpu))
                tests.append((einx_torchcompile, backend, test_gpu))

    return tests


test_configs["torch"] = torch_config


def tensorflow_config():
    tests = []
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

    return tests


test_configs["tensorflow"] = tensorflow_config


def mlx_config():
    tests = []
    if importlib.util.find_spec("mlx"):
        import mlx.core as mx

        backend = einx.backend.mlx.create()

        einx_compile = WrappedEinx(mx.compile, "mx.compile")

        test = types.SimpleNamespace(
            full=lambda shape, value=0, dtype="float32", backend=backend: mx.full(
                shape, value, dtype=backend.to_dtype(dtype)
            ),
            to_tensor=mx.array,
            to_numpy=np.asarray,
        )

        tests.append((einx, backend, test))
        tests.append((einx_compile, backend, test))

    return tests


test_configs["mlx"] = mlx_config


def dask_config():
    tests = []
    if importlib.util.find_spec("dask"):
        import dask.array as da

        backend = einx.backend.dask.create()

        test = types.SimpleNamespace(
            full=lambda shape, value=0.0, dtype="float32": da.full(shape, value, dtype=dtype),
            to_tensor=np.asarray,
            to_numpy=np.asarray,
        )

        tests.append((einx, backend, test))

    return tests


test_configs["dask"] = dask_config


def tinygrad_config():
    tests = []
    if importlib.util.find_spec("tinygrad"):
        import os

        os.environ["PYTHON"] = "1"
        from tinygrad import Tensor

        backend = einx.backend.tinygrad.create()

        test = types.SimpleNamespace(
            full=lambda shape, value=0.0, dtype="float32": Tensor.full(
                shape, value, dtype=backend.to_dtype(dtype)
            ),
            to_tensor=Tensor,
            to_numpy=lambda x: x.numpy(),
        )

        tests.append((einx, backend, test))

    return tests


test_configs["tinygrad"] = tinygrad_config


def pytest_addoption(parser):
    parser.addoption("--backend", type=str, default="all")


def pytest_generate_tests(metafunc):
    if "test" in metafunc.fixturenames:
        backend = metafunc.config.getoption("backend")
        if backend == "all":
            tests = []
            for config in test_configs.values():
                tests.extend(config())
        else:
            tokens = backend.split(".")
            if tokens[0] not in test_configs:
                raise ValueError(f"Unknown backend: {tokens[0]}")
            tests = test_configs[tokens[0]]()
            if len(tokens) > 1:
                mode = ".".join(tokens[1:])
                tests = [t for t in tests if getattr(t[0], "name", "") == mode]
                if len(tests) == 0:
                    raise ValueError(f"Unknown backend: {backend}")

        metafunc.parametrize("test", tests)


def pytest_collection_modifyitems(config, items):
    backend = config.getoption("backend")
    if backend != "all":
        if backend not in test_configs:
            raise ValueError(f"Unknown backend: {tokens[0]}")
        skip = pytest.mark.skip(reason="backend is disabled")
        for item in items:
            if "all" not in item.keywords and backend not in item.name:
                item.add_marker(skip)


def pytest_configure(config):
    for name in ["all", "numpy", "jax", "torch", "tensorflow", "mlx", "dask", "tinygrad"]:
        config.addinivalue_line(
            "markers", f"{name}: mark test to run with {name} backend{'s' if name == 'all' else ''}"
        )

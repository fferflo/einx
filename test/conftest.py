import importlib
import numpy as np
import types
import einx
import threading
import multiprocessing
import os
import pytest
from functools import partial
import einx._src.adapter as adapter
from collections import defaultdict
import functools


def use_backend(func):
    @functools.wraps(func)
    def wrapper(setup_backend):
        if setup_backend.backend is not None:
            with setup_backend.backend:
                return func(setup_backend)
        else:
            return func(setup_backend)

    return wrapper


def assert_allclose(x, y, setup):
    if isinstance(x, list | int | float | tuple | np.ndarray):
        x = np.asarray(x)
    else:
        x = setup.to_numpy(x)
    if isinstance(y, list | int | float | tuple | np.ndarray):
        y = np.asarray(y)
    else:
        y = setup.to_numpy(y)

    assert x.shape == y.shape
    if x.dtype.kind in "f":
        assert np.allclose(x, y, rtol=1e-3)
    else:
        assert np.all(x == y)


def to_version(x):
    x = x.split("+")[0].split(".")
    return tuple(int(i) for i in x)


class DYNAMIC:
    pass


def default_is_static_arg(x):
    return "shape" not in dir(x) or isinstance(x, np.ndarray)


def default_is_static_kwargs(k, v):
    return True


def wrap_einx_function(op_no, wrap, is_static_arg=default_is_static_arg, is_static_kwarg=default_is_static_kwargs):
    def op_all(*all_args, **all_kwargs):
        dynamic_args = [a for a in all_args if not is_static_arg(a)]
        dynamic_kwargs = {k: v for k, v in all_kwargs.items() if not is_static_kwarg(k, v)}
        all_args = [a if is_static_arg(a) else DYNAMIC for a in all_args]
        all_kwargs = {k: v if is_static_kwarg(k, v) else DYNAMIC for k, v in all_kwargs.items()}

        def op_some(*dynamic_args, **dynamic_kwargs):
            dynamic_args = list(dynamic_args)
            all_args2 = [dynamic_args.pop(0) if (isinstance(a, type) and a == DYNAMIC) else a for a in all_args]
            all_kwargs2 = {**all_kwargs, **dynamic_kwargs}

            return op_no(*all_args2, **all_kwargs2)

        op_some = wrap(op_some)

        return op_some(*dynamic_args, **dynamic_kwargs)

    return op_all


class WrappedEinx:
    def __init__(self, wrap):
        import einx

        self.einx = einx
        self.wrap = partial(wrap_einx_function, wrap=wrap)

    def __getattr__(self, attr):
        return self.wrap(getattr(self.einx, attr))


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


einx_multithread = WrappedEinx(in_new_thread)


# def in_new_process(op):
#     def inner(*args, **kwargs):
#         result = multiprocessing.Queue()
#         exception = multiprocessing.Queue()

#         def run(result, exception):
#             try:
#                 result.put(op(*args, **kwargs))
#             except Exception as e:
#                 exception.put(e)

#         process = multiprocessing.Process(target=run, args=(result, exception))
#         process.start()
#         process.join()
#         if not exception.empty():
#             raise exception.get()
#         else:
#             return result.get()

#     return inner


# einx_multiprocess = WrappedEinx(in_new_process)


setup_backend = []
setup_adapt = []


def arrayapi_is_available():
    if not importlib.util.find_spec("array_api_compat"):
        return False
    try:
        import array_api_compat

        return True
    except ImportError:
        return False


numpy_backends = ["numpy", "numpy.numpylike", "numpy.einsum", None]
if arrayapi_is_available():
    numpy_backends.extend(["arrayapi.numpylike", "arrayapi.einsum", "arrayapi"])
setup = types.SimpleNamespace(
    full=lambda shape, value=1.0, dtype="float32": np.full(shape, value, dtype=dtype),
    to_tensor=np.asarray,
    to_numpy=np.asarray,
    exceptions=(),
    classical=adapter.classical_from_numpy.ops(np),
    dtypes=types.SimpleNamespace(int="int64", float="float64"),
    version=to_version(np.__version__),
)

for backend_name in numpy_backends:
    backend = einx.backend.get(backend_name) if backend_name is not None else None

    setup_backend.append(types.SimpleNamespace(name=f"numpy.{backend_name}", **setup.__dict__, einx=einx, backend=backend))
    setup_backend.append(types.SimpleNamespace(name=f"numpy.{backend_name}.in_new_thread", **setup.__dict__, einx=einx_multithread, backend=backend))

setup_adapt.append(
    types.SimpleNamespace(
        name="numpy",
        **setup.__dict__,
        wrap=lambda x: x,
        adapt_numpylike_elementwise=einx.numpy.adapt_numpylike_elementwise,
        adapt_numpylike_reduce=einx.numpy.adapt_numpylike_reduce,
    )
)


def jax_is_available():
    if not importlib.util.find_spec("jax"):
        return False
    try:
        import jax
        import jax.numpy as jnp

        return True
    except ImportError:
        return False


jax_backends = ["jax.numpylike", "jax.vmap", "jax.einsum", "jax", None]
if arrayapi_is_available():
    jax_backends.extend(["arrayapi.numpylike", "arrayapi.einsum", "arrayapi"])
if jax_is_available():
    import jax
    import jax.numpy as jnp

    classical = adapter.classical_from_jax.ops(jax)
    try:
        jax.devices("gpu")
        has_gpu = True
    except:
        has_gpu = False

    def make_setup(device):
        return types.SimpleNamespace(
            full=lambda shape, value=1.0, dtype="float32": jax.device_put(jnp.full(shape, value, dtype=dtype), device=jax.devices(device)[0]),
            to_tensor=lambda x: jax.device_put(jnp.asarray(x), device=jax.devices(device)[0]),
            to_numpy=np.asarray,
            exceptions=(),
            classical=classical,
            dtypes=types.SimpleNamespace(int="int32", float="float32"),
            version=to_version(jax.__version__),
        )

    setup_cpu = make_setup("cpu")
    if has_gpu:
        setup_gpu = make_setup("gpu")

    einx_jit = WrappedEinx(jax.jit)

    for backend_name in jax_backends:
        backend = einx.backend.get(backend_name) if backend_name is not None else None

        setup_backend.append(types.SimpleNamespace(name=f"jax.{backend_name}.cpu", **setup_cpu.__dict__, backend=backend, einx=einx))
        setup_backend.append(types.SimpleNamespace(name=f"jax.{backend_name}.cpu.jit", **setup_cpu.__dict__, einx=einx_jit, backend=backend))

        if has_gpu:
            setup_backend.append(types.SimpleNamespace(name=f"jax.{backend_name}.gpu", **setup_gpu.__dict__, einx=einx, backend=backend))
            setup_backend.append(types.SimpleNamespace(name=f"jax.{backend_name}.gpu.jit", **setup_gpu.__dict__, einx=einx_jit, backend=backend))

    for wrap in [lambda x: x, jax.jit]:
        setup_adapt.append(
            types.SimpleNamespace(
                name="jax.cpu",
                **setup_cpu.__dict__,
                wrap=partial(wrap_einx_function, wrap=wrap),
                adapt_with_vmap=einx.jax.adapt_with_vmap,
                adapt_numpylike_elementwise=einx.jax.adapt_numpylike_elementwise,
                adapt_numpylike_reduce=einx.jax.adapt_numpylike_reduce,
            )
        )
        if has_gpu:
            setup_adapt.append(
                types.SimpleNamespace(
                    name="jax.gpu",
                    **setup_gpu.__dict__,
                    wrap=partial(wrap_einx_function, wrap=wrap),
                    adapt_with_vmap=einx.jax.adapt_with_vmap,
                    adapt_numpylike_elementwise=einx.jax.adapt_numpylike_elementwise,
                    adapt_numpylike_reduce=einx.jax.adapt_numpylike_reduce,
                )
            )


def torch_is_available():
    if not importlib.util.find_spec("torch"):
        return False
    try:
        import torch

        return True
    except ImportError:
        return False


torch_backends = ["torch.numpylike", "torch.vmap", "torch.einsum", "torch", None]
if torch_is_available():
    import torch

    has_gpu = torch.cuda.is_available()
    version = tuple(int(i) for i in torch.__version__.split(".")[:2])

    torch_dtypes = {"float32": torch.float32, "int64": torch.int64, "bool": torch.bool}

    def make_setup(device):
        exceptions = []
        try:
            exceptions.append(torch._dynamo.exc.TorchRuntimeError)
        except:
            pass
        try:
            exceptions.append(torch._inductor.exc.InductorError)
        except:
            pass
        return types.SimpleNamespace(
            full=lambda shape, value=1.0, dtype="float32": torch.full(tuple(shape), value, dtype=torch_dtypes[dtype], device=torch.device(device)),
            to_tensor=lambda tensor: torch.asarray(tensor, device=torch.device(device)),
            to_numpy=lambda tensor: tensor.numpy(),
            exceptions=tuple(exceptions),
            classical=adapter.classical_from_torch.ops(torch, get_device=lambda: torch.device(device)),
            dtypes=types.SimpleNamespace(int="int64", float="float32"),
            version=to_version(torch.__version__),
        )

    setup_cpu = make_setup("cpu")
    if has_gpu:
        setup_gpu = make_setup("cuda")

    def wrap_torchcompile(op):
        torch.compiler.reset()
        return torch.compile(op)

    einx_torchcompile = WrappedEinx(wrap_torchcompile)

    def wrap_torchstaticcompile(op):
        torch.compiler.reset()
        return torch.compile(op, dynamic=False)

    einx_torchstaticcompile = WrappedEinx(wrap_torchstaticcompile)

    for backend_name in torch_backends:
        backend = einx.backend.get(backend_name) if backend_name is not None else None
        setup_backend.append(types.SimpleNamespace(name=f"torch.{backend_name}.cpu", **setup_cpu.__dict__, backend=backend, einx=einx))
        setup_backend.append(
            types.SimpleNamespace(name=f"torch.{backend_name}.cpu.compile(dynamic=False)", **setup_cpu.__dict__, backend=backend, einx=einx_torchstaticcompile)
        )
        setup_backend.append(types.SimpleNamespace(name=f"torch.{backend_name}.cpu.compile", **setup_cpu.__dict__, backend=backend, einx=einx_torchcompile))

        if has_gpu:
            setup_backend.append(types.SimpleNamespace(name=f"torch.{backend_name}.gpu", **setup_gpu.__dict__, backend=backend, einx=einx))
            setup_backend.append(
                types.SimpleNamespace(
                    name=f"torch.{backend_name}.gpu.compile(dynamic=False)", **setup_gpu.__dict__, backend=backend, einx=einx_torchstaticcompile
                )
            )
            setup_backend.append(types.SimpleNamespace(name=f"torch.{backend_name}.gpu.compile", **setup_gpu.__dict__, backend=backend, einx=einx_torchcompile))

    wraps = [(lambda x: x, ""), (wrap_torchcompile, ".compile"), (wrap_torchstaticcompile, ".compile(dynamic=False)")]
    for wrap_fn, wrap_name in wraps:
        setup_adapt.append(
            types.SimpleNamespace(
                name=f"torch.cpu{wrap_name}",
                **setup_cpu.__dict__,
                wrap=partial(wrap_einx_function, wrap=wrap_fn),
                adapt_with_vmap=einx.torch.adapt_with_vmap,
                adapt_numpylike_elementwise=einx.torch.adapt_numpylike_elementwise,
                adapt_numpylike_reduce=einx.torch.adapt_numpylike_reduce,
            )
        )
        if has_gpu:
            setup_adapt.append(
                types.SimpleNamespace(
                    name=f"torch.gpu{wrap_name}",
                    **setup_gpu.__dict__,
                    wrap=partial(wrap_einx_function, wrap=wrap_fn),
                    adapt_with_vmap=einx.torch.adapt_with_vmap,
                    adapt_numpylike_elementwise=einx.torch.adapt_numpylike_elementwise,
                    adapt_numpylike_reduce=einx.torch.adapt_numpylike_reduce,
                )
            )


def mlx_is_available():
    if not importlib.util.find_spec("mlx"):
        return False
    try:
        import mlx.core

        return True
    except ImportError:
        return False


mlx_backends = ["mlx.numpylike", "mlx.vmap", "mlx.einsum", "mlx", None]
if mlx_is_available():
    import mlx.core as mx
    import mlx.nn
    import mlx

    classical = adapter.classical_from_mlx.ops(mlx)

    mlx_dtypes = {"float32": mx.float32, "int64": mx.int64, "bool": mx.bool_}
    setup = types.SimpleNamespace(
        full=lambda shape, value=1, dtype="float32": mx.full(shape, value, dtype=mlx_dtypes[dtype]),
        to_tensor=mx.array,
        to_numpy=np.asarray,
        exceptions=(),
        classical=classical,
        dtypes=types.SimpleNamespace(int="int32", float="float32"),
        version=to_version(mx.__version__),
    )

    einx_compile = WrappedEinx(mx.compile)

    for backend_name in mlx_backends:
        backend = einx.backend.get(backend_name) if backend_name is not None else None

        setup_backend.append(types.SimpleNamespace(name=f"mlx.{backend_name}", **setup.__dict__, backend=backend, einx=einx))
        setup_backend.append(types.SimpleNamespace(name=f"mlx.{backend_name}.compile", **setup.__dict__, backend=backend, einx=einx_compile))

    wraps = [(lambda x: x, ""), (mx.compile, ".compile")]
    for wrap_fn, wrap_name in wraps:
        setup_adapt.append(
            types.SimpleNamespace(
                name=f"mlx{wrap_name}",
                **setup.__dict__,
                wrap=partial(wrap_einx_function, wrap=wrap_fn),
                adapt_with_vmap=einx.mlx.adapt_with_vmap,
                adapt_numpylike_elementwise=einx.mlx.adapt_numpylike_elementwise,
                adapt_numpylike_reduce=einx.mlx.adapt_numpylike_reduce,
            )
        )


def tf_is_available():
    if not importlib.util.find_spec("tensorflow"):
        return False
    try:
        import tensorflow
        import tensorflow.experimental.numpy

        return True
    except ImportError:
        return False


tf_backends = ["tensorflow.numpylike", "tensorflow.einsum", "tensorflow", None]
if tf_is_available():
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    import tensorflow.experimental.numpy as tnp

    classical = adapter.classical_from_tensorflow.ops(tf)

    setup = types.SimpleNamespace(
        full=lambda shape, value=1, dtype="float32": tnp.full(shape, value, dtype=dtype),
        to_tensor=tf.convert_to_tensor,
        to_numpy=lambda x: x.numpy(),
        exceptions=(),
        classical=classical,
        dtypes=types.SimpleNamespace(int="int32", float="float32"),
        version=to_version(mx.__version__),
    )

    for backend_name in tf_backends:
        backend = einx.backend.get(backend_name) if backend_name is not None else None

        setup_backend.append(types.SimpleNamespace(name=f"tensorflow.{backend_name}", **setup.__dict__, backend=backend, einx=einx))

    wraps = [(lambda x: x, "")]
    for wrap_fn, wrap_name in wraps:
        setup_adapt.append(
            types.SimpleNamespace(
                name=f"tensorflow{wrap_name}",
                **setup.__dict__,
                wrap=partial(wrap_einx_function, wrap=wrap_fn),
                adapt_numpylike_elementwise=einx.tensorflow.adapt_numpylike_elementwise,
                adapt_numpylike_reduce=einx.tensorflow.adapt_numpylike_reduce,
            )
        )


def dask_is_available():
    if not importlib.util.find_spec("dask"):
        return False
    try:
        import dask.array

        return True
    except ImportError:
        return False


dask_backends = ["arrayapi.numpylike", "arrayapi.einsum", "arrayapi", None]
if dask_is_available() and arrayapi_is_available():
    import dask
    import dask.array as da
    import array_api_compat.dask.array as aac_da

    classical = adapter.classical_from_arrayapi.ops(aac_da)

    setup = types.SimpleNamespace(
        full=lambda shape, value=1, dtype="float32": da.full(shape, value, dtype=dtype),
        to_tensor=da.asarray,
        to_numpy=np.asarray,
        exceptions=(),
        classical=classical,
        dtypes=types.SimpleNamespace(int="int32", float="float32"),
        version=to_version(dask.__version__),
    )

    for backend_name in dask_backends:
        backend = einx.backend.get(backend_name) if backend_name is not None else None

        setup_backend.append(types.SimpleNamespace(name=f"dask.{backend_name}", **setup.__dict__, backend=backend, einx=einx))

    warps = [(lambda x: x, "")]
    for wrap_fn, wrap_name in warps:
        setup_adapt.append(
            types.SimpleNamespace(
                name=f"dask{wrap_name}",
                **setup.__dict__,
                wrap=partial(wrap_einx_function, wrap=wrap_fn),
                adapt_numpylike_elementwise=einx.arrayapi.adapt_numpylike_elementwise,
                adapt_numpylike_reduce=einx.arrayapi.adapt_numpylike_reduce,
            )
        )


def tinygrad_is_available():
    if not importlib.util.find_spec("tinygrad"):
        return False
    try:
        import tinygrad

        return True
    except ImportError:
        return False


tinygrad_backends = ["tinygrad.numpylike", "tinygrad.einsum", "tinygrad", None]
if tinygrad_is_available():
    os.environ["PYTHON"] = "1"
    import tinygrad

    classical = adapter.classical_from_tinygrad.ops(tinygrad)

    setup = types.SimpleNamespace(
        full=lambda shape, value=1, dtype="float32": tinygrad.Tensor.cast(tinygrad.Tensor.full(shape, value), dtype),
        to_tensor=tinygrad.Tensor,
        to_numpy=lambda x: x.numpy(),
        exceptions=(),
        classical=classical,
        dtypes=types.SimpleNamespace(int="int32", float="float32"),
        version=to_version("0.0.0"),
    )

    for backend_name in tinygrad_backends:
        backend = einx.backend.get(backend_name) if backend_name is not None else None

        setup_backend.append(types.SimpleNamespace(name=f"tinygrad.{backend_name}", **setup.__dict__, backend=backend, einx=einx))

    wraps = [(lambda x: x, "")]
    for wrap_fn, wrap_name in wraps:
        setup_adapt.append(
            types.SimpleNamespace(
                name=f"tinygrad{wrap_name}",
                **setup.__dict__,
                wrap=partial(wrap_einx_function, wrap=wrap_fn),
                adapt_numpylike_elementwise=einx.tinygrad.adapt_numpylike_elementwise,
                adapt_numpylike_reduce=einx.tinygrad.adapt_numpylike_reduce,
            )
        )


def pytest_addoption(parser):
    parser.addoption("--backend", type=str, default=None)
    parser.addoption("--disable-mlx-compile-values", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    backend = config.getoption("backend")

    if backend is not None:
        # If --backend is provided, only run tests with fixture "setup_backend"
        skip_marker = pytest.mark.skip(reason="Only running tests for specified backend")
        for item in items:
            if "setup_backend" not in item.fixturenames:
                item.add_marker(skip_marker)


def pytest_runtest_setup(item):
    # if --disable-mlx-compile-values is provided, skip all tests marked with @pytest.mark.computes_values and using mlx backend with compile wrapper
    if item.get_closest_marker("computes_values") and item.config.getoption("--disable-mlx-compile-values") and "setup_backend" in item.fixturenames:
        setup = item.callspec.params.get("setup_backend")
        if setup and "mlx" in setup.name and "compile" in setup.name:
            pytest.skip("Skipped due to --disable-mlx-compile-values")


def pytest_generate_tests(metafunc):
    backend = metafunc.config.getoption("backend")

    if "setup_backend" in metafunc.fixturenames:
        if backend is None:
            # Run for all backends
            tests = setup_backend
        else:
            # Run only for the specified backend
            tests = []
            for setup in setup_backend:
                if (setup.backend is None and backend == "default") or (setup.backend is not None and setup.backend.name == backend):
                    tests.append(setup)

            if len(tests) == 0:
                raise ValueError(f"Backend '{backend}' is not available")

        metafunc.parametrize("setup_backend", tests)

    if "setup_adapt" in metafunc.fixturenames:
        tests = setup_adapt

        metafunc.parametrize("setup_adapt", tests)

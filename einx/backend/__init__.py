import sys
import einx
import threading

from .base import Backend

backends = []
backend_factories = {}
lock = threading.Lock()

from ._numpy import numpy

backends.append(numpy)

from ._jax import make_jax_backend

backend_factories["jax"] = make_jax_backend

from ._torch import make_torch_backend

backend_factories["torch"] = make_torch_backend

from ._tensorflow import make_tensorflow_backend

backend_factories["tensorflow"] = make_tensorflow_backend

from ._mlx import make_mlx_backend

backend_factories["mlx"] = make_mlx_backend

from ._dask import make_dask_backend

backend_factories["dask"] = make_dask_backend

from .tracer import tracer

backends.append(tracer)


def _update():
    for backend_name in list(backend_factories.keys()):
        if backend_name in sys.modules:
            backend = backend_factories[backend_name]()
            backends.append(backend)
            globals()[backend_name] = backend
            del backend_factories[backend_name]


_update()


type_to_backend = {}


def _get1(tensor):
    tensor_backend = type_to_backend.get(type(tensor), None)
    if tensor_backend is None:
        _update()

        if tensor_backend is None:
            for tensor_backend in backends:
                if isinstance(tensor, tensor_backend.tensor) and not isinstance(
                    tensor, numpy.tensor
                ):
                    break
            else:
                # Default backend is numpy
                tensor_backend = numpy

        type_to_backend[type(tensor)] = tensor_backend
    return tensor_backend


def get(arg):
    with lock:
        if isinstance(arg, str):
            name = arg
            for backend in backends:
                if backend.name == name:
                    return backend
            _update()
            for backend in backends:
                if backend.name == name:
                    return backend
            raise ValueError(f"Backend {name} not found")
        else:
            tensors = arg
            if len(tensors) == 1:
                return _get1(tensors[0])
            backend = None
            for tensor in tensors:
                if tensor is not None:
                    backend2 = _get1(tensor)
                    if backend2 != numpy:
                        if backend is not None and backend != backend2:
                            raise ValueError(
                                "Got tensors with conflicting backends: "
                                f"{backend.__name__} and {backend2.__name__}"
                            )
                        backend = backend2
            if backend is None:
                return numpy
            else:
                return backend

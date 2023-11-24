import sys, einx

backends = []
backend_factories = {}

from ._numpy import numpy
backends.append(numpy)

from ._jax import make_jax_backend
backend_factories["jax"] = make_jax_backend

from ._torch import make_torch_backend
backend_factories["torch"] = make_torch_backend

from ._tensorflow import make_tensorflow_backend
backend_factories["tensorflow"] = make_tensorflow_backend

from .tracer import tracer
backends.append(tracer)

def update():
    for backend_name in list(backend_factories.keys()):
        if backend_name in sys.modules:
            backends.append(backend_factories[backend_name]())
            del backend_factories[backend_name]
update()


type_to_backend = {}

def _get1(tensor):
    tensor_backend = type_to_backend.get(type(tensor), None)
    if tensor_backend is None:
        update()

        if tensor_backend is None:
            for tensor_backend in backends:
                if isinstance(tensor, tensor_backend.tensor) and not isinstance(tensor, numpy.tensor):
                    break
            else:
                # Default backend is numpy
                tensor_backend = numpy

        type_to_backend[type(tensor)] = tensor_backend
    return tensor_backend

def get(tensors):
    if len(tensors) == 1:
        return _get1(tensors[0])
    backend = None
    for tensor in tensors:
        backend2 = _get1(tensor)
        if backend2 != numpy:
            if not backend is None and backend != backend2:
                raise ValueError(f"Got tensors with conflicting backends: {backend.__name__} and {backend2.__name__}")
            backend = backend2
    if backend is None:
        return numpy
    else:
        return backend

def get_by_name(name):
    update()
    for backend in backends:
        if backend.name == name:
            return backend
    raise ValueError(f"Backend {name} not found")

def __getattr__(name):
    return get_by_name(name)
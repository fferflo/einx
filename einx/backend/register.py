import sys
import einx
import threading
import importlib
import numpy as np
from .base import Backend

backends = []
backend_factories = {}  # module-name: [backend-factory]
tensortype_to_backend = {}
name_to_backend = {}
lock = threading.RLock()


def register_for_module(module_name, backend_factory):
    with lock:
        if module_name in sys.modules:
            # Module is already imported -> create backend now
            register(backend_factory())
        else:
            # Module is not yet imported -> register factory
            if not module_name in backend_factories:
                backend_factories[module_name] = []
            backend_factories[module_name].append(backend_factory)


def register(backend):
    with lock:
        if not isinstance(backend, Backend):
            raise ValueError("Backend must be an instance of einx.backend.Backend")
        backends.append(backend)
        for type in backend.tensor_types:
            tensortype_to_backend[type] = backend
        name_to_backend[backend.name] = backend

        return backend


from . import _numpy
from . import _torch
from . import _tensorflow
from . import _jax
from . import _dask
from . import _mlx
from . import _tinygrad

# Create numpy backend now
numpy = register(_numpy.create())

# Register other backends to be created after the corresponding modules are imported
register_for_module("torch", _torch.create)
register_for_module("tensorflow", _tensorflow.create)
register_for_module("jax", _jax.create)
register_for_module("dask.array", _dask.create)
register_for_module("mlx", _mlx.create)
register_for_module("tinygrad", _tinygrad.create)


# Check if any new modules have been imported and construct backends that have been
# registered for them
def _update():
    for module_name in list(backend_factories.keys()):
        if module_name in sys.modules:
            for factory in list(backend_factories[module_name]):
                register(factory())
            del backend_factories[module_name]


def _get1(tensor):
    backend = tensortype_to_backend.get(type(tensor), None)
    if not backend is None:
        return backend

    _update()

    for backend in backends:
        if any(isinstance(tensor, type) for type in backend.tensor_types) and not isinstance(
            tensor, np.ndarray
        ):
            # Found matching backend
            break
    else:
        return None

    tensortype_to_backend[type(tensor)] = backend
    return backend


def get(arg):
    with lock:
        if isinstance(arg, str):
            if arg in name_to_backend:
                return name_to_backend[arg]
            _update()
            if arg in name_to_backend:
                return name_to_backend[arg]
            raise ValueError(f"Backend {arg} not found")
        else:
            tensors = arg
            if len(tensors) == 1:
                return _get1(tensors[0])
            backend = None
            for tensor in tensors:
                if tensor is not None:
                    backend2 = _get1(tensor)
                    if not backend2 is None:
                        if (
                            backend is not None
                            and backend != backend2
                            and backend != numpy
                            and backend2 != numpy
                        ):
                            raise ValueError(
                                "Got tensors with conflicting backends: "
                                f"{backend.__name__} and {backend2.__name__}"
                            )
                        if backend is None or backend2 != numpy:
                            backend = backend2
            if backend is None:
                raise ValueError(f"Could not determine the backend to use in this operation")
            else:
                return backend

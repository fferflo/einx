import sys
import traceback
import types
import threading
from einx._src.frontend.errors import ImportBackendError
from einx._src.frontend.errors import OperationNotSupportedError
from einx._src.frontend.errors import BackendResolutionError
import functools
import numpy as np


class InvalidBackend:
    def __init__(self, name, message, priority=0):
        self.name = name
        self.message = message
        self.priority = priority

    def is_supported_tensor(self, tensor):
        return False

    def __getattr__(self, name):
        raise ImportBackendError(self.message)

    def raise_on_import_failure(self):
        raise ImportBackendError(self.message)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __enter__(self):
        Use(self, registry).__enter__()

    def __exit__(self, *args):
        Use(self, registry).__exit__(*args)


class Backend:
    def __init__(self, ops, name, priority, optimizations, compiler, is_supported_tensor, get_shape):
        self.ops = ops
        self.name = name
        self.priority = priority
        self.optimizations = optimizations
        self.compiler = compiler
        self.is_supported_tensor = is_supported_tensor
        self.get_shape = get_shape

    def __getattr__(self, name):
        def op(*args, **kwargs):
            if name not in self.ops:
                raise OperationNotSupportedError(f"{name} operation is not supported by the {self.name} backend.")
            return self.ops[name](*args, **kwargs)

        if name in self.ops:
            op = functools.wraps(self.ops[name])(op)
        else:
            op.__name__ = name

        return op

    def raise_on_import_failure(self):
        pass

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __enter__(self):
        Use(self, registry).__enter__()

    def __exit__(self, *args):
        Use(self, registry).__exit__(*args)


class Use:
    def __init__(self, backend, registry):
        self.backend = backend
        self.registry = registry

    def __enter__(self):
        self.registry.enter(self.backend)

    def __exit__(self, exc_type, exc_value, traceback):
        self.registry.exit(self.backend)


class BackendRegistryState:
    def __init__(self, state=None):
        self.seen_module_names = set()
        self.uninitialized_backends = {}  # module name -> (backend factory, tensor-types)
        self.backends = []
        self.tensortypes_to_backend = {}
        self.name_to_backend = {}
        self.use_stack = []
        if state is not None:
            self.seen_module_names.update(state.seen_module_names)
            self.uninitialized_backends.update(state.uninitialized_backends)
            self.backends.extend(state.backends)
            self.tensortypes_to_backend.update(state.tensortypes_to_backend)
            self.name_to_backend.update(state.name_to_backend)
            self.use_stack.extend(state.use_stack)

    def _invalid_backend_reasons(self):
        invalid_backends = [backend for backend in self.backends if isinstance(backend, InvalidBackend)]
        if len(invalid_backends) > 0:
            message = "\n\nThe following backends could not be initialized:\n"
            for backend in invalid_backends:
                message += f"\n############################ {backend.name} ############################\n"
                message += f"{backend.message}\n"
        else:
            message = ""
        return message

    def _run_factory(self, module_name, backend_name, backend_factory):
        try:
            backend = backend_factory()
        except Exception:
            backend = InvalidBackend(
                backend_name, f'Failed to import backend "{backend_name}" for module "{module_name}" due to the following error:\n{traceback.format_exc()}'
            )
        self._register(backend)

    def _register(self, backend):
        if not isinstance(backend, Backend | InvalidBackend):
            raise ValueError("Backend must be an instance of Backend or InvalidBackend.")

        self.backends.append(backend)
        self.name_to_backend[backend.name] = backend

    def _register_on_import(self, module_name, backend_name, backend_factory):
        if module_name in sys.modules:
            # Module is already imported -> register backend now
            self._run_factory(module_name, backend_name, backend_factory)
        else:
            # Module is not yet imported -> register factory
            if module_name not in self.uninitialized_backends:
                self.uninitialized_backends[module_name] = []
            self.uninitialized_backends[module_name].append((backend_name, backend_factory))

    def _check_new_imports(self, has_checked):
        if has_checked[0]:
            return False
        has_checked[0] = True

        if any(module_name not in self.seen_module_names for module_name in sys.modules):
            new_module_names = [module_name for module_name in sys.modules if module_name not in self.seen_module_names]

            for new_module_name in new_module_names:
                self.seen_module_names.add(new_module_name)
                if new_module_name in self.uninitialized_backends:
                    for backend_name, backend_factory in self.uninitialized_backends[new_module_name]:
                        self._run_factory(new_module_name, backend_name, backend_factory)
                    del self.uninitialized_backends[new_module_name]

            return True
        else:
            return False

    def _get_by_tensors(self, tensors, has_checked_new_imports=None):
        tensortypes = tuple(type(tensor) for tensor in tensors)
        if tensortypes in self.tensortypes_to_backend:
            return [self.tensortypes_to_backend[tensortypes]]

        if has_checked_new_imports is None:
            has_checked_new_imports = [False]

        def _get_by_tensor(tensor):
            # Find backends that support this tensor type
            backends = [backend for backend in self.backends if backend.is_supported_tensor(tensor)]

            if len(backends) > 0:
                return backends
            else:
                # If no backend was found and we haven't already checked for new imports, check and try again
                changed = self._check_new_imports(has_checked_new_imports)
                if changed:
                    return _get_by_tensor(tensor)
                return []

        backends = set()
        for tensor in tensors:
            backends.update(_get_by_tensor(tensor))
        backends = list(backends)

        if all(isinstance(tensor, float | int | bool | np.floating | np.integer | np.bool_) for tensor in tensors):
            backends = [self._get_by_name("numpy")]

        # Keep only backends with highest priority
        if len(backends) > 1:
            max_priority = max(backend.priority for backend in backends)
            backends = [backend for backend in backends if backend.priority == max_priority]

        # If exactly one backend was found, cache it for these tensor types
        if len(backends) == 1:
            self.tensortypes_to_backend[tensortypes] = backends[0]

        return backends

    def _get_by_name(self, name, has_checked_new_imports=None):
        if name not in self.name_to_backend:
            if has_checked_new_imports is None:
                has_checked_new_imports = [False]
            changed = self._check_new_imports(has_checked_new_imports)
            if not changed or name not in self.name_to_backend:
                raise ValueError(
                    f"Backend with name {name} not found. Currently registered backends are: "
                    f"{list(self.name_to_backend.keys())}{self._invalid_backend_reasons()}"
                )
        return self.name_to_backend.get(name)

    def _get(self, backend=None, tensors=None, has_checked_new_imports=None):
        if tensors is None:
            tensors = []

        # If backend object is given
        if isinstance(backend, Backend | InvalidBackend):
            return backend

        if has_checked_new_imports is None:
            has_checked_new_imports = [False]

        # If backend name is given
        if isinstance(backend, str):
            return self._get_by_name(backend, has_checked_new_imports)

        # If global default backend is specified
        if len(self.use_stack) > 0:
            return self.use_stack[-1]

        # Other backend parameters are invalid
        if backend is not None:
            raise ValueError("Backend must be either a Backend instance, a string, or None.")

        # If no backend is specified, determine backend from tensors
        backends = self._get_by_tensors(tensors, has_checked_new_imports)
        if len(backends) == 1:
            return backends[0]
        elif len(backends) > 1:
            raise BackendResolutionError(
                "Failed to determine which backend to use for this operation:\n"
                " - The 'backend' parameter is not specified.\n"
                " - No global default backend is specified using 'with backend:'.\n"
                " - Multiple registered backends with same priority match the tensor types of the arguments: "
                f"{', '.join([backend.name for backend in backends])}"
            )
        else:
            message = (
                "Failed to determine which backend to use for this operation:\n"
                " - The 'backend' parameter is not specified.\n"
                " - No global default backend is specified using 'with backend:'.\n"
                f" - No registered backends match the tensor types of the arguments: {', '.join([str(type(t)) for t in tensors])}"
            )
            message += self._invalid_backend_reasons()

            raise BackendResolutionError(message)

    def _enter(self, backend):
        self.use_stack.append(backend)

    def _exit(self, backend):
        assert id(self.use_stack[-1]) == id(backend)
        self.use_stack.pop()

    def register(self, backend):
        new_state = BackendRegistryState(self)
        new_state._register(backend)
        return new_state

    def register_on_import(self, module_name, backend_name, backend_factory):
        new_state = BackendRegistryState(self)
        new_state._register_on_import(module_name, backend_name, backend_factory)
        return new_state

    def get_by_tensors(self, tensors):
        new_state = BackendRegistryState(self)
        backends = new_state._get_by_tensors(tensors)
        return new_state, backends

    def get_by_name(self, name):
        new_state = BackendRegistryState(self)
        backend = new_state._get_by_name(name)
        return new_state, backend

    def get(self, backend=None, tensors=None):
        new_state = BackendRegistryState(self)
        backend = new_state._get(backend, tensors)
        return new_state, backend

    def enter(self, backend):
        new_state = BackendRegistryState(self)
        new_state._enter(backend)
        return new_state

    def exit(self, backend):
        new_state = BackendRegistryState(self)
        new_state._exit(backend)
        return new_state


class BackendRegistry:
    def __init__(self):
        self.state = BackendRegistryState()
        self.use_lock = threading.Lock()

    def register(self, backend):
        self.state = self.state.register(backend)

    def register_on_import(self, module_name, backend_name, backend_factory):
        self.state = self.state.register_on_import(module_name, backend_name, backend_factory)

    def get_by_tensors(self, tensor):
        self.state, backends = self.state.get_by_tensors(tensor)
        return backends

    def get_by_name(self, name):
        self.state, backend = self.state.get_by_name(name)
        return backend

    def get(self, backend=None, tensors=None):
        self.state, backend = self.state.get(backend, tensors)
        return backend

    def enter(self, backend):
        with self.use_lock:
            self.state = self.state.enter(backend)

    def exit(self, backend):
        with self.use_lock:
            self.state = self.state.exit(backend)


registry = BackendRegistry()

import functools
import inspect
import traceback
import types
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar, overload

import numpy as np

import einx._src.tracer as tracer
from einx._src.frontend.errors import CallOperationError
from einx._src.util.lru_cache import lru_cache

from .backend import registry
from .types import Tensor

_F = TypeVar("_F", bound=Callable[..., Any])


class TensorArg:
    def __init__(self, value):
        self.value = value


def _split_tensors(signature, args, kwargs):
    tensor_args = []

    def replace_with_tensorarg(value):
        tensor_args.append(value)
        return TensorArg(value)

    try:
        bound = signature.bind(*args, **kwargs)
        bound_arguments_without_defaults = bound.arguments.copy()
        bound.apply_defaults()
        bound_arguments_with_defaults = bound.arguments.copy()
    except TypeError as e:
        raise TypeError(f"The einx operation received incorrect arguments. {e}\nThe function's signature is: {signature}") from e

    new_args = []
    new_kwargs = {}

    for name, param in signature.parameters.items():
        if name in bound_arguments_with_defaults:
            val = bound_arguments_with_defaults[name]

            # Determine if the parameter is a tensor and will be traced
            if param.annotation is Tensor or param.annotation == "Tensor":
                # Parameter is marked as Tensor => is a tensor
                is_tensor = True
            else:
                # Parameter is marked as non-Tensor or has no annotation => not a tensor
                is_tensor = False

            if is_tensor:
                # Tensors are traced and replaced with TensorArg
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    val = tuple(replace_with_tensorarg(v) for v in val)
                else:
                    val = replace_with_tensorarg(val)

            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                new_args.append(val)
            elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                assert isinstance(val, tuple | list)
                new_args.extend(val)
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                new_kwargs[name] = val
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                assert isinstance(val, dict)
                new_kwargs.update(val)
            elif param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                # Positional-or-keyword arguments are ...
                if name in kwargs:
                    # ... forwarded as keyword arguments if they are given as keyword arguments
                    new_kwargs[name] = val
                elif name not in bound_arguments_without_defaults:
                    # ... forwarded as keyword arguments if they are given only as default parameters
                    new_kwargs[name] = val
                else:
                    # ... forwarded as positional arguments if they are given as positional arguments
                    new_args.append(val)
            else:
                raise AssertionError(f"Unknown parameter kind: {param.kind}")

    return new_args, new_kwargs, tensor_args


def _is_scalar(x):
    return isinstance(x, int | float | bool | np.integer | np.floating | np.bool_)


def _get_signature(func):
    try:
        signature = inspect.signature(func)
    except ValueError:
        signature = inspect.signature(lambda shape: None)
    return {**signature.parameters}


def _to_tracer(x, backend, name):
    if isinstance(x, TensorArg):
        if backend.is_supported_tensor(x.value):
            return tracer.signature.classical.Tensor(None, shape=backend.get_shape(x.value))
        elif isinstance(x.value, np.ndarray):
            return tracer.signature.classical.ConvertibleTensor(
                None, shape=tuple(int(x) for x in x.value.shape), concrete=types.SimpleNamespace(type=type(x.value))
            )
        elif _is_scalar(x.value):
            return tracer.signature.classical.ConvertibleTensor(None, shape=(), concrete=types.SimpleNamespace(type=type(x.value)))
        elif callable(x.value):
            return tracer.signature.classical.ConvertibleTensor(
                None, shape=None, concrete=types.SimpleNamespace(type=type(x.value), parameters=_get_signature(x.value))
            )
        else:
            raise ValueError(f"The {name} to the einx function has an invalid type: {type(x.value)}")
    else:
        return x


def _construct_graph(args, kwargs, func, backend=None):
    if backend is None:
        backend = kwargs["backend"]

    # Trace function with the given tracer objects
    input_tracers = [x for x in list(args) + list(kwargs.values()) if isinstance(x, tracer.Tracer)]
    with tracer.depend_on(
        *input_tracers
    ):  # Ensure that no constant tensors are allocated at graph construction time -> all functions must be invoked inside the compiled function
        output_tracer = func(*args, **kwargs)

    # Create graph object
    graph = tracer.Graph(inputs=input_tracers, output=output_tracer, name="op")

    # from einx._src.tracer.visualize import visualize # TODO: remove
    # dot = visualize(graph)
    # dot.render(filename="~/graph1", format="pdf", cleanup=True)

    # Optimize graph
    graph = tracer.optimize(graph, optimizations=backend.optimizations)

    # from einx._src.tracer.visualize import visualize # TODO: remove
    # dot = visualize(graph)
    # dot.render(filename="~/graph2", format="pdf", cleanup=True)

    # Compile graph to callable function
    function, code = backend.compiler.compile(graph, return_code=True)

    return function, code


def to_ord_str(x):
    if x == 0:
        return "1st"
    elif x == 1:
        return "2nd"
    elif x == 2:
        return "3rd"
    else:
        return f"{x + 1}th"


def _api_withoutbackend(func, signature):
    if signature is None:
        signature = inspect.signature(func)

    construct_graph_with_cache = lru_cache(partial(_construct_graph, func=func))

    @functools.wraps(func)
    def inner(*args, backend=None, graph=False, **kwargs):
        # Find tensor arguments that will be traced
        args, kwargs, tensor_args = _split_tensors(signature, args, kwargs)
        del kwargs["backend"]

        # Find backend
        backend = registry.get(backend, tensor_args)
        backend.raise_on_import_failure()

        # Convert tensor arguments to Tracer
        args = [_to_tracer(x, backend, name=f"{to_ord_str(i)} positional argument") for i, x in enumerate(args)]
        kwargs = {k: _to_tracer(v, backend, name=f"keyword argument '{k}'") for k, v in kwargs.items()}

        # Construct function or retrieve from cache
        function, code = construct_graph_with_cache(args=args, kwargs=kwargs | {"backend": backend})
        if graph:
            return code

        # Call function
        try:
            return function(*tensor_args)
        except Exception as e:
            raise CallOperationError.create(e, code) from e

    return inner


def _api_withbackend(func, backend, signature):
    if signature is None:
        signature = inspect.signature(func)

    construct_graph_with_cache = lru_cache(partial(_construct_graph, func=func, backend=backend))

    @functools.wraps(func)
    def inner(*args, graph=False, **kwargs):
        # Find tensor arguments that will be traced
        args, kwargs, tensor_args = _split_tensors(signature, args, kwargs)

        # Convert tensor arguments to Tracer
        args = [_to_tracer(x, backend, name=f"{to_ord_str(i)} positional argument") for i, x in enumerate(args)]
        kwargs = {k: _to_tracer(v, backend, name=f"keyword argument '{k}'") for k, v in kwargs.items()}

        # Construct function or retrieve from cache
        function, code = construct_graph_with_cache(args=args, kwargs=kwargs)
        if graph:
            return code

        # Call function
        try:
            return function(*tensor_args)
        except Exception as e:
            raise CallOperationError.create(e, code) from e

    return inner


@overload
def api(func: _F) -> _F: ...
@overload
def api(func: None = None, backend: Any = None, signature: Any = None) -> Callable[[_F], _F]: ...
def api(func=None, backend=None, signature=None):
    if func is None:
        return partial(api, backend=backend, signature=signature)
    if backend is None:
        return _api_withoutbackend(func, signature=signature)
    else:
        return _api_withbackend(func, backend, signature=signature)

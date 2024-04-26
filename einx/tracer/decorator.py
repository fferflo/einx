import functools
import os
import einx
import threading
import frozendict
import inspect
import sys
from functools import partial
import numpy as np
from collections import defaultdict
from .compile import CompiledFunction
from .tracer import *


def _freeze(x):
    if isinstance(x, np.ndarray):
        return tuple(x.tolist())
    elif isinstance(x, (list, tuple)):
        return tuple(_freeze(x) for x in x)
    elif isinstance(x, dict):
        return frozendict.frozendict({k: _freeze(v) for k, v in x.items()})
    else:
        return x


def freeze(func):
    @functools.wraps(func)
    def func_frozen(*args, **kwargs):
        args = [_freeze(a) for a in args]
        kwargs = {k: _freeze(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return func_frozen


traced_functions_decorators = []
traced_functions = []
traced_functions_lock = threading.Lock()

thread_local = threading.local()
thread_local.warn = True


def _with_retrace_warning(func):
    warn_on_retrace_num = int(os.environ.get("EINX_WARN_ON_RETRACE", 0))

    if warn_on_retrace_num > 0:
        cache_failures = defaultdict(lambda: 0)

        @functools.wraps(func)
        def func_with_warn(*args, **kwargs):
            has_warned = False
            if warn_on_retrace_num > 0:
                stack = inspect.stack()
                # Exclude frames called from this file
                last_index = 0
                for i, frame in enumerate(stack):
                    if frame.filename == __file__:
                        last_index = i
                stack = stack[last_index + 1 :]

                if len(stack) > 0:
                    # Generate string description of call stack
                    trace = ""
                    for frame in reversed(stack):
                        trace += (
                            f'File "{frame.filename}", line {frame.lineno}, in {frame.function}\n'
                        )
                        if frame.code_context is not None:
                            trace += f"  {frame.code_context[0].strip()}\n"
                    cache_failures[trace] += 1
                    if thread_local.warn and cache_failures[trace] == warn_on_retrace_num:
                        # Print warning
                        has_warned = True
                        print(
                            f"WARNING (einx): The following call stack has resulted in "
                            f"{warn_on_retrace_num} retraces of an einx function.\n"
                            f"A retrace happens when the function is called with "
                            "different signatures of input arguments.\n"
                            f"Call stack (most recent call last):\n"
                            f"{trace}"
                        )

            # Don't warn in inner functions that also use lru_cache
            if has_warned:
                thread_local.warn = False
                result = func(*args, **kwargs)
                thread_local.warn = True
            else:
                result = func(*args, **kwargs)
            return result

        return func_with_warn
    else:
        return func


def lru_cache(func):
    func = _with_retrace_warning(func)

    max_cache_size = int(os.environ.get("EINX_CACHE_SIZE", -1))
    if max_cache_size > 0:
        func = functools.lru_cache(maxsize=max_cache_size if max_cache_size > 0 else None)(func)
    elif max_cache_size < 0:
        if "cache" in vars(functools):
            func = functools.cache(func)
        else:
            func = functools.lru_cache(maxsize=None)(func)
    func = freeze(func)

    return func


_thread_local = threading.local()


def _get_trace_stack():
    if not hasattr(_thread_local, "stack"):
        _thread_local.stack = []
    return _thread_local.stack


class _trace_context:
    def __init__(self, backend):
        self.backend = backend

    def __enter__(self):
        _get_trace_stack().append(self)

    def __exit__(self, *args):
        assert id(_get_trace_stack()[-1]) == id(self)
        _get_trace_stack().pop()


def _is_tracing():
    return len(_get_trace_stack()) > 0


trace_all = lambda t, c: lambda *args, **kwargs: c(
    *[t(arg) for arg in args], **{k: t(v) for k, v in kwargs.items()}
)
trace_none = lambda t, c: lambda *args, **kwargs: c(args, kwargs)


def jit(func=None, trace=trace_all):
    if func is None:
        return partial(jit, trace=trace)

    @lru_cache
    def construct_graph(args, kwargs, backend):
        with _trace_context(backend):
            # Replace input keys with tracers and retrieve list of traced arguments
            virtual_arg = einx.tracer.Tracer()
            input_tracers, (args, kwargs) = einx.tracer.input.key_to_tracer(
                (args, kwargs), backend, virtual_arg
            )

            # Trace function
            output_tracer = func(*args, backend=backend, **kwargs)

            # Create function that takes only traced arguments as input
            function = TracableFunction(
                args=input_tracers,
                output=output_tracer,
                name=backend.function_name,
                virtual_args=[virtual_arg],
            )
            for decorator in backend.decorators:
                function = decorator(function)

            # Convert to graph
            return CompiledFunction(function)

    def find_backend_and_construct_graph(args, kwargs, traced_input_values, backend):
        # Determine backend
        if backend is None:
            backend = einx.backend.get_default()
            if backend is None:
                backend = einx.backend.get(traced_input_values)
        elif isinstance(backend, str):
            backend = einx.backend.get(backend)

        # Construct graph/ retrieve from cache
        graph = construct_graph(args=args, kwargs=kwargs, backend=backend)

        return graph

    has_decorated = False

    @functools.wraps(func)
    def func_jit(*args, backend=None, graph=False, **kwargs):
        if _is_tracing():
            assert not graph
            if backend is None:
                backend = _get_trace_stack()[-1].backend
            elif backend != _get_trace_stack()[-1].backend:
                raise ValueError("Cannot change backend during tracing")

            return func(*args, backend=backend, **kwargs)

        return_graph = graph

        # Replace concrete values with tracers
        traced_input_values = []

        def new_input(x):
            value, key = einx.tracer.input.concrete_to_value_and_key(x)
            if not value is None:
                traced_input_values.append(value)
            return key

        args, kwargs = trace(new_input, lambda *args, **kwargs: (args, kwargs))(*args, **kwargs)

        # Disable torch.compile for graph construction (if torch is imported)
        nonlocal has_decorated, find_backend_and_construct_graph
        if not has_decorated and "torch" in sys.modules:
            import torch._dynamo as _dynamo

            find_backend_and_construct_graph = _dynamo.disable(find_backend_and_construct_graph)
            has_decorated = True

        graph = find_backend_and_construct_graph(args, kwargs, traced_input_values, backend)

        # Execute/ return graph
        if return_graph:
            return graph
        else:
            return graph(*traced_input_values)

    with traced_functions_lock:
        traced_functions.append(func_jit)
        for decorator in traced_functions_decorators:
            decorator(func_jit)

    return func_jit


def decorate_traced_functions(decorator):
    with traced_functions_lock:
        for func in traced_functions:
            decorator(func)
        traced_functions_decorators.append(decorator)


jit.decorate_traced_functions = decorate_traced_functions

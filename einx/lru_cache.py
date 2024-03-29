import collections
import functools
import os
import einx
import threading
import frozendict
import inspect
from functools import partial
import numpy as np
from collections import defaultdict


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
    def inner(*args, **kwargs):
        args = [_freeze(a) for a in args]
        kwargs = {k: _freeze(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return inner


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


def lru_cache(func=None, trace=None):
    if func is None:
        return partial(lru_cache, trace=trace)

    max_cache_size = int(os.environ.get("EINX_CACHE_SIZE", -1))
    if max_cache_size == 0:
        # Don't use cache, don't trace arguments
        inner = func
    elif trace is None:
        # No arguments are traced: Wrap function in cache
        func = _with_retrace_warning(func)
        if max_cache_size < 0:
            inner = freeze(functools.lru_cache(maxsize=None)(func))  # No cache limit
        else:
            inner = freeze(functools.lru_cache(maxsize=max_cache_size)(func))
    else:
        # Arguments are traced: Create cache for graph, then wrap
        # cache in a function that executes graph
        @lru_cache
        def construct_graph(*args, backend, **kwargs):
            output_tracers = func(*args, **kwargs, backend=einx.backend.tracer)
            return einx.backend.tracer.Graph(
                output_tracers, args=args, kwargs=kwargs, backend=backend
            )

        @functools.wraps(func)
        def inner(*args, backend=None, graph=False, **kwargs):
            return_graph = graph

            # Get traced arguments and cache key
            traced_input_values = []
            index = 0

            def new_input(x):
                nonlocal index
                traced_input_values.append(x)
                x = einx.backend.tracer.Input(
                    shape=einx.param.get_shape(x), index=index, original_type=type(x)
                )
                index += 1
                return x

            def get_args_kwargs(*args, **kwargs):
                return args, kwargs

            args, kwargs = trace(new_input, get_args_kwargs)(*args, **kwargs)

            if backend is None:
                backend = einx.backend.get(traced_input_values)
            elif isinstance(backend, str):
                backend = einx.backend.get(backend)
            backend._decorate_construct_graph(construct_graph)

            graph = construct_graph(*args, backend=backend, **kwargs)

            if return_graph:
                return graph
            else:
                return graph(*traced_input_values)

        with traced_functions_lock:
            traced_functions.append(inner)
            for decorator in traced_functions_decorators:
                decorator(inner)

    return inner


def decorate_traced_functions(decorator):
    with traced_functions_lock:
        for func in traced_functions:
            decorator(func)
        traced_functions_decorators.append(decorator)


lru_cache.decorate_traced_functions = decorate_traced_functions

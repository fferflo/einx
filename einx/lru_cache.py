import collections, threading, functools, os, einx, inspect, threading, frozendict
from functools import partial
import numpy as np

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

def lru_cache(func=None, trace=None):
    if func is None:
        return partial(lru_cache, trace=trace)

    if trace is None:
        # No arguments are traced: Wrap function in cache
        max_cache_size = int(os.environ.get("EINX_CACHE_SIZE", -1))
        if max_cache_size == 0:
            inner = func
        elif max_cache_size < 0:
            inner = freeze(functools.lru_cache(maxsize=None)(func)) # No cache limit
        else:
            inner = freeze(functools.lru_cache(maxsize=max_cache_size)(func))
    else:
        # Arguments are traced: Create cache for graph, then wrap cache in a function that executes graph
        @lru_cache
        def construct_graph(*args, backend=None, **kwargs):
            output_tracers = func(*args, **kwargs, backend=einx.backend.tracer)
            return einx.backend.tracer.Graph(output_tracers, name=func.__name__, args=args, kwargs=kwargs)

        @functools.wraps(func)
        def inner(*args, backend=None, graph=False, **kwargs):
            return_graph = graph

            input_tracer_values = []
            index = 0
            def new_input(x):
                nonlocal index
                input_tracer_values.append(x)
                x = einx.backend.tracer.Input(shape=einx.param.get_shape(x), index=index)
                index += 1
                return x
            graph = trace(new_input, construct_graph)(*args, backend=backend, **kwargs)

            if return_graph:
                return graph
            else:
                return graph(*input_tracer_values, backend=backend)

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
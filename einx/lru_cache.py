import collections, threading, functools, os, einx, inspect, threading
from functools import partial
import numpy as np

def _hash(x):
    if isinstance(x, list) or isinstance(x, tuple):
        h = 91724
        for v in x:
            h += _hash(v)
            h *= 18738
    elif isinstance(x, dict):
        h = 697123
        for v in x.values():
            h += _hash(v)
            h *= 9583
    else:
        h = hash(x)
    return int(h) % 2147483648 # Fixes issue with torch.compile

def _prepare(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    else:
        return x

traced_functions_decorators = []
traced_functions = []
traced_functions_lock = threading.Lock()

def lru_cache(func=None, trace=None):
    if func is None:
        return partial(lru_cache, trace=trace)

    if trace is None:
        # No arguments are traced: Wrap function in LRU cache
        max_cache_size = int(os.environ.get("EINX_CACHE_SIZE", -1))
        if max_cache_size == 0:
            return func

        print_cache_miss = str(os.environ.get("EINX_PRINT_CACHE_MISS", "false")).lower() in ["true", "yes", "1"]
        cache = collections.OrderedDict()
        lock = threading.Lock()

        @functools.wraps(func)
        def inner(*args, **kwargs):
            key = (args, kwargs)
            key = einx.tree_util.tree_map(_prepare, key)
            h = _hash(key)

            with lock:
                if h in cache:
                    # Hash collision
                    for k, v in cache[h]:
                        if k == key:
                            # Cache hit
                            cache.move_to_end(h)
                            return v

            # Cache miss
            if print_cache_miss:
                print(f"einx: Cache miss on {inner.__name__} with args={args} kwargs={kwargs} hash={h} cache_size={len(cache)}")

            result = func(*args, **kwargs)

            with lock:
                if h in cache:
                    candidates = cache[h]
                else:
                    candidates = cache[h] = []
                    if max_cache_size >= 0 and len(cache) > max_cache_size:
                        cache.popitem(False)
                candidates.append((key, result))

            return result
    else:
        # Arguments are traced: Create LRU cache for graph, then wrap cache in a function that executes graph
        if len(inspect.signature(trace).parameters) == 1:
            trace0 = trace
            trace = lambda key, value: trace0(key)

        @lru_cache
        def construct_graph(*args, **kwargs):
            output_tracers = func(*args, **kwargs, backend=einx.backend.tracer)
            return einx.backend.tracer.Graph(output_tracers, name=func.__name__, args=args, kwargs=kwargs)

        @functools.wraps(func)
        def inner(*args, backend=None, graph=False, **kwargs):
            return_graph = graph

            # Replace marked arguments with tracers
            def map(x, key):
                if trace(key, x):
                    return einx.backend.tracer.Input(key, einx.param.get_shape(x))
                else:
                    return x
            args_replaced_with_tracers = einx.tree_util.tree_map_with_key(map, args)
            kwargs_replaced_with_tracers = einx.tree_util.tree_map_with_key(map, kwargs)

            graph = construct_graph(*args_replaced_with_tracers, **kwargs_replaced_with_tracers)

            if return_graph:
                return graph
            else:
                return graph(*args, backend=backend, **kwargs)

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
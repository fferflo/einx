import collections, threading, functools, os, einx, inspect
from functools import partial

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


class _tensor_factory:
    pass

def lru_cache(func=None, trace=None):
    if func is None:
        return partial(lru_cache, trace=trace)

    max_cache_size = int(os.environ.get("EINX_CACHE_SIZE", 1024))
    if max_cache_size == 0:
        return func

    print_cache_miss = str(os.environ.get("EINX_PRINT_CACHE_MISS", "false")).lower() in ["true", "yes", "1"]
    cache = collections.OrderedDict()
    lock = threading.Lock()

    if trace is None:
        @functools.wraps(func)
        def inner(*args, **kwargs):
            key = (args, kwargs)
            h = _hash(key)

            result = None
            with lock:
                if h in cache:
                    for k, v in cache[h]:
                        if k == key:
                            cache.move_to_end(h)
                            result = v
                            break

            if result is None:
                if print_cache_miss:
                    print(f"einx: Cache miss on {inner.__name__} with args={args} kwargs={kwargs} hash={h} cache_size={len(cache)}")

                result = func(*args, **kwargs)

                with lock:
                    if h in cache:
                        candidates = cache[h]
                    else:
                        candidates = cache[h] = []
                        if len(cache) > max_cache_size:
                            cache.popitem(False)
                    candidates.append((key, result))

            return result
    else:
        if len(inspect.signature(trace).parameters) == 1:
            trace0 = trace
            trace = lambda key, value: trace0(key)

        @functools.wraps(func)
        def inner(*args, backend=None, graph=False, **kwargs):
            return_graph = graph
            def map(x, key):
                if trace(key, x):
                    shape = einx.param.get_shape(x)
                    if shape is None:
                        return _tensor_factory
                    else:
                        return shape
                else:
                    return x
            args_key = einx.tree_util.tree_map_with_key(map, args)
            kwargs_key = einx.tree_util.tree_map_with_key(map, kwargs)
            key = (args_key, kwargs_key)
            h = _hash(key)

            graph = None
            with lock:
                if h in cache:
                    for k, v in cache[h]:
                        if k == key:
                            cache.move_to_end(h)
                            graph = v
                            break

            if graph is None:
                if print_cache_miss:
                    map = lambda x, key: f"TracedTensor({einx.param.get_shape(x)})" if trace(key, x) else x
                    args_print = einx.tree_util.tree_map_with_key(map, args)
                    kwargs_print = einx.tree_util.tree_map_with_key(map, kwargs)
                    print(f"einx: Cache miss on {inner.__name__} with args={args_print} kwargs={kwargs_print} hash={h} cache_size={len(cache)}")

                map = lambda x, key: einx.backend.tracer.Input(key, einx.param.get_shape(x)) if trace(key, x) else x
                args_replaced_with_tracers = einx.tree_util.tree_map_with_key(map, args)
                kwargs_replaced_with_tracers = einx.tree_util.tree_map_with_key(map, kwargs)

                output_tracers = func(*args_replaced_with_tracers, **kwargs_replaced_with_tracers, backend=einx.backend.tracer)

                graph = einx.backend.tracer.Graph(output_tracers, name=func.__name__, args=args_replaced_with_tracers, kwargs=kwargs_replaced_with_tracers)

                with lock:
                    if h in cache:
                        candidates = cache[h]
                    else:
                        candidates = cache[h] = []
                        if len(cache) > max_cache_size:
                            cache.popitem(False)
                    candidates.append((key, graph))

            if return_graph:
                return graph
            else:
                return graph(*args, backend=backend, **kwargs)

    return inner
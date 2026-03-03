import os
import functools
import threading
import inspect
from collections import defaultdict
import numpy as np
from functools import partial
import frozendict
import types

_thread_local = threading.local()

warn_on_retrace_num = int(os.environ.get("EINX_WARN_ON_RETRACE", 0))
max_cache_size = int(os.environ.get("EINX_CACHE_SIZE", -1))


def _freeze_value(x):
    if isinstance(x, np.ndarray):
        return _freeze_value(x.tolist())
    elif isinstance(x, list | tuple):
        return tuple(_freeze_value(x) for x in x)
    elif isinstance(x, dict):
        return frozendict.frozendict({k: _freeze_value(v) for k, v in x.items()})
    elif isinstance(x, types.SimpleNamespace):
        return _freeze_value(vars(x))
    elif isinstance(x, inspect.Parameter):
        return _freeze_value((x.name, x.default, x.annotation, x.kind))
    else:
        return x


def _freeze_args(func):
    @functools.wraps(func)
    def func_frozen(*args, **kwargs):
        args = [_freeze_value(a) for a in args]
        kwargs = {k: _freeze_value(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return func_frozen


def _with_retrace_warning(func):
    if warn_on_retrace_num > 0:
        cache_failures = defaultdict(lambda: 0)

        @functools.wraps(func)
        def func_with_warn(*args, **kwargs):
            has_warned = False
            if warn_on_retrace_num > 0:
                if not hasattr(_thread_local, "warn"):
                    _thread_local.warn = True

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
                        trace += f'File "{frame.filename}", line {frame.lineno}, in {frame.function}\n'
                        if frame.code_context is not None:
                            trace += f"  {frame.code_context[0].strip()}\n"
                    cache_failures[trace] += 1
                    if _thread_local.warn and cache_failures[trace] == warn_on_retrace_num:
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
                _thread_local.warn = False
                result = func(*args, **kwargs)
                _thread_local.warn = True
            else:
                result = func(*args, **kwargs)
            return result

        return func_with_warn
    else:
        return func


# An LRU-cache that
# 1. allows using some mutable objects (np.ndarray, list and dict) as keys
# 2. warns if there are more than EINX_WARN_ON_RETRACE cache failures from the same call site
def lru_cache(func):
    func = _with_retrace_warning(func)

    if max_cache_size > 0:
        func = functools.lru_cache(maxsize=max_cache_size if max_cache_size > 0 else None)(func)
    elif max_cache_size < 0:
        if "cache" in vars(functools):
            func = functools.cache(func)
        else:
            func = functools.lru_cache(maxsize=None)(func)
    func = _freeze_args(func)

    return func

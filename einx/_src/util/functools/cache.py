import os
import functools
import threading
import inspect
from collections import defaultdict
import numpy as np
from functools import partial
import types
import warnings

_thread_local = threading.local()

warn_on_retrace_num = int(os.environ.get("EINX_WARN_ON_RETRACE", 0))
max_cache_size = os.environ.get("EINX_CACHE_SIZE", "inf")
if max_cache_size != "inf":
    try:
        max_cache_size = int(max_cache_size)
    except ValueError:
        warnings.warn(
            f"Invalid EINX_CACHE_SIZE={max_cache_size}, using inf instead.",
            RuntimeWarning,
            stacklevel=10,
        )
        max_cache_size = "inf"


def _freeze_value(x):
    if isinstance(x, np.ndarray):
        return (0, _freeze_value(x.tolist()))
    elif isinstance(x, list):
        return (1, tuple(_freeze_value(xi) for xi in x))
    elif isinstance(x, tuple):
        return (2, tuple(_freeze_value(xi) for xi in x))
    elif isinstance(x, dict):
        return (3, tuple((k, _freeze_value(v)) for k, v in sorted(x.items())))
    elif isinstance(x, types.SimpleNamespace):
        return (4, _freeze_value(vars(x)))
    elif isinstance(x, inspect.Parameter):
        return (5, _freeze_value((x.name, x.default, x.annotation, x.kind)))
    else:
        return (6, x)


def _unfreeze_value(x):
    if x[0] == 0:
        return np.asarray(_unfreeze_value(x[1]))
    elif x[0] == 1:
        return [_unfreeze_value(xi) for xi in x[1]]
    elif x[0] == 2:
        return tuple(_unfreeze_value(xi) for xi in x[1])
    elif x[0] == 3:
        return {k: _unfreeze_value(v) for k, v in x[1]}
    elif x[0] == 4:
        return types.SimpleNamespace(**_unfreeze_value(x[1]))
    elif x[0] == 5:
        name, default, annotation, kind = _unfreeze_value(x[1])
        return inspect.Parameter(name, kind, default=default, annotation=annotation)
    else:
        return x[1]


def _unfreeze_args_in_function(func):
    @functools.wraps(func)
    def func_unfrozen(*args, **kwargs):
        args = [_unfreeze_value(a) for a in args]
        kwargs = {k: _unfreeze_value(v) for k, v in kwargs.items()}
        return func(*args, **kwargs)

    return func_unfrozen


def _freeze_args_in_function(func):
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

            # Don't warn in inner functions that also use cache
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


def cache(func):
    """A cache decorator that
    1. allows using some mutable objects as arguments
    2. follows EINX_CACHE_SIZE environment variable
    3. warns if there are more than EINX_WARN_ON_RETRACE cache failures from the same call site
    """
    if isinstance(max_cache_size, int) and max_cache_size <= 0:
        return func
    func = _with_retrace_warning(func)

    func = _unfreeze_args_in_function(func)
    if max_cache_size == "inf":
        func = functools.cache(func)
    elif max_cache_size > 0:
        func = functools.lru_cache(maxsize=max_cache_size)(func)
    func = _freeze_args_in_function(func)

    return func

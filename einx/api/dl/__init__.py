from .dl import *

def __getattr__(key):
    if key == "torch":
        from . import _torch
        globals()["torch"] = _torch
        return _torch
    elif key == "haiku":
        from . import _haiku
        globals()["haiku"] = _haiku
        return _haiku
    elif key == "flax":
        from . import _flax
        globals()["flax"] = _flax
        return _flax
    else:
        raise AttributeError(f"Module {__name__} has no attribute {key}")
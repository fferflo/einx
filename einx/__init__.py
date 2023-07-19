anonymous_ellipsis_name = "anonymous_ellipsis"

from .lru_cache import lru_cache
from . import expr, op
from . import backend
from .api import *
from . import api

def __getattr__(key):
    if key in ["torch", "flax", "haiku"]:
        return getattr(einx.api.dl, key)
    else:
        raise AttributeError(f"Module {__name__} has no attribute {key}")
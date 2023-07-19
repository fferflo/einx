from .elementwise import *
from .reduce import *
from .dot import *
from .rearrange import *
from . import dl

def __getattr__(key):
    if key in ["torch", "flax", "haiku"]:
        return getattr(dl, key)
    else:
        raise AttributeError(f"Module {__name__} has no attribute {key}")
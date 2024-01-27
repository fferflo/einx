import sys, importlib

_frameworks = None

def get_frameworks():
    global _frameworks
    if _frameworks is None:
        _frameworks = []
        for name in ["torch", "haiku", "flax", "equinox", "keras"]:
            if name in sys.modules:
                try:
                    einn = importlib.import_module(f"einx.nn.{name}")
                    _frameworks.append(einn)
                except ImportError:
                    continue
    return _frameworks
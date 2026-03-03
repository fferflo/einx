import functools
import einx._src.tracer as tracer


def _get_name(op):
    if isinstance(op, str):
        return op
    elif isinstance(op, functools.partial):
        return _get_name(op.func)
    elif hasattr(op, "__name__"):
        return op.__name__
    elif isinstance(op, tracer.Tracer):
        if isinstance(op.origin, tracer.signature.python.GetAttr):
            obj = _get_name(op.origin.obj)
            key = op.origin.key
            if obj is None:
                return key
            else:
                return f"{obj}.{key}"
        elif isinstance(op.origin, tracer.signature.python.Import):
            if op.origin.as_ is not None:
                return op.origin.as_
            else:
                return op.origin.import_
        else:
            return None
    else:
        return None


def use_name_of(signature_op):
    name = _get_name(signature_op)

    def inner(op):
        if name is None:
            return op
        else:

            def inner(*args, **kwargs):
                return op(*args, **kwargs)

            inner.__name__ = name
            return inner

    return inner

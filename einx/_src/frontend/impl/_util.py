import inspect
from einx._src.frontend.errors import OperationNotSupportedError


def _make_iskwarg(op):
    if not callable(op):
        raise ValueError(f"The given operation must be callable, but found type {type(op)}.")

    signature = inspect.signature(op)
    parameters = {**signature.parameters}

    kwargnames = []
    for name, param in parameters.items():
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargnames.append(name)
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            raise ValueError(f"The given operation must not have a var-keyword parameter, but found '**{name}'.")
    return lambda name: name in kwargnames


def _unsupported_op(name, backend, message=None):
    def op(*args, **kwargs):
        message2 = f"{name} operation is not supported by the {backend} backend."
        if message is not None:
            message2 += " " + message
        raise OperationNotSupportedError(message2)

    op.__name__ = name
    return op

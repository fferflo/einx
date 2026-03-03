import einx._src.tracer as tracer
from einx._src.util import pytree
import inspect


class GetAttr(tracer.Application):
    def __init__(self, obj, key):
        if not isinstance(key, str):
            raise TypeError(f"key must be str, not {type(key)}")
        super().__init__(inputs=[obj, key], output=Value)
        self.obj = obj
        self.key = key

    def _tracer_transform(self, transform):
        return GetAttr(transform(self.obj), self.key)

    def __eq__(self, other):
        if isinstance(other, GetAttr):
            return self.obj == other.obj and self.key == other.key
        return False


def getattr(obj, key):
    return GetAttr(obj, key).output


class GetItem(tracer.Application):
    def __init__(self, obj, key):
        super().__init__(inputs=[obj, key], output=Value)
        self.obj = obj
        self.key = key

    def _tracer_transform(self, transform):
        return GetItem(transform(self.obj), transform(self.key))

    def __eq__(self, other):
        if isinstance(other, GetItem):
            return self.obj == other.obj and self.key == other.key
        return False


def getitem(obj, key):
    return GetItem(obj, key).output


class UpdateItem(tracer.Application):
    def __init__(self, obj, key, value, op):
        super().__init__(inputs=[obj, key, value], output=Value)
        if not isinstance(op, str):
            raise TypeError(f"op must be str, not {type(op)}")
        self.obj = obj
        self.key = key
        self.value = value
        self.op = op

    def _tracer_transform(self, transform):
        return UpdateItem(transform(self.obj), transform(self.key), transform(self.value), self.op)

    def __eq__(self, other):
        if isinstance(other, UpdateItem):
            return self.obj == other.obj and self.key == other.key and self.value == other.value and self.op == other.op
        return False


def setitem(obj, key, value):
    return UpdateItem(obj, key, value, "=").output


def additem(obj, key, value):
    return UpdateItem(obj, key, value, "+=").output


def subtractitem(obj, key, value):
    return UpdateItem(obj, key, value, "-=").output


class Call(tracer.Application):
    def __init__(self, function, args, kwargs, additional_dependencies):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if not isinstance(args, list | tuple):
            raise ValueError("args must be a list or tuple")
        if not isinstance(kwargs, dict):
            raise ValueError("kwargs must be a dict")
        for k in kwargs:
            if not isinstance(k, str):
                raise TypeError(f"key in kwargs must be str, not {type(k)}")
        super().__init__(inputs=[function] + list(args) + list(kwargs.values()) + list(additional_dependencies), output=Value)
        self.function = function
        self.args = list(args)
        self.kwargs = dict(kwargs)
        self.additional_dependencies = additional_dependencies

    def _tracer_transform(self, transform):
        return Call(
            transform(self.function),
            [transform(arg) for arg in self.args],
            {k: transform(v) for k, v in self.kwargs.items()},
            [transform(dep) for dep in self.additional_dependencies],
        )

    def __eq__(self, other):
        if isinstance(other, Call):
            return (
                self.function == other.function
                and self.args == other.args
                and self.kwargs == other.kwargs
                and self.additional_dependencies == other.additional_dependencies
            )
        return False


def call(func, args=None, kwargs=None):
    return Call(func, args, kwargs, list(tracer.get_additional_dependencies())).output


class CallInplace(tracer.Application):
    def __init__(self, xs, function, args, kwargs, additional_dependencies):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if not isinstance(function, tracer.Tracer):
            raise ValueError("function must be Tracer")
        if not isinstance(args, list | tuple):
            raise ValueError("args must be a list or tuple")
        if not isinstance(kwargs, dict):
            raise ValueError("kwargs must be a dict")
        for k in kwargs:
            if not isinstance(k, str):
                raise TypeError(f"key in kwargs must be str, not {type(k)}")
        super().__init__(inputs=[xs, function] + list(args) + list(kwargs.values()) + list(additional_dependencies), output=xs._tracer_type)
        self.xs = xs
        self.function = function
        self.args = list(args)
        self.kwargs = dict(kwargs)
        self.additional_dependencies = additional_dependencies

    def _tracer_transform(self, transform):
        return CallInplace(
            transform(self.xs),
            transform(self.function),
            [transform(arg) for arg in self.args],
            {k: transform(v) for k, v in self.kwargs.items()},
            [transform(dep) for dep in self.additional_dependencies],
        )

    def __eq__(self, other):
        if isinstance(other, CallInplace):
            return (
                self.xs == other.xs
                and self.function == other.function
                and self.args == other.args
                and self.kwargs == other.kwargs
                and self.additional_dependencies == other.additional_dependencies
            )
        return False


def call_inplace(xs, func, args=None, kwargs=None):
    return CallInplace(xs, func, args, kwargs, list(tracer.get_additional_dependencies())).output


class Import(tracer.Application):
    def __init__(self, import_, from_=None, as_=None):
        if not isinstance(import_, str):
            raise TypeError(f"import_ must be str, not {type(import_)}")
        if from_ is not None and not isinstance(from_, str):
            raise TypeError(f"from_ must be str, not {type(from_)}")
        if as_ is not None and not isinstance(as_, str):
            raise TypeError(f"as_ must be str, not {type(as_)}")
        if as_ is None and "." in import_:
            raise ValueError("Cannot use '.' in import_ or from_ without as_")
        super().__init__(inputs=[], output=Value)
        self.import_ = import_
        self.from_ = from_
        self.as_ = as_

    def _tracer_transform(self, transform):
        return Import(self.import_, self.from_, self.as_)

    def __eq__(self, other):
        if isinstance(other, Import):
            return self.import_ == other.import_ and self.from_ == other.from_ and self.as_ == other.as_
        return False


def import_(import_, as_=None, from_=None):
    return Import(import_, from_, as_).output


class OperatorApplication(tracer.Application):
    def __init__(self, operator, operands):
        if not isinstance(operator, str):
            raise TypeError(f"operator must be str, not {type(operator)}")
        super().__init__(inputs=[operator] + list(operands), output=Value)
        self.operator = operator
        self.operands = operands

    def _tracer_transform(self, transform):
        return OperatorApplication(self.operator, [transform(operand) for operand in self.operands])

    def __eq__(self, other):
        if isinstance(other, OperatorApplication):
            return self.operator == other.operator and self.operands == other.operands
        return False


def operator(operator, *operands):
    return OperatorApplication(operator, operands).output


def add(a, b):
    return operator("+", a, b)


def mul(a, b):
    return operator("*", a, b)


def less(a, b):
    return operator("<", a, b)


def less_equal(a, b):
    return operator("<=", a, b)


def greater(a, b):
    return operator(">", a, b)


def greater_equal(a, b):
    return operator(">=", a, b)


def equal(a, b):
    return operator("==", a, b)


def not_equal(a, b):
    return operator("!=", a, b)


class Builtin(tracer.Application):
    def __init__(self, name):
        if not isinstance(name, str):
            raise TypeError(f"name must be str, not {type(name)}")
        super().__init__(inputs=[], output=Value)
        self.name = name

    def _tracer_transform(self, transform):
        return Builtin(self.name)

    def __eq__(self, other):
        if isinstance(other, Builtin):
            return self.name == other.name
        return False


class Builtins:
    def __getattr__(self, name):
        return Builtin(name).output


builtins = Builtins()


class Assert(tracer.Application):
    def __init__(self, xs, condition, message=None):
        super().__init__(inputs=list(pytree.flatten(xs)) + [condition], output=lambda origin: pytree.map(lambda x: x._tracer_type(origin), xs))
        self.xs = xs
        self.condition = condition
        self.message = message
        if message is not None and not isinstance(message, str):
            raise TypeError(f"message must be str or None, not {type(message)}")

    def _tracer_transform(self, transform):
        return Assert(pytree.map(transform, self.xs), transform(self.condition), self.message)

    def __eq__(self, other):
        if isinstance(other, Assert):
            return self.xs == other.xs and self.condition == other.condition and self.message == other.message
        return False


def assert_(xs, condition, message=None):
    return Assert(xs, condition, message).output


class Constant(tracer.Application):
    def __init__(self, value):
        super().__init__(inputs=[], output=Value)
        self.value = value

    def _tracer_transform(self, transform):
        return Constant(self.value)

    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value == other.value
        return False


def constant(value):
    return Constant(value).output


class Value(tracer.Tracer):
    def __init__(self, origin):
        super().__init__(origin=origin)

    @property
    def _tracer_type(self):
        return Value

    def __eq__(self, other):
        if isinstance(other, Value):
            return self.origin == other.origin
        return False

    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            return object.__getattribute__(self, key)
        else:
            return getattr(self, key)

    def __getitem__(self, key):
        return getitem(self, key)

    def __call__(self, *args, **kwargs):
        return call(self, args, kwargs)


def function(func, args=None, kwargs=None):
    if args is None and kwargs is None:
        signature = inspect.signature(func)
        args = []
        kwargs = {}
        for param in signature.parameters.values():
            if param.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]:
                args.append(tracer.signature.python.Value(origin=None))
            elif param.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs[param.name] = tracer.signature.python.Value(origin=None)
            else:
                raise ValueError(f"Unsupported parameter kind: {param.kind}")
    elif args is None:
        args = []
    elif kwargs is None:
        kwargs = {}

    input_tracers = list(args) + list(kwargs.values())
    for value in input_tracers:
        if not isinstance(value, tracer.Tracer):
            raise TypeError(f"All arguments must be Tracer instances, got {type(value)}")

    output_tracer = func(*args, **kwargs)

    graph = tracer.Graph(inputs=input_tracers, output=output_tracer)
    return graph

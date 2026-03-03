from einx._src.util import pytree
import threading


class Tracer:
    def __init__(self, origin):
        if origin is not None and not isinstance(origin, Application):
            raise TypeError(f"origin must be Application, not {type(origin)}")
        self.origin = origin


class Graph:
    def __init__(self, inputs, output, name=None):
        self.inputs = inputs
        self.output = output
        self.name = name

    def __eq__(self, other):
        if not isinstance(other, Graph):
            return False
        if self.name != other.name:
            return False
        if self.inputs != other.inputs:
            return False
        if self.output != other.output:
            return False
        return True


_dependon = threading.local()


class DependOn:
    def __init__(self, dependencies):
        self.dependencies = list(dependencies)

    def __enter__(self):
        if not hasattr(_dependon, "stack"):
            _dependon.stack = []
        _dependon.stack.append(self.dependencies)

    def __exit__(self, exc_type, exc_value, traceback):
        _dependon.stack.pop()


def depend_on(*dependencies):
    return DependOn(dependencies)


def get_additional_dependencies():
    result = []
    if hasattr(_dependon, "stack"):
        for deps in _dependon.stack:
            result.extend(deps)
    return result


class Application:
    def __init__(self, inputs, output):
        self.output = output(origin=self)
        self.inputs = inputs


def depends_on(x, predecessor):
    if isinstance(x, Tracer) and isinstance(predecessor, Tracer):
        if id(x) == id(predecessor):
            return True
        elif x.origin is not None:
            return any(depends_on(inp, predecessor) for inp in x.origin.inputs)
        else:
            return False
    else:
        return False


class Cast(Application):
    def __init__(self, input, output):
        super().__init__(inputs=[input], output=output)
        self.input = input
        if not isinstance(input, Tracer):
            raise TypeError(f"input must be Tracer, not {type(input)}")

    def _tracer_transform(self, transform):
        return Cast(transform(self.input), lambda origin: pytree.map(lambda x: x._tracer_type(origin), self.output))

    def __eq__(self, other):
        if not isinstance(other, Cast):
            return False
        if self.input != other.input:
            return False
        if pytree.map(lambda x: x._tracer_type, self.output) != pytree.map(lambda x: x._tracer_type, other.output):
            return False
        return True


def cast(input, output):
    return Cast(input, output).output

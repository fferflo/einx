import einx
import inspect
from functools import partial
import threading


class Application:
    def __init__(self, op, args, kwargs, output, signature, inplace_updates, comment, depend_on):
        self.op = op
        self.args = args
        self.kwargs = kwargs
        self.output = Tracer() if output is None else output
        self.signature = signature
        self.inplace_updates = inplace_updates
        self.comment = comment
        self.depend_on = depend_on

        def update_origin(tracer, key):
            tracer.origin = self

        einx.tree_util.tree_map_with_key(update_origin, self.output)

        # TODO: move this somewhere else?
        if signature == "reshape":
            params = inspect.getcallargs(lambda tensor, shape: None, *args, **kwargs)
            self.shape = params["shape"]
            self.tensor = params["tensor"]
        elif signature == "broadcast_to":
            params = inspect.getcallargs(lambda tensor, shape: None, *args, **kwargs)
            self.shape = params["shape"]
            self.tensor = params["tensor"]
        elif signature == "transpose":
            params = inspect.getcallargs(lambda tensor, permutation: None, *args, **kwargs)
            self.permutation = params["permutation"]
            self.tensor = params["tensor"]

    @property
    def dependencies(self):
        return [self.op] + list(self.args) + list(self.kwargs.values()) + self.depend_on


def apply(
    op,
    args=[],
    kwargs={},
    output=None,
    signature=None,
    inplace_updates=[],
    comment=None,
    depend_on=[],
):
    if isinstance(op, partial):
        return apply(
            op.func,
            args=list(op.args) + list(args),
            kwargs={**op.keywords, **kwargs},
            output=output,
            signature=signature,
            inplace_updates=inplace_updates,
            comment=comment,
            depend_on=depend_on,
        )
    elif isinstance(op, TracableFunction):
        assert len(inplace_updates) == 0
        got_output = op(*args, **kwargs)
        if not output is None:

            def check(got_output, expected_output):
                if type(got_output) != type(expected_output):
                    # TODO: also compare shape etc
                    raise ValueError(
                        f"Expected output type {type(expected_output)} when tracing "
                        f"TracableFunction, got {type(got_output)}"
                    )

            einx.tree_util.tree_map(check, got_output, output)
        return got_output
    else:
        return Application(
            op,
            args=args,
            kwargs=kwargs,
            output=output,
            signature=signature,
            inplace_updates=inplace_updates,
            comment=comment,
            depend_on=depend_on + _get_depend_on_stack(),
        ).output


_thread_local = threading.local()


def _get_depend_on_stack():
    if not hasattr(_thread_local, "depend_on"):
        _thread_local.depend_on = []
    return _thread_local.depend_on


class depend_on:
    def __init__(self, tracers):
        self.tracer = list(einx.tree_util.tree_flatten(tracers))

    def __enter__(self):
        _get_depend_on_stack().append(self.tracer)

    def __exit__(self, *args):
        assert _get_depend_on_stack()[-1] is self.tracer
        _get_depend_on_stack().pop()


class Tracer:
    def __init__(self, origin=None):
        self.origin = origin

    def __getattr__(self, key):
        if key.startswith("__") and key.endswith("__"):
            return object.__getattribute__(self, key)
        else:
            return MemberAccess()(self, key)

    def __getitem__(self, key):
        return GetAt()(self, key)

    def __call__(self, *args, **kwargs):
        return apply(self, args=args, kwargs=kwargs)

    def __copy__(self):
        assert type(self) == Tracer
        return Tracer()


class Import(Tracer):
    def __init__(self, import_, as_, from_):
        Tracer.__init__(self, origin="constant")
        self.import_ = import_
        self.as_ = as_
        self.from_ = from_

    def __call__(self):  # Overwrite allowed arguments
        return apply(self)


def import_(import_, as_=None, from_=None):
    return Import(import_, as_, from_)()


class MemberAccess(Tracer):
    def __init__(self):
        Tracer.__init__(self, origin="constant")

    def __call__(self, obj, key):  # Overwrite allowed arguments
        assert isinstance(key, str)
        return apply(self, args=[obj, key])


class Operator(Tracer):
    def __init__(self, op: str):
        Tracer.__init__(self, origin="constant")
        self.op = op

    def __call__(self, *args):  # Overwrite allowed arguments
        return apply(self, args=args)


class AssignAt(Tracer):
    def __init__(self, op: str):
        Tracer.__init__(self, origin="constant")
        self.op = op

    def __call__(self, obj, key, update):  # Overwrite allowed arguments
        return apply(self, args=[obj, key, update])


class GetAt(Tracer):
    def __init__(self):
        Tracer.__init__(self, origin="constant")

    def __call__(self, obj, key):  # Overwrite allowed arguments
        return apply(self, args=[obj, key])


class Function(Tracer):
    def __init__(self, output):
        self.output = output

    def __copy__(self):
        return Function(self.output)

    def __call__(self, *args, **kwargs):
        return apply(
            self,
            args=args,
            kwargs=kwargs,
            output=einx.tree_util.tree_map(lambda x: x.__copy__(), self.output),
        )


class TracableFunction(Tracer):
    def __init__(self, func=None, args=None, kwargs=None, virtual_args=[], output=None, name=None):
        Tracer.__init__(self)

        if isinstance(func, Tracer):
            raise ValueError(f"func cannot be a tracer object")
        if not output is None and args is None and kwargs is None:
            raise ValueError(f"Cannot create a TracableFunction with an output but no input")

        if args is None and not kwargs is None:
            args = []
        if not args is None and kwargs is None:
            kwargs = {}

        if not func is None and output is None and (not args is None or not kwargs is None):
            output = func(*args, **kwargs)

        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.virtual_args = virtual_args
        self.output = output
        self.name = name

    def __call__(self, *args, **kwargs):
        if self.func is None:
            raise NotImplementedError(
                f"Cannot call a TracableFunction that was created without a callable function"
            )
        return self.func(*args, **kwargs)


class Usages:
    def __init__(self, tracers):
        self.usages = {}  # tracer-id: [using-applications]

        def _capture_usages(x):
            if not id(x) in self.usages:
                self.usages[id(x)] = []
            if isinstance(x, (list, tuple)):
                for y in x:
                    _capture_usages(y)
            elif isinstance(x, dict):
                for y in x.values():
                    _capture_usages(y)
            elif isinstance(x, Tracer) and isinstance(x.origin, Application):
                for y in x.origin.dependencies:
                    if isinstance(y, Tracer):
                        # Add x.origin to y's usages
                        if not id(y) in self.usages:
                            self.usages[id(y)] = []
                        for usage in self.usages[id(y)]:
                            if id(usage) == id(x.origin):
                                break
                        else:
                            self.usages[id(y)].append(x.origin)
                    # Continue capturing usages with y
                    _capture_usages(y)
            elif isinstance(x, TracableFunction):
                _capture_usages(x.func)
                _capture_usages(x.args)
                _capture_usages(x.kwargs)
                _capture_usages(x.output)

        _capture_usages(tracers)

    def get(self, tracers):
        usages = []

        def retrieve_usages(tracer):
            if id(tracer) in self.usages:
                usages.extend(self.usages[id(tracer)])

        einx.tree_util.tree_map(retrieve_usages, tracers)
        return usages


def trace(func=None, args=None, kwargs=None):
    if func is None:
        return partial(trace, args=args, kwargs=kwargs)
    else:
        return TracableFunction(func=func, args=args, kwargs=kwargs)

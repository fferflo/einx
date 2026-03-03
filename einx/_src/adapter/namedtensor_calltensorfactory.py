from einx._src.namedtensor import NamedTensor
import einx._src.tracer as tracer
from collections.abc import Callable
import functools
import inspect
import types
from einx._src.util.functools import use_name_of
import contextlib


def _call_tensorfactory(tensor, kwargs):
    expr = tensor.expr
    tensor = tensor.value
    called = False

    if isinstance(tensor, tracer.signature.classical.ConvertibleTensor) and issubclass(tensor.concrete.type, Callable):
        # Determine arguments that are passed to the tensor factory
        shape = tuple(tensor.shape)
        if kwargs is None:
            kwargs = {}
        has_var_kwargs = any(param.kind in [inspect.Parameter.VAR_KEYWORD] for param in tensor.concrete.parameters.values())

        def use_parameter(name):
            return has_var_kwargs or (
                name in tensor.concrete.parameters
                and tensor.concrete.parameters[name].kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]
            )

        kwargs = {name: value for name, value in kwargs.items() if use_parameter(name)}

        # Call the factory
        tensor = tracer.signature.python.call(tensor, args=[shape], kwargs=kwargs)
        called = True

    return tensor, expr, called


def _assert_output(tensor, expr, called, expected_type):
    if called:
        if callable(expected_type) and not isinstance(expected_type, type) and not isinstance(expected_type, tracer.Tracer):
            expected_type = expected_type()

        tensor = tracer.signature.python.assert_(
            tensor,
            tracer.signature.python.builtins.isinstance(tensor, expected_type),
            "Invalid type as output of tensor factory",  # TODO:
        )
        tensor = tracer.signature.python.assert_(
            tensor,
            tracer.signature.python.equal(tracer.signature.python.builtins.tuple(tensor.shape), expr.shape),
            f"Expected shape {expr.shape} as output of tensor factory",  # TODO:
        )
        tensor = tracer.cast(tensor, lambda origin: tracer.signature.classical.Tensor(origin, shape=expr.shape))

    return NamedTensor(tensor, expr)


class namedtensor_calltensorfactory:
    @staticmethod
    def op(op, expected_type, context=None, kwargs=None):
        if context is None:
            context = lambda *args: contextlib.nullcontext()
        factory_kwargs = kwargs if kwargs is not None else {}

        @use_name_of(op)
        def inner(*tensors, out, **kwargs):
            signature = types.SimpleNamespace(exprs_in=tuple(t.expr for t in tensors), exprs_out=tuple(out) if isinstance(out, tuple | list) else (out,))
            signature = tracer.signature.python.constant(signature)
            xs = [
                _call_tensorfactory(tensor, kwargs={"signature": signature, "arg_index": arg_index} | factory_kwargs)
                for arg_index, tensor in enumerate(tensors)
            ]

            with context([tensor for tensor, expr, called in xs]):
                tensors = [_assert_output(tensor, expr, called, expected_type) for tensor, expr, called in xs]

                return op(*tensors, out=out, **kwargs)

        return inner

    @staticmethod
    def ops(ops, expected_type, context=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        return {name: namedtensor_calltensorfactory.op(op, expected_type, kwargs={"name": name} | kwargs, context=context) for name, op in ops.items()}

from einx._src.util.functools import use_name_of
import threading
from contextlib import contextmanager
import einx._src.tracer as tracer
import types


class ArrayApiNamespaceStack:
    def __init__(self):
        self._thread_local = threading.local()
        self.namedtensor = types.SimpleNamespace(op=self._wrap_namedtensor_op, ops=self._wrap_namedtensor_ops)
        self.array_api_compat = tracer.signature.python.import_("array_api_compat")

    def get_xp(self):
        stack = self._get_stack()
        assert len(stack) > 0
        return stack[-1]

    def _wrap_namedtensor_op(self, op):
        @use_name_of(op)
        def inner(*tensors, out, **kwargs):
            self._enter([t.value for t in tensors])
            try:
                return op(*tensors, out=out, **kwargs)
            finally:
                self._exit([t.value for t in tensors])

        return inner

    def _wrap_namedtensor_ops(self, ops):
        return {name: self._wrap_namedtensor_op(op) for name, op in ops.items()}

    def _get_stack(self):
        if not hasattr(self._thread_local, "stack"):
            self._thread_local.stack = []
        return self._thread_local.stack

    def _enter(self, tensors):
        xp = self.array_api_compat.array_namespace(*tensors)
        stack = self._get_stack()
        stack.append(xp)

    def _exit(self, tensors):
        stack = self._get_stack()
        assert len(stack) > 0
        stack.pop()

    @contextmanager
    def __call__(self, tensors):
        self._enter(tensors)
        try:
            yield
        finally:
            self._exit(tensors)

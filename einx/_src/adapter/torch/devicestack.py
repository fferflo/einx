import einx._src.tracer as tracer
import threading
import types
from einx._src.util.functools import use_name_of


class TorchDeviceStack:
    def __init__(self):
        self._thread_local = threading.local()
        self.namedtensor = types.SimpleNamespace(op=self._wrap_namedtensor_op, ops=self._wrap_namedtensor_ops)

    def get_device(self):
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
        device = None
        for tensor in tensors:
            if isinstance(tensor, tracer.signature.classical.Tensor):
                device = tracer.signature.python.getattr(tensor, "device")
                break
        # if device is None:
        #     raise ValueError("Failed to determine the PyTorch device placement of parameters. Maybe convert the given arguments to a tensor first.")
        stack = self._get_stack()
        stack.append(device)

    def _exit(self, tensors):
        stack = self._get_stack()
        assert len(stack) > 0
        stack.pop()

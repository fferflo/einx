import einx._src.tracer as tracer
import einx._src.util.pytree as pytree
from ._util import _skip_id


class InlineGraph:
    def __call__(self, x, transform):
        if isinstance(x, tracer.Graph):
            output = _skip_id(x.output)
            if isinstance(output, tracer.Tracer) and isinstance(output.origin, tracer.signature.python.Call) and len(output.origin.kwargs) == 0:
                function_inputs = [_skip_id(i) for i in output.origin.args]
                graph_inputs = x.inputs

                if [id(i) for i in graph_inputs] != [id(i) for i in function_inputs]:
                    # Function is not called with the same inputs as the graph
                    return False, None

                if any(tracer.depends_on(output.origin.function, input) for input in graph_inputs):
                    # Function depends on graph inputs
                    return False, None

                return True, transform(output.origin.function)

        return False, None


class SkipCast:
    def _is_result_of_call(self, x):
        return isinstance(x, tracer.Tracer) and isinstance(x.origin, tracer.Cast)

    def __call__(self, x, transform):
        if self._is_result_of_call(x):
            input = x.origin.input
            output = x.origin.output
            input_signature = pytree.map(lambda x: x._tracer_type(None), input)
            output_signature = pytree.map(lambda x: x._tracer_type(None), output)

            if input_signature == output_signature:
                # Skip cast
                return True, transform(input)
            # TODO: merge consecutive casts

        return False, None

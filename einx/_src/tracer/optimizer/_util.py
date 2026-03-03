from einx._src.util import pytree
import einx._src.tracer as tracer


def _skip_id(output):
    if isinstance(output, tuple):
        output = tuple(_skip_id(i) for i in output)
    elif isinstance(output, list):
        output = [_skip_id(i) for i in output]
    elif isinstance(output, dict):
        output = {k: _skip_id(v) for k, v in output.items()}

    origins = [x.origin for x in pytree.flatten(output) if isinstance(x, tracer.Tracer)]
    if len(origins) == 0:
        return output
    origin = origins[0]

    if isinstance(origin, tracer.Cast) and pytree.all(lambda x, y: id(x) == id(y), origin.output, output):
        return _skip_id(origin.input)
    else:
        return output

import einx._src.tracer as tracer
from einx._src.util import pytree
import numpy as np


def visualize(x):
    from graphviz import Digraph

    dot = Digraph()

    pythonid_to_nodeid = {}

    next_node_id = 0

    def new_node_id(obj=None):
        nonlocal next_node_id
        node_id = next_node_id
        next_node_id += 1
        node_id = str(node_id)

        if obj is not None:
            assert id(obj) not in pythonid_to_nodeid
            pythonid_to_nodeid[id(obj)] = node_id

        return node_id

    op_kwargs = {"shape": "box", "style": "filled", "fillcolor": "#aa7777"}
    value_kwargs = {"shape": "oval", "style": "filled", "fillcolor": "#7777aa"}
    graph_kwargs = {"shape": "box", "style": "filled", "fillcolor": "#77aa77"}

    def _value_to_str(x):
        if isinstance(x, int | float | bool | np.floating | np.integer):
            return str(x)
        elif isinstance(x, str):
            return f'"{x}"'
        elif x is None:
            return "None"
        elif isinstance(x, list):
            return f"[{', '.join(map(_value_to_str, x))}]" + f" - {x}"
        elif isinstance(x, tuple):
            comma = "," if len(x) == 1 else ""
            return f"({', '.join(map(_value_to_str, x))}{comma})" + f" - {x}"
        elif isinstance(x, slice):
            start = _value_to_str(x.start) if x.start is not None else "None"
            stop = _value_to_str(x.stop) if x.stop is not None else "None"
            step = _value_to_str(x.step) if x.step is not None else "None"
            return f"slice({start}, {stop}, {step})" + f" - {x}"
        elif isinstance(x, tracer.signature.classical.Tensor):
            return f"Tensor({', '.join(map(_value_to_str, x.shape))})" + f" - {x}"
        elif isinstance(x, tracer.signature.classical.ConvertibleTensor):
            return f"ConvertibleTensor({', '.join(map(_value_to_str, x.shape)) if x.shape is not None else 'None'})" + f" - {x}"
        elif isinstance(x, tracer.Tracer):
            return "Tracer" + f" - {x}"
        elif isinstance(x, tracer.Graph):
            if x.name is None:
                return "Graph" + f" - {x}"
            else:
                return f'Graph("{x.name}")' + f" - {x}"
        else:
            raise NotImplementedError(f"Unsupported type: {type(x)}")

    def _op_to_str(op):
        return op.__class__.__name__

    def _add_op(text, inputs, outputs):
        op_id = new_node_id()
        dot.node(op_id, text, **op_kwargs)

        for input in inputs:
            input_id = _add_value(input)
            dot.edge(input_id, op_id)

        for output in outputs:
            output_id = new_node_id(output)
            dot.node(output_id, _value_to_str(output), **value_kwargs)
            dot.edge(op_id, output_id)

    def _add_value(x):
        if isinstance(x, str | int | float | bool | np.floating | np.integer) or x is None:
            node_id = new_node_id()
            dot.node(node_id, _value_to_str(x), **value_kwargs)
            return node_id

        if id(x) in pythonid_to_nodeid:
            return pythonid_to_nodeid[id(x)]

        if isinstance(x, tracer.Tracer):
            if x.origin is not None:
                _add_op(_op_to_str(x.origin), x.origin.inputs, pytree.flatten(x.origin.output))
                return pythonid_to_nodeid[id(x)]
            else:
                node_id = new_node_id(x)
                dot.node(node_id, _value_to_str(x), **value_kwargs)
                return node_id
        elif isinstance(x, list):
            _add_op("create-list", x, [x])
            return pythonid_to_nodeid[id(x)]
        elif isinstance(x, tuple):
            _add_op("create-tuple", x, [x])
            return pythonid_to_nodeid[id(x)]
        elif isinstance(x, slice):
            _add_op("create-slice", [x.start, x.stop, x.step], [x])
            return pythonid_to_nodeid[id(x)]
        elif isinstance(x, tracer.Graph):
            _add_value(x.output)
            graph_id = new_node_id(x)
            dot.node(graph_id, _value_to_str(x), **graph_kwargs)
            for input in x.inputs:
                input_id = _add_value(input)
                dot.edge(input_id, graph_id, style="dashed", color="blue")

            def _add_out_edge(output):
                output_id = _add_value(output)
                dot.edge(graph_id, output_id, style="dashed", color="red")

            pytree.map(_add_out_edge, x.output)
            return graph_id
        else:
            raise NotImplementedError(f"Unsupported type: {type(x)}")

    output_id = _add_value(x)
    dot.node(output_id, _value_to_str(x), shape="octagon", style="filled", fillcolor="#ffaa00")

    return dot

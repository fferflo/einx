from .tracer import *
from .tensor import *


def get_signature(node):
    if not node.origin is None:
        return node.origin.signature
    else:
        return None


class Optimizer:
    def __init__(self):
        self.optimized_nodes = {}
        self.changed = False

    def __call__(self, node):
        if id(node) in self.optimized_nodes:
            return self.optimized_nodes[id(node)]

        if isinstance(node, TracableFunction):
            if node.output is None:
                raise ValueError("Function output is None")
            new_node = TracableFunction(
                func=self(node.func),
                args=node.args,
                kwargs=node.kwargs,
                virtual_args=self(node.virtual_args),
                output=self(node.output),
            )
        elif isinstance(node, Tracer):
            if isinstance(node.origin, Application):
                if (
                    get_signature(node) == "reshape"
                    and get_signature(node.origin.tensor) == "reshape"
                ):
                    # Merge consecutive reshape ops
                    shape = node.origin.shape
                    new_node = apply(
                        self(node.origin.op),
                        [self(node.origin.tensor.origin.tensor), shape],
                        output=Tensor(shape),
                        signature="reshape",
                    )
                    self.changed = True
                elif (
                    get_signature(node) == "reshape"
                    and get_shape(node.origin.tensor) == node.origin.shape
                ):
                    # Skip reshape op if tensor already has right shape
                    new_node = self(node.origin.tensor)
                    self.changed = True
                elif (
                    get_signature(node) == "broadcast_to"
                    and get_shape(node.origin.tensor) == node.origin.shape
                ):
                    # Skip broadcast_to op if tensor already has right shape
                    new_node = self(node.origin.tensor)
                    self.changed = True
                elif get_signature(node) == "transpose" and list(node.origin.permutation) == list(
                    range(len(node.shape))
                ):
                    # Skip transpose op if permutation is identity
                    new_node = self(node.origin.tensor)
                    self.changed = True
                else:
                    # Optimize only arguments
                    new_output_nodes = einx.tree_util.tree_map(
                        lambda node: node.__copy__(), node.origin.output
                    )

                    def store(new_node, node):
                        assert not id(node) in self.optimized_nodes
                        self.optimized_nodes[id(node)] = new_node

                    einx.tree_util.tree_map(store, new_output_nodes, node.origin.output)
                    new_node = self.optimized_nodes[id(node)]

                    apply(
                        self(node.origin.op),
                        self(node.origin.args),
                        self(node.origin.kwargs),
                        output=new_output_nodes,
                        signature=node.origin.signature,
                        inplace_updates=[
                            (
                                self.optimized_nodes[id(tensor_in)],
                                self.optimized_nodes[id(tensor_out)],
                            )
                            for tensor_in, tensor_out in node.origin.inplace_updates
                        ],
                        comment=node.origin.comment,
                        depend_on=self(node.origin.depend_on),
                    )
            else:
                new_node = node
        elif isinstance(node, list):
            new_node = [self(x) for x in node]
        elif isinstance(node, tuple):
            new_node = tuple(self(x) for x in node)
        elif isinstance(node, dict):
            new_node = {k: self(v) for k, v in node.items()}
        else:
            new_node = node

        self.optimized_nodes[id(node)] = new_node
        return new_node


def optimize(node):
    while True:
        optimizer = Optimizer()
        node = optimizer(node)
        if not optimizer.changed:
            break

    return node

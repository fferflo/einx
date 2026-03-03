import einx._src.namedtensor.stage3 as stage3
from einx._src.namedtensor import NamedTensor


def _expr_to_einsumstr(exprs_in, expr_out, allow_repeat_axes=False):
    einsum_variables = {}

    next_ord = ord("a")

    def get_einsum_variable(key, make_new):
        if not make_new and key in einsum_variables:
            return einsum_variables[key]
        else:
            nonlocal next_ord
            if next_ord > ord("z"):
                raise ValueError(f"The function only supports up to {ord('z') - ord('a') + 1} unique input axes")
            v = chr(next_ord)
            next_ord += 1
            einsum_variables[key] = v
            return v

    def to_einsum(expr, make_new):
        axes = [a for a in expr if isinstance(a, stage3.Axis)]
        einsum_str = ""
        for a in axes:
            einsum_str += get_einsum_variable(a.name, make_new=make_new)
        return einsum_str

    einsum_str_in = ",".join(to_einsum(expr, make_new=allow_repeat_axes) for expr in exprs_in)
    einsum_str_out = to_einsum(expr_out, make_new=False)
    einsum_str = einsum_str_in + "->" + einsum_str_out
    return einsum_str


def _op(einsum, name, allow_repeat_axes=False):
    def op(*tensors, out):
        exprs_in = [t.expr for t in tensors]
        tensors = [t.value for t in tensors]

        tensor = einsum(_expr_to_einsumstr(exprs_in, out, allow_repeat_axes=allow_repeat_axes), *tensors)

        return NamedTensor(tensor, out)

    op.__name__ = name
    return op


def dot(einsum):
    return _op(einsum, "dot")


def sum(einsum):
    return _op(einsum, "sum", allow_repeat_axes=True)


def multiply(einsum):
    return _op(einsum, "multiply")


def id(einsum):
    def id(*tensors, out):
        if len(tensors) == 1:
            assert isinstance(out, stage3.Expression)
            out = [out]
        assert len(tensors) == len(out)

        def _single_id(tensor, out):
            expr = tensor.expr
            tensor = tensor.value

            # Ignore broadcasting axes -> are handled by namedtensor_from_decomposednamedtensor
            input_axis_names = {axis.name for axis in expr.nodes() if isinstance(axis, stage3.Axis)}
            out = stage3.List.create([axis for axis in out.nodes() if isinstance(axis, stage3.Axis) if axis.name in input_axis_names])

            einsum_str = _expr_to_einsumstr([expr], out)
            tokens = einsum_str.split("->")
            if tokens[0] != tokens[1]:
                tensor = einsum(_expr_to_einsumstr([expr], out), tensor)

            return NamedTensor(tensor, out)

        tensors = [_single_id(tensor, out) for tensor, out in zip(tensors, out, strict=False)]

        return tensors if len(tensors) > 1 else tensors[0]

    return id

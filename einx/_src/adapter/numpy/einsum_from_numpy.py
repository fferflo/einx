from .._util import _to_tensor


class _detachscalars:
    def __init__(self, einsum, multiply):
        self._einsum = einsum
        self._multiply = multiply

    def __call__(self, equation, *operands):
        exprs = equation.split("->")
        if len(exprs) != 2:
            raise ValueError("Invalid einsum equation")
        in_exprs = exprs[0].split(",")
        out_expr = exprs[1]

        # Remove scalars
        scalars = []
        for in_expr, operand in zip(in_exprs, operands, strict=False):
            if (len(in_expr) == 0) != (operand.shape == ()):
                raise ValueError(f"Tensor and einsum expression do not match: {in_expr} and {operand.shape}")
            if operand.shape == ():
                scalars.append(operand)
        operands = [operand for operand in operands if operand.shape != ()]
        in_exprs = [in_expr for in_expr in in_exprs if len(in_expr) > 0]
        assert len(in_exprs) == len(operands)
        equation = ",".join(in_exprs) + "->" + out_expr

        # Call without scalars
        if len(operands) == 1:
            if in_exprs[0] != out_expr:
                output = self._einsum(equation, *operands)
            else:
                output = operands[0]
        elif len(operands) > 1:
            output = self._einsum(equation, *operands)
        else:
            output = None

        # Multiply scalars
        if len(scalars) > 0:
            if output is None:
                output = self._multiply(*scalars)
            else:
                output = self._multiply(output, *scalars)

        return output


def einsum_from_numpy(op=None, to_tensor=None, multiply=None):
    if to_tensor is None:
        np = op
        op = np.einsum

        def to_tensor(*args):
            to_tensor_one = _to_tensor(np.asarray, forward=["numpy", "scalar"], convert=[])
            return [to_tensor_one(arg) for arg in args]

    def einsum(subscripts, *operands):
        operands = to_tensor(*operands)
        return op(subscripts, *operands)

    if multiply is not None:
        einsum = _detachscalars(einsum, multiply)
    return einsum

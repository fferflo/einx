def _einsum_diag_string(ndim, axes_in, axis_out):
    _next_ord = 1

    def get_input_name_at(idx):
        nonlocal _next_ord
        if idx in axes_in:
            return "a"
        else:
            result = chr(ord("a") + _next_ord)
            _next_ord += 1
            return result

    einsum_str_input = "".join(get_input_name_at(i) for i in range(ndim))

    _next_ord = 1

    def get_output_name_at(idx):
        nonlocal _next_ord
        if idx == axis_out:
            return "a"
        else:
            result = chr(ord("a") + _next_ord)
            _next_ord += 1
            return result

    einsum_str_output = "".join(get_output_name_at(i) for i in range(ndim - len(axes_in) + 1))

    return einsum_str_input + "->" + einsum_str_output


def diagonal(einsum):
    def diagonal(x, axes_in, axis_out):
        einsum_str = _einsum_diag_string(x.ndim, axes_in, axis_out)
        return einsum(einsum_str, x)

    return diagonal

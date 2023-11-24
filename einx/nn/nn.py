import einx

def meanvar_norm(x, stats, params="b... [c]", moving_average=None, epsilon=0, mean=True, var=True, scale=None, bias=None, fastvar=True):
    if moving_average is None:
        moving_average = lambda f, name: f()
    backend = einx.backend.get([x])

    # Compute mean and variance
    if mean:
        mean = backend.cast(moving_average(lambda: einx.mean(stats, x), name="mean"), x.dtype)
    else:
        mean = None
    if var:
        if mean is None:
            var = lambda: einx.mean(stats, backend.square(x))
        else:
            if fastvar:
                def var():
                    mean_of_squares = einx.mean(stats, backend.square(x))
                    var = mean_of_squares - backend.square(mean)
                    var = backend.maximum(var, 0)
                    return var
            else:
                var = lambda: einx.var(stats, x)
        var = backend.cast(moving_average(var, name="var"), x.dtype)
        inv_std = backend.rsqrt(var + epsilon)
    else:
        inv_std = None

    # Normalize mean and variance
    (expr_in,), (expr_stats,) = einx.reduce.parse(stats, einx.param.get_shape(x))
    expr_in = einx.expr.stage3.demark(expr_in)
    expr_stats = einx.expr.stage3.demark(expr_stats)
    if not mean is None:
        x, _ = einx.subtract([expr_in, expr_stats], [x, mean], expr_in)
    if not inv_std is None:
        x, _ = einx.multiply([expr_in, expr_stats], [x, inv_std], expr_in)

    # TODO: need optimizer like opt_einsum that can optimize elementwise expressions like: (x - mean) * scale * inv_std + bias

    # Apply scale and bias
    if not scale is None:
        x = einx.multiply(params, x, scale)
    if not bias is None:
        x = einx.add(params, x, bias)

    return x

def linear(x, expr, weight, bias=None, **kwargs):
    (expr_in1, expr_in2), expr_afterdot = einx.dot.parse(expr, einx.param.get_shape(x), einx.param.get_shape(weight), **kwargs)

    # Weight matrix multiplication
    x = einx.dot(expr, x, weight, **kwargs)

    if not bias is None:
        # Bias expression includes all axes in output that are also in weight matrix
        weight_axes_names = set(a.name for a in expr_in2.all() if isinstance(a, einx.expr.stage3.Axis))
        expr_bias = []
        for a in expr_afterdot.all():
            if isinstance(a, einx.expr.stage3.Axis) and a.name in weight_axes_names:
                expr_bias.append(a.__deepcopy__())
        expr_bias = einx.expr.stage3.List(expr_bias)

        x, _ = einx.add([expr_afterdot, expr_bias], [x, bias], expr_afterdot)

    return x
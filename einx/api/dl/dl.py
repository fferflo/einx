import einx

def meanvar_norm(x, stats, params="b... [c]", moving_average=None, epsilon=0, mean=True, var=True, scale=None, bias=None, fastvar=True):
    if moving_average is None:
        moving_average = lambda f, name: f()
    backend = einx.backend.get([x])

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

    expr_in, expr_out = einx.reduce.parse(stats, einx.api.util.get_shape(x))
    if not mean is None:
        x = einx.subtract(f"{expr_in}, {expr_out}", x, mean)
    if not inv_std is None:
        x = einx.multiply(f"{expr_in}, {expr_out}", x, inv_std)

    # TODO: need optimizer like opt_einsum that can optimize elementwise expressions like: (x - mean) * scale * inv_std + bias

    if not scale is None:
        x = einx.multiply(params, x, scale)
    if not bias is None:
        x = einx.add(params, x, bias)

    return x

def linear(x, expr, weight, bias=None, **kwargs):
    (expr_in1, expr_in2), expr_afterdot = einx.dot.parse(expr, einx.api.util.get_shape(x), einx.api.util.get_shape(weight), **kwargs)

    x = einx.dot(expr, x, weight, **kwargs)

    if not bias is None:
        def remove(n):
            return isinstance(n, einx.expr.stage3.Variable) and not any(n == v for v in expr_in2.traverse())
        expr_bias = einx.expr.stage3.remove(expr_afterdot, remove, drop_empty_groups=True)
        x = einx.op.add([einx.op.Tensor(x, expr_afterdot), einx.op.Tensor(bias, expr_bias)]).value

    return x
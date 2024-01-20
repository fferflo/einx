import einx
from typing import Union, Optional, Any

@einx.lru_cache(trace=lambda t, c: lambda x, stats, params="b... [c]", mean=True, var=True, scale=None, bias=None, epsilon=0, fastvar=True, backend=None, **kwargs:
    c(t(x), stats, params, t(mean) if not isinstance(mean, bool) and not mean is None else mean, t(var) if not isinstance(var, bool) and not var is None else var,
        t(scale) if not scale is None else scale, t(bias) if not bias is None else bias, epsilon, fastvar, **kwargs))
def norm(x: einx.Tensor, stats: str, params: str = "b... [c]", mean: Union[einx.Tensor, bool] = True, var: Union[einx.Tensor, bool] = True, scale: Optional[einx.Tensor] = None, bias: Optional[einx.Tensor] = None, epsilon: float = 0, fastvar: bool = True, backend: Union[einx.Backend, str, None] = None, **kwargs: Any):
    if mean is None or var is None:
        raise ValueError("mean and var cannot be None")
    if backend is None:
        backend = einx.backend.get([x, mean if not isinstance(mean, bool) else None, var if not isinstance(var, bool) else None, scale, bias])
    elif isinstance(backend, str):
        backend = einx.backend.get(backend)

    expr_in, expr_stats = einx.reduce.parse(stats, einx.param.get_shape(x), **kwargs)
    expr_in = einx.expr.stage3.demark(expr_in)
    expr_stats = einx.expr.stage3.demark(expr_stats)

    # Instantiate moving averages
    if not isinstance(mean, bool) and not mean is None:
        mean = einx.param.instantiate(mean, shape=expr_stats.shape, backend=backend, init="add")
    if not isinstance(var, bool) and not var is None:
        var = einx.param.instantiate(var, shape=expr_stats.shape, backend=backend, init="multiply")

    # Compute mean and variance
    if isinstance(mean, bool):
        if mean:
            mean = einx.mean(stats, x, **kwargs)
        else:
            mean = None
    if isinstance(var, bool):
        if var:
            if mean is None:
                # RMS norm
                var = einx.mean(stats, backend.square(x), **kwargs)
            else:
                if fastvar:
                    mean_of_squares = einx.mean(stats, backend.square(x), **kwargs)
                    var = mean_of_squares - backend.square(mean)
                    var = backend.maximum(var, 0)
                else:
                    var = einx.var(stats, x, **kwargs)
        else:
            var = None

    # Normalize mean and variance
    if not mean is None:
        x, _ = einx.subtract_stage3([expr_in, expr_stats], [x, mean], expr_in)
    if not var is None:
        inv_std = backend.rsqrt(var + epsilon)
        x, _ = einx.multiply_stage3([expr_in, expr_stats], [x, inv_std], expr_in)

    # Apply scale and bias
    if not scale is None:
        x = einx.multiply(params, x, scale, **kwargs)
    if not bias is None:
        x = einx.add(params, x, bias, **kwargs)

    return x, mean, var

@einx.lru_cache(trace=lambda t, c: lambda x, expr, weight, bias=None, **kwargs:
    c(t(x), expr, t(weight), t(bias) if not bias is None else None, **kwargs))
def linear(x: einx.Tensor, expr: str, weight: einx.Tensor, bias: Optional[einx.Tensor], **kwargs: Any):
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

        x, _ = einx.add_stage3([expr_afterdot, expr_bias], [x, bias], expr_afterdot)

    return x

@einx.lru_cache(trace=lambda t, c: lambda x, expr, drop_rate, rng=None, **kwargs:
    c(t(x), expr, drop_rate, t(rng) if not rng is None else None, **kwargs))
def dropout(x: einx.Tensor, expr: str, drop_rate: float, rng: Any = None, **kwargs: Any):
    backend = einx.backend.get([x])
    keep_rate = 1 - drop_rate

    (expr_in, expr_mask), expr_out = einx.elementwise.parse(expr, einx.param.get_shape(x), None, **kwargs)

    drop_mask = backend.random.bernoulli(rng=rng, p=keep_rate, shape=expr_mask.shape)
    x, _ = einx.where_stage3([expr_mask, expr_in, einx.expr.stage3.List([])], [drop_mask, x, 0], expr_out)

    return x / keep_rate

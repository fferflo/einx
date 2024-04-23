import einx
import jax
import equinox as eqx
from functools import partial
import jax.numpy as jnp
from typing import Optional, Callable, Any

# TODO: type annotations

tjax = einx.tracer.import_("jax")


def create_or_retrieve(concrete, name, shape, dtype, init):
    if name in vars(concrete.module) and vars(concrete.module)[name] is not None:
        tensor = vars(concrete.module)[name]
    else:
        tensor = vars(concrete.module)[name] = init(concrete.rng, shape, dtype)
    return tensor


class ParamFactory:
    class Concrete(einx.tracer.input.Input):
        def __init__(self, module, name, init, dtype, rng):
            self.module = module
            self.name = name
            self.init = init

            if dtype is None:
                if hasattr(module, "dtype"):
                    dtype = module.dtype
                else:
                    dtype = "float32"
            self.dtype = dtype

            self.rng = rng

        def to_value_and_key(self):
            return self, ParamFactory.CacheKey(self.name, self.init, self.dtype)

    class CacheKey(einx.tracer.input.CacheKey):
        def __init__(self, name, init, dtype):
            self.name = name
            self.init = init
            self.dtype = dtype

        def __hash__(self):
            return hash((self.name, self.init, self.dtype))

        def __eq__(self, other):
            return (
                isinstance(other, ParamFactory.CacheKey)
                and self.name == other.name
                and self.init == other.init
                and self.dtype == other.dtype
            )

        def to_tracer(self, backend, virtual_arg):
            x = ParamFactory.Tracer(self.name, self.init, self.dtype)
            return x, x

    class Tracer(einx.tracer.TensorFactory):
        def __init__(self, name, init, dtype):
            self.name = name
            self.init = init
            self.dtype = dtype

        def __call__(self, shape, kwargs):
            name = self.name if not self.name is None else kwargs.get("name", None)
            init = self.init if not self.init is None else kwargs.get("init", None)
            dtype = self.dtype if not self.dtype is None else kwargs.get("dtype", None)

            if name is None:
                raise ValueError("Must specify name for tensor factory eqx.Module")

            if init is None:
                raise ValueError("Must specify init for tensor factory eqx.Module")
            elif isinstance(init, str):
                if init == "get_at" or init == "rearrange":
                    init = tjax.nn.initializers.normal(stddev=0.02)
                elif init == "add":
                    init = tjax.nn.initializers.constant(0.0, dtype=dtype)
                elif init == "multiply":
                    init = tjax.nn.initializers.constant(1.0, dtype=dtype)
                elif init == "dot":
                    init = tjax.nn.initializers.lecun_normal(
                        kwargs["in_axis"], kwargs["out_axis"], kwargs["batch_axis"]
                    )
                else:
                    raise ValueError(f"Don't know which initializer to use for operation '{init}'")
            elif isinstance(init, (int, float)):
                init = tjax.nn.initializers.constant(init, dtype=dtype)

            return einx.tracer.apply(
                create_or_retrieve,  # TODO: make tracable
                args=[self, name, shape, dtype, init],
                output=einx.tracer.Tensor(shape),
            )


def param(module, name=None, init=None, dtype=None, rng=None):
    """Create a tensor factory for Equinox parameters.

    Args:
        module: The module to create the parameter in. Must be an instance of ``eqx.Module``.
        name: Name of the parameter. If ``None``, uses a default name determined from the calling
            operation. Defaults to ``None``.
        init: Initializer for the parameter. If ``None``, uses a default init method determined
            from the calling operation. Defaults to ``None``.
        dtype: Data type of the parameter. If ``None``, uses the ``dtype`` member of the calling
            module or ``float32`` if it does not exist. Defaults to ``None``.

    Returns:
        A tensor factory with the given default parameters.
    """
    return ParamFactory.Concrete(module, name, init, dtype, rng)


class Norm(eqx.Module):
    """Normalization layer.

    Args:
        stats: Einstein string determining the axes along which mean and variance are computed.
            Will be passed to ``einx.reduce``.
        params: Einstein string determining the axes along which learnable parameters are applied.
            Will be passed to ``einx.elementwise``. Defaults to ``"b... [c]"``.
        mean: Whether to apply mean normalization. Defaults to ``True``.
        var: Whether to apply variance normalization. Defaults to ``True``.
        scale: Whether to apply a learnable scale according to ``params``. Defaults to ``True``.
        bias: Whether to apply a learnable bias according to ``params``. Defaults to ``True``.
        epsilon: A small float added to the variance to avoid division by zero. Defaults
            to ``1e-5``.
        fastvar: Whether to use a fast variance computation. Defaults to ``True``.
        dtype: Data type of the weights. Defaults to ``"float32"``.
        decay_rate: Decay rate for exponential moving average of mean and variance. If ``None``,
            no moving average is applied. Defaults to ``None``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    stats: str
    params: str
    mean: bool
    var: bool
    use_scale: bool
    use_bias: bool
    scale: Optional[jax.Array]
    bias: Optional[jax.Array]
    decay_rate: Optional[float]
    epsilon: float
    fastvar: bool
    dtype: str
    kwargs: dict

    def __init__(
        self,
        stats: str,
        params: str = "b... [c]",
        mean: bool = True,
        var: bool = True,
        scale: bool = True,
        bias: bool = True,
        decay_rate: Optional[float] = None,
        epsilon: float = 1e-5,
        fastvar: bool = True,
        dtype: Any = "float32",
        **kwargs: Any,
    ):
        if decay_rate is not None:
            raise ValueError("Stateful layers are currently not supported in Equinox")
        self.stats = stats
        self.params = params
        self.mean = mean
        self.var = var
        self.use_scale = scale
        self.use_bias = bias
        self.scale = None
        self.bias = None
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.fastvar = fastvar
        self.dtype = dtype
        self.kwargs = kwargs

    def __call__(self, x, rng=None):
        x, _mean, _var = einx.nn.norm(
            x,
            self.stats,
            self.params,
            mean=self.mean,
            var=self.var,
            scale=param(self, name="scale", rng=rng) if self.use_scale else None,
            bias=param(self, name="bias", rng=rng) if self.use_bias else None,
            epsilon=self.epsilon,
            fastvar=self.fastvar,
            **self.kwargs,
        )
        return x


class Linear(eqx.Module):
    """Linear layer.

    Args:
        expr: Einstein string determining the axes along which the weight matrix is multiplied.
            Will be passed to ``einx.dot``.
        bias: Whether to apply a learnable bias. Defaults to ``True``.
        dtype: Data type of the weights. Defaults to ``"float32"``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    expr: str
    weight: jax.Array
    bias: Optional[jax.Array]
    use_bias: bool
    kwargs: dict

    def __init__(self, expr: str, bias: bool = True, dtype: Any = "float32", **kwargs: Any):
        self.expr = expr
        self.use_bias = bias
        self.weight = None
        self.bias = None
        self.kwargs = kwargs

    def __call__(self, x, rng=None):
        return einx.nn.linear(
            x,
            self.expr,
            bias=param(self, name="bias", rng=rng) if self.use_bias is not None else None,
            weight=param(self, name="weight", rng=rng),
            **self.kwargs,
        )


class Dropout(eqx.Module):
    """Dropout layer.

    Args:
        expr: Einstein string determining the axes along which dropout is applied. Will be
            passed to ``einx.elementwise``.
        drop_rate: Drop rate.
        inference: Whether the layer is used in inference mode (i.e. not apply dropout). Defaults
            to ``False``.
        **kwargs: Additional parameters that specify values for single axes, e.g. ``a=4``.
    """

    expr: str
    drop_rate: float
    kwargs: dict
    inference: bool

    def __init__(self, expr: str, drop_rate: float, inference: bool = False, **kwargs: Any):
        self.expr = expr
        self.drop_rate = drop_rate
        self.kwargs = kwargs
        self.inference = inference

    def __call__(self, x, rng):
        if not self.inference:
            return einx.nn.dropout(
                x,
                self.expr,
                drop_rate=self.drop_rate,
                rng=rng,
                **self.kwargs,
            )
        else:
            return x

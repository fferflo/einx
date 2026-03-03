import einx._src.tracer as tracer
import einx._src.tracer.signature as signature
import types


class jax:
    def __init__(self):
        jax = tracer.signature.python.import_("jax")
        jnp = tracer.signature.python.import_("jax.numpy", as_="jnp")

        self.numpy = signature.numpy(jnp)
        self.numpy.ndarray.at = signature.classical.at

        self.vmap = signature.classical.vmap(jax.vmap)
        self.nn = types.SimpleNamespace(
            logsumexp=signature.classical.reduce(jax.nn.logsumexp),
            softmax=signature.classical.preserve_shape(jax.nn.softmax),
            log_softmax=signature.classical.preserve_shape(jax.nn.log_softmax),
        )
        self.lax = types.SimpleNamespace(stop_gradient=signature.classical.preserve_shape(jax.lax.stop_gradient))

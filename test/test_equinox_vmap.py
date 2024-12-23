import importlib
from typing import Any

if importlib.util.find_spec("equinox"):
    import equinox as eqx
    import jax.numpy as jnp
    import einx.nn.equinox as einn
    import einx
    import jax

    def test_equinox_vmap():
        class DummyModule(eqx.Module):
            w: jnp.ndarray
            linear: einn.Linear
            key_linear: Any = eqx.field(static = True)

            def __init__( self, size, key ):
                kw, kl = jax.random.split(key, 2)
                self.key_linear = kl
                self.w = jax.random.normal(kw, (size, size))
                self.linear = einn.Linear("[c -> s]", s = size)

            def __call__( self, x ):
                x = self.linear(x, rng = self.key_linear)
                return self.w @ x

        dummy = DummyModule(100, jax.random.PRNGKey(0))

        arr = einx.add("a, b -> a b", jnp.arange(10), 10 * jnp.arange(10))
        dummy(jnp.arange(10))
        ret = einx.vmap("a [b] -> a [s]", arr, s=100, op=dummy)
        assert ret.shape == (10, 100)

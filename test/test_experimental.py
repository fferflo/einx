import importlib
import einx
import numpy as np

if importlib.util.find_spec("jax"):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh

    def assert_sharding(x, mesh=None, partition=None):
        assert {**x.sharding.mesh.shape} == mesh
        assert tuple(x.sharding.spec) == partition

    def test_sharding():
        mesh24 = Mesh(np.asarray(jax.devices("cpu")).reshape(2, 4), axis_names=("d1", "d2"))
        mesh42 = Mesh(np.asarray(jax.devices("cpu")).reshape(4, 2), axis_names=("d1", "d2"))
        mesh4 = Mesh(np.asarray(jax.devices("cpu"))[:4], axis_names=("d1",))

        # Pass mesh=jax.devices("cpu") instead of mesh=None since we cannot set
        # global device to cpu here
        x = jnp.ones((128, 64))
        assert_sharding(
            einx.experimental.shard("([d1] a) b", x, mesh=jax.devices("cpu")), {"d1": 8}, ("d1",)
        )
        assert_sharding(
            einx.experimental.shard("([d1] a) ([d2] b)", x, d2=2, mesh=jax.devices("cpu")),
            {"d1": 4, "d2": 2},
            ("d1", "d2"),
        )
        assert_sharding(
            einx.experimental.shard("([batch] _) ...", x, d2=2, mesh=jax.devices("cpu")),
            {"batch": 8},
            ("batch",),
        )
        assert_sharding(
            einx.experimental.shard("([d1] a) ([d2] b)", x, mesh=mesh24),
            {"d1": 2, "d2": 4},
            ("d1", "d2"),
        )
        assert_sharding(einx.experimental.shard("([d1] a) b", x, mesh=mesh4), {"d1": 4}, ("d1",))
        assert_sharding(
            einx.experimental.shard("b ([d1] a)", x, mesh=mesh4),
            {"d1": 4},
            (
                None,
                "d1",
            ),
        )
        assert_sharding(
            einx.experimental.shard("a ([d1] b)", x, mesh=mesh42),
            {"d1": 4, "d2": 2},
            (
                None,
                "d1",
            ),
        )

        x = jnp.ones((4, 1024, 1024))
        assert_sharding(
            einx.experimental.shard("a ([d2] b) ([d1] c)", x, mesh=mesh42),
            {"d1": 4, "d2": 2},
            (None, "d2", "d1"),
        )

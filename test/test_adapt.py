import pytest
import conftest
import einx

EinxError = einx.errors.EinxError


def test_shape_adapt_with_vmap(setup_adapt):
    setup = setup_adapt
    if not hasattr(setup, "adapt_with_vmap"):
        pytest.skip("adapt_with_vmap is not supported")

    @setup.wrap
    @setup.adapt_with_vmap
    def func(x):
        x = setup.classical.sum(x)
        return x, x

    x = setup.full((10, 20, 3))
    x, y = func("a b [c] -> a b, a b", x)
    assert x.shape == (10, 20)
    assert y.shape == (10, 20)

    with pytest.raises((EinxError, *setup.exceptions)):

        @setup.wrap
        @setup.adapt_with_vmap
        def func(x):
            x = setup.classical.sum(x)
            return x

        x = setup.full((10, 20, 3))
        func("a b [c] -> a b, a b", x)

    with pytest.raises((EinxError, *setup.exceptions)):

        @setup.wrap
        @setup.adapt_with_vmap
        def func(x):
            return x

        x = setup.full((10, 20, 3))
        func("a b [c] -> a b", x)

    @setup.wrap
    @setup.adapt_with_vmap
    def func(x):
        a = setup.classical.sum(x)
        b = setup.classical.sum(x, axis=0)
        return a, b

    x = setup.full((10, 20 * 3))
    x, y = func("a ([b c]) -> a, a [c]", x, b=20)
    assert x.shape == (10,)
    assert y.shape == (10, 3)

    @setup.wrap
    @setup.adapt_with_vmap
    def func(x, y):
        return x + y, x - y

    x = setup.full((10, 20, 3))
    y = setup.full((2 * 3 * 10,))
    x, y = func("a b c, (x c a) -> a b c x, x c b a", x, y)
    assert x.shape == (10, 20, 3, 2)
    assert y.shape == (2, 3, 20, 10)

    @setup.wrap
    @setup.adapt_with_vmap
    def func(x):
        return x * setup.classical.arange(17)

    x = setup.full((10, 20, 3))
    x = func("a b c -> a b c [d]", x, d=17)  # TODO: dynamic dimensions -> dont need d=17
    assert x.shape == (10, 20, 3, 17)

    if not ("torch" in setup.name and "compile" in setup.name and setup.version <= (2, 4, 0)):

        @setup.wrap
        @setup.adapt_with_vmap
        def func(x, *, arg):
            return x * 2

        x = setup.full((10, 20, 3))
        x = func("a b c -> a b c", x, d=17, arg="asd")
        assert x.shape == (10, 20, 3)

    @setup.wrap
    @setup.adapt_with_vmap
    def func(x):
        return setup.classical.transpose(x, (1, 0))

    x = setup.full((2, 10, 20, 3))
    assert func("a [b c] x -> x [c b] a", x).shape == (3, 20, 10, 2)
    with pytest.raises((EinxError, *setup.exceptions)):
        assert func("a [b c] x -> x [b c] a", x).shape == (3, 20, 10, 2)


def test_shape_adapt_numpylike_elementwise(setup_adapt):
    setup = setup_adapt
    if not hasattr(setup, "adapt_numpylike_elementwise"):
        pytest.skip("adapt_numpylike_elementwise is not supported")

    @setup.wrap
    @setup.adapt_numpylike_elementwise
    def func(x, y):
        return setup.classical.add(x, y)

    x = setup.full((10, 20, 3))
    y = setup.full((3, 1, 10, 2))
    x = func("a (b c) d, d 1 c b -> d b c a", x, y, b=2)
    assert x.shape == (3, 2, 10, 10)

    with pytest.raises((EinxError, *setup.exceptions)):

        @setup.wrap
        @setup.adapt_numpylike_elementwise
        def func(x, y):
            return y

        x = setup.full((10, 20, 3))
        y = setup.full((3, 1, 10, 2))
        func("a (b c) d, d 1 c b -> d b c a", x, y, b=2)


def test_shape_adapt_numpylike_reduce(setup_adapt):
    setup = setup_adapt
    if not hasattr(setup, "adapt_numpylike_reduce"):
        pytest.skip("adapt_numpylike_reduce is not supported")

    @setup.wrap
    @setup.adapt_numpylike_reduce
    def func(x, axis):
        return setup.classical.sum(x, axis=axis)

    x = setup.full((10, 20, 3))
    x = func("a (b [c]) d", x, b=2)
    assert x.shape == (10, 2, 3)

    with pytest.raises((EinxError, *setup.exceptions)):

        @setup.wrap
        @setup.adapt_numpylike_reduce
        def func(x, axis):
            return x

        x = setup.full((10, 20, 3))
        func("a (b [c]) d", x, b=2)

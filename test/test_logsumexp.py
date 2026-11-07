import numpy as np
import pytest

from conftest import use_backend
import einx


@pytest.mark.computes_values
@use_backend
def test_logsumexp_nonfinite(setup_backend):
    setup = setup_backend
    x = np.array(
        [
            [-np.inf, -np.inf],
            [np.inf, 0.0],
            [np.inf, -np.inf],
            [-np.inf, 3.0],
            [1000.0, 1000.0],
            [-1000.0, -1000.0],
            [np.nan, 0.0],
        ],
        dtype="float32",
    )
    with np.errstate(invalid="ignore"):
        expected = np.logaddexp.reduce(x, axis=1)

    cases = [
        ("b [c]", x, expected),
        ("[c] b", x.T.copy(), expected),
        ("b ([c])", x, expected[:, None]),
        ("b [c d]", x.reshape(7, 1, 2), expected),
        ("[b c]", x[:3], np.array(np.inf, dtype="float32")),
        ("", np.array(-np.inf, dtype="float32"), np.array(-np.inf, dtype="float32")),
    ]
    for description, values, reference in cases:
        with np.errstate(invalid="raise", divide="ignore"):
            try:
                actual = setup.einx.logsumexp(description, setup.to_tensor(values))
            except einx.errors.OperationNotSupportedError:
                pytest.skip("The selected backend does not implement logsumexp")
        actual = setup.to_numpy(actual)
        assert actual.shape == reference.shape
        assert actual.dtype == reference.dtype
        np.testing.assert_allclose(actual, reference, rtol=1e-5, equal_nan=True)


@pytest.mark.computes_values
@use_backend
def test_log_softmax_multi_axis(setup_backend):
    setup = setup_backend
    backend_name = setup.backend.name if setup.backend is not None else setup.name
    if backend_name.split(".")[0] not in {"numpy", "arrayapi", "torch", "tensorflow", "dask"}:
        pytest.skip("This regression exercises shared log_softmax implementations")
    x = np.linspace(-2, 2, 24, dtype="float32").reshape(2, 3, 4)
    for masked in [False, True]:
        if masked:
            x[0, 0, 0] = -np.inf
        for description, axes in [("[a b c]", (0, 1, 2)), ("[a] b [c]", (0, 2))]:
            expected = x - np.log(np.sum(np.exp(x), axis=axes, keepdims=True))
            try:
                actual = setup.einx.log_softmax(description, setup.to_tensor(x))
            except einx.errors.OperationNotSupportedError:
                pytest.skip("The selected backend does not implement log_softmax")
            actual = setup.to_numpy(actual)
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

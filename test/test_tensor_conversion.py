import numpy as np
import pytest
import functools
from contextlib import suppress
import einx
from conftest import use_backend

OperationNotSupportedError = einx.errors.OperationNotSupportedError


def to_numpy_array(x, dtype=None):
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x


@use_backend
def test_tensor_conversion(setup_backend):
    einx, setup = setup_backend.einx, setup_backend

    if "torch" in setup.name and "compile" in setup.name:
        # Maybe the failure is due to torch.asarray being called inside torch.compile? Not sure
        pytest.skip("Skipping tensor conversion tests for torch.compile")
    if "arrayapi" in setup.name:
        # arrayapi cannot uniquely determine the framework to use
        pytest.skip("Skipping tensor conversion tests for arrayapi backend")

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.id(" -> a b", 1, a=2, b=3)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.id("a -> a b", to_numpy_array([1, 2]), a=2, b=3)
    if "torch.vmap" not in setup.name:
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            einx.id("a, b -> (a + b)", to_numpy_array([1, 2]), to_numpy_array([1, 2]))
    if "mlx.vmap" not in setup.name:
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            einx.id(", b -> (1 + b)", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
        with suppress((OperationNotSupportedError, *setup.exceptions)):
            einx.id("a, -> (a + 1)", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.id(", -> (1 + 1)", 1, 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.id("(1 + 1) -> ,", to_numpy_array([1, 1]))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.id("a a -> a", to_numpy_array([[1, 2], [2, 1]]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.sum("->", 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.sum("[a]", to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.mean("->", 1.0)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.mean("[a]", to_numpy_array([1.0, 2.0]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.var("->", 1.0)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.var("[a]", to_numpy_array([1.0, 2.0]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.std("->", 1.0)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.std("[a]", to_numpy_array([1.0, 2.0]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.prod("->", 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.prod("[a]", to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.count_nonzero("->", 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.count_nonzero("[a]", to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.any("->", True)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.any("[a]", to_numpy_array([True, False]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.all("->", True)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.all("[a]", to_numpy_array([True, False]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.min("->", 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.min("[a]", to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.max("->", 1)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.max("[a]", to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.logsumexp("->", 1.0)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.logsumexp("[a]", to_numpy_array([1.0, 2.0]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.dot("[a], [a] ->", to_numpy_array([1.0, 2.0]), to_numpy_array([1.0, 2.0]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.add(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.add("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.subtract(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.subtract("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.multiply(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.multiply("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.true_divide(", a", 1.0, to_numpy_array([1.0, 2.0], dtype=setup.dtypes.float))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.true_divide("a,", to_numpy_array([1.0, 2.0], dtype=setup.dtypes.float), 1.0)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.floor_divide(", a", 1.0, to_numpy_array([1.0, 2.0], dtype=setup.dtypes.float))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.floor_divide("a,", to_numpy_array([1.0, 2.0], dtype=setup.dtypes.float), 1.0)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.divide(", a", 1.0, to_numpy_array([1.0, 2.0], dtype=setup.dtypes.float))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.divide("a,", to_numpy_array([1.0, 2.0], dtype=setup.dtypes.float), 1.0)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.logical_and(", a", True, to_numpy_array([True, False]))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.logical_and("a,", to_numpy_array([True, False]), True)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.logical_or(", a", True, to_numpy_array([True, False]))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.logical_or("a,", to_numpy_array([True, False]), True)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.where(", ,", True, 1, 2)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.where(", a, a", True, to_numpy_array([1, 2]), to_numpy_array([1, 2]))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.where("a,,", to_numpy_array([True, False]), 1, 2)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.where("a, a, a", to_numpy_array([True, False]), to_numpy_array([1, 2]), to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.maximum(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.maximum("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.minimum(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.minimum("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.less(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.less("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.less_equal(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.less_equal("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.greater(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.greater("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.greater_equal(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.greater_equal("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.equal(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.equal("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.not_equal(", a", 1, to_numpy_array([1, 2], dtype=setup.dtypes.int))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.not_equal("a,", to_numpy_array([1, 2], dtype=setup.dtypes.int), 1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.logaddexp(", a", 1.0, to_numpy_array([1.0, 2.0], dtype=setup.dtypes.float))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.logaddexp("a,", to_numpy_array([1.0, 2.0], dtype=setup.dtypes.float), 1.0)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.get_at("[h], ->", to_numpy_array([1, 2]), 0)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.get_at("[h], a -> a", to_numpy_array([1, 2]), to_numpy_array([1, 0]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.set_at("[h], , -> [h]", to_numpy_array([1, 2], dtype=setup.dtypes.int), 0, 0)
    # with suppress((OperationNotSupportedError, *setup.exceptions)):
    #     einx.set_at("[h], , a -> [h]", to_numpy_array([1, 2]), 0, to_numpy_array([1, 2]))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.set_at("[h], a, -> [h]", to_numpy_array([1, 2], dtype=setup.dtypes.int), to_numpy_array([1, 0]), 0)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.set_at("[h], a, a -> [h]", to_numpy_array([1, 2]), to_numpy_array([1, 0]), to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.add_at("[h], , -> [h]", to_numpy_array([1, 2], dtype=setup.dtypes.int), 0, 0)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.add_at("[h], , a -> [h]", to_numpy_array([1, 2]), 0, to_numpy_array([1, 2]))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.add_at("[h], a, -> [h]", to_numpy_array([1, 2], dtype=setup.dtypes.int), to_numpy_array([1, 0]), 0)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.add_at("[h], a, a -> [h]", to_numpy_array([1, 2]), to_numpy_array([1, 0]), to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.subtract_at("[h], , -> [h]", to_numpy_array([1, 2], dtype=setup.dtypes.int), 0, 0)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.subtract_at("[h], , a -> [h]", to_numpy_array([1, 2]), 0, to_numpy_array([1, 2]))
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.subtract_at("[h], a, -> [h]", to_numpy_array([1, 2], dtype=setup.dtypes.int), to_numpy_array([1, 0]), 0)
    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.subtract_at("[h], a, a -> [h]", to_numpy_array([1, 2]), to_numpy_array([1, 0]), to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.softmax("[a]", to_numpy_array([1.0, 2.0]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.log_softmax("[a]", to_numpy_array([1.0, 2.0]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.sort("[a]", to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.argsort("[a]", to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.flip("[a]", to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.roll("[a]", to_numpy_array([1, 2]), shift=1)

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.argmax("[a]", to_numpy_array([1, 2]))

    with suppress((OperationNotSupportedError, *setup.exceptions)):
        einx.argmin("[a]", to_numpy_array([1, 2]))

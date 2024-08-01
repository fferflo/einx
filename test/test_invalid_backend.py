import pytest
import sys


@pytest.mark.all
@pytest.mark.skipif("einx" in sys.modules, reason="einx is already imported")
def test_import():
    # Create an invalid jax module
    import sys

    class jax:
        pass

    sys.modules["jax"] = jax

    # Check that einx avoids raising an error here
    import einx
    import numpy as np

    x = np.zeros((10,))
    einx.add("a, a", x, x, backend="numpy")

    # The error should only be raised when the backend is actually used
    with pytest.raises(einx.backend.InvalidBackendException):
        einx.add("a, a", x, x, backend="jax")

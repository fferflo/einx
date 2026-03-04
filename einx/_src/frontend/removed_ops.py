import warnings
from typing import Union
import numpy.typing as npt
from .ops import id
from .backend import Backend
from .types import Tensor


class RemovedOperationError(Exception):
    pass


def vmap(*args, **kwargs):
    raise RemovedOperationError(
        "einx.vmap has been removed since version 0.4.0. Please use 'einx.{framework}.adapt_with_vmap' instead. See https://einx.readthedocs.io/en/latest/api/adapters.html"
    )


def vmap_with_axis(*args, **kwargs):
    raise RemovedOperationError(
        "einx.vmap_with_axis has been removed since version 0.4.0. Please use one of the new adapters instead. See https://einx.readthedocs.io/en/latest/api/adapters.html"
    )


def reduce(*args, **kwargs):
    raise RemovedOperationError(
        "einx.reduce has been removed since version 0.4.0. Please use 'einx.{framework}.adapt_numpylike_reduce' instead. See https://einx.readthedocs.io/en/latest/api/adapters.html"
    )


def elementwise(*args, **kwargs):
    raise RemovedOperationError(
        "einx.elementwise has been removed since version 0.4.0. Please use 'einx.{framework}.adapt_numpylike_elementwise' instead. See https://einx.readthedocs.io/en/latest/api/adapters.html"
    )


def arange(*args, **kwargs):
    raise RemovedOperationError("einx.arange has been removed since version 0.4.0. Please use einx.id with np.arange instead.")


def rearrange(description: str, *tensors: Tensor, backend: Backend | str | None = None, **parameters: npt.ArrayLike) -> Tensor | tuple[Tensor, ...]:
    warnings.warn("einx.rearrange is deprecated and will be removed in a future release. Please use einx.id instead.", DeprecationWarning, stacklevel=2)
    return id(description, *tensors, backend=backend, **parameters)

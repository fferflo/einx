from .register import register_for_module, register, get, backends, numpy
from .base import Backend, get_default

from . import _numpy as numpy
from . import _torch as torch
from . import _tensorflow as tensorflow
from . import _jax as jax
from . import _dask as dask
from . import _mlx as mlx
from . import _tinygrad as tinygrad

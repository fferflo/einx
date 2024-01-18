from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import typing as t

	from einx.type_util import Backend, TArray

	def sum(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="sum"``
		"""
		...

	def mean(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="mean"``
		"""
		...

	def var(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="var"``
		"""
		...

	def std(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="std"``
		"""
		...

	def prod(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="prod"``
		"""
		...

	def count_nonzero(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="count_nonzero"``
		"""
		...

	def any(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="any"``
		"""
		...

	def all(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="all"``
		"""
		...

	def max(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="max"``
		"""
		...

	def min(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="min"``
		"""
		...

	def logsumexp(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.reduce` with ``op="logsumexp"``
		"""
		...


from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import typing as t

	from einx.type_util import Backend, TArray

	def get_at(description: str, tensor: TArray, coordinates: TArray, update: t.Optional[TArray] = None, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.index` with ``op="get_at"``
		"""
		...

	def set_at(description: str, tensor: TArray, coordinates: TArray, update: t.Optional[TArray] = None, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.index` with ``op="set_at"``
		"""
		...

	def add_at(description: str, tensor: TArray, coordinates: TArray, update: t.Optional[TArray] = None, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.index` with ``op="add_at"``
		"""
		...

	def subtract_at(description: str, tensor: TArray, coordinates: TArray, update: t.Optional[TArray] = None, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.index` with ``op="subtract_at"``
		"""
		...


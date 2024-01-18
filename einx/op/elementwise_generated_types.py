from typing import TYPE_CHECKING

if TYPE_CHECKING:
	import typing as t

	from einx.type_util import Backend, TArray

	def add(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="add"``
		"""
		...

	def subtract(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="subtract"``
		"""
		...

	def multiply(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="multiply"``
		"""
		...

	def true_divide(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="true_divide"``
		"""
		...

	def floor_divide(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="floor_divide"``
		"""
		...

	def divide(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="divide"``
		"""
		...

	def logical_and(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="logical_and"``
		"""
		...

	def logical_or(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="logical_or"``
		"""
		...

	def where(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="where"``
		"""
		...

	def less(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="less"``
		"""
		...

	def less_equal(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="less_equal"``
		"""
		...

	def greater(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="greater"``
		"""
		...

	def greater_equal(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="greater_equal"``
		"""
		...

	def equal(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="equal"``
		"""
		...

	def not_equal(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="not_equal"``
		"""
		...

	def maximum(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="maximum"``
		"""
		...

	def minimum(description: str, *tensors: TArray, backend: t.Optional[Backend] = None, cse: bool = True, graph: bool = False, **parameters: TArray) -> TArray:
		"""
		Alias for :func:`einx.elementwise` with ``op="minimum"``
		"""
		...


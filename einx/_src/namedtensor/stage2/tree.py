import numpy as np


class Expression:
    def __init__(self, ellipsis_indices, begin_pos=None, end_pos=None):
        self.ellipsis_indices = ellipsis_indices
        self.parent = None
        self.begin_pos = begin_pos
        self.end_pos = end_pos


class FlattenedAxis(Expression):
    @staticmethod
    def create(inner, ellipsis_indices, begin_pos=None, end_pos=None):
        if isinstance(inner, FlattenedAxis):
            return inner
        else:
            return FlattenedAxis(inner, ellipsis_indices, begin_pos, end_pos)

    def __init__(self, inner, ellipsis_indices, begin_pos=None, end_pos=None):
        Expression.__init__(self, ellipsis_indices, begin_pos=begin_pos, end_pos=end_pos)
        self.inner = inner
        inner.parent = self
        assert not isinstance(inner, FlattenedAxis)

    def __str__(self):
        return f"({self.inner})"

    @property
    def ndim(self):
        return 1

    @property
    def value(self):
        return self.inner.value

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return FlattenedAxis(self.inner.__deepcopy__(), ellipsis_indices=self.ellipsis_indices, begin_pos=self.begin_pos, end_pos=self.end_pos)

    def nodes(self):
        yield self
        yield from self.inner.nodes()


class List(Expression):
    @staticmethod
    def create(children, ellipsis_indices, begin_pos=-1, end_pos=-1):
        children2 = []

        def _add(child):
            if isinstance(child, List):
                for c in child.children:
                    _add(c)
            else:
                children2.append(child)

        for c in children:
            _add(c)
        children = children2

        if len(children) == 1:
            return children[0]
        else:
            return List(children, ellipsis_indices, begin_pos, end_pos)

    def __init__(self, children, ellipsis_indices, begin_pos=None, end_pos=None):
        Expression.__init__(self, ellipsis_indices, begin_pos=begin_pos, end_pos=end_pos)
        self.children = children
        assert len(self.children) != 1
        for c in children:
            c.parent = self
            assert not isinstance(c, List)

    def __str__(self):
        return " ".join([str(c) for c in self.children])

    @property
    def ndim(self):
        return sum(c.ndim for c in self.children)

    @property
    def value(self):
        values = [c.value for c in self.children]
        if any(v is None for v in values):
            return None
        else:
            return np.prod(values)

    def __iter__(self):
        for c in self.children:
            yield from c

    def __deepcopy__(self):
        return List([c.__deepcopy__() for c in self.children], ellipsis_indices=self.ellipsis_indices, begin_pos=self.begin_pos, end_pos=self.end_pos)

    def nodes(self):
        yield self
        for c in self.children:
            yield from c.nodes()


class Axis(Expression):
    def __init__(self, name, value, ellipsis_indices, begin_pos=None, end_pos=None):
        Expression.__init__(self, ellipsis_indices, begin_pos=begin_pos, end_pos=end_pos)
        if not isinstance(name, str):
            raise TypeError(f"Axis name must be a string, but got {type(name)}")
        if value is not None and not isinstance(value, int | np.integer):
            raise TypeError(f"Axis value must be an int or None, but got {type(value)}")
        self.name = name
        self.value = int(value) if value is not None else None

    def __str__(self):
        return self.name if self.value is None else str(self.value)

    @property
    def ndim(self):
        return 1

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return Axis(self.name, self.value, ellipsis_indices=self.ellipsis_indices, begin_pos=self.begin_pos, end_pos=self.end_pos)

    def nodes(self):
        yield self


class ConcatenatedAxis(Expression):
    @staticmethod
    def create(children, ellipsis_indices, begin_pos=None, end_pos=None):
        if len(children) == 1:
            return children[0]
        elif len(children) == 0:
            raise ValueError("ConcatenatedAxis must have at least one child")
        else:
            return ConcatenatedAxis(children, ellipsis_indices, begin_pos, end_pos)

    def __init__(self, children, ellipsis_indices, begin_pos=None, end_pos=None):
        Expression.__init__(self, ellipsis_indices, begin_pos=begin_pos, end_pos=end_pos)
        for c in children:
            if c.ndim != 1:
                raise ValueError(f"ConcatenatedAxis can only be used on expressions of length 1, but got expression '{c}'")
        self.children = children
        for c in children:
            c.parent = self

    def __str__(self):
        return "(" + " + ".join([str(c) for c in self.children]) + ")"

    @property
    def ndim(self):
        return 1

    @property
    def value(self):
        values = [c.value for c in self.children]
        if any(v is None for v in values):
            return None
        else:
            return np.sum(values)

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return ConcatenatedAxis(
            [c.__deepcopy__() for c in self.children], ellipsis_indices=self.ellipsis_indices, begin_pos=self.begin_pos, end_pos=self.end_pos
        )

    def nodes(self):
        yield self
        for c in self.children:
            yield from c.nodes()


class Brackets(Expression):
    @staticmethod
    def create(inner, ellipsis_indices, begin_pos=None, end_pos=None):
        if isinstance(inner, Brackets):
            return inner
        elif inner.ndim == 0:
            return List([], ellipsis_indices)
        else:
            return Brackets(inner, ellipsis_indices, begin_pos, end_pos)

    def __init__(self, inner, ellipsis_indices, begin_pos=None, end_pos=None):
        Expression.__init__(self, ellipsis_indices, begin_pos=begin_pos, end_pos=end_pos)
        self.inner = inner
        inner.parent = self

    def __str__(self):
        return f"[{self.inner}]"

    @property
    def ndim(self):
        return self.inner.ndim

    @property
    def value(self):
        return self.inner.value

    def __iter__(self):
        yield from self.inner

    def __deepcopy__(self):
        return Brackets(self.inner.__deepcopy__(), ellipsis_indices=self.ellipsis_indices, begin_pos=self.begin_pos, end_pos=self.end_pos)

    def nodes(self):
        yield self
        yield from self.inner.nodes()

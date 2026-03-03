import numpy as np
import uuid


class Expression:
    def __init__(self, value, begin_pos=None, end_pos=None):
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"Expected int, got {type(value)}")
        self.value = int(value)
        self.parent = None
        self.begin_pos = begin_pos
        self.end_pos = end_pos

    @property
    def shape(self):
        return tuple(x.value for x in self)

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return self.ndim


class FlattenedAxis(Expression):
    @staticmethod
    def create(inner, begin_pos=None, end_pos=None):
        if isinstance(inner, FlattenedAxis):
            return inner
        else:
            return FlattenedAxis(inner, begin_pos=begin_pos, end_pos=end_pos)

    def __init__(self, inner, begin_pos=None, end_pos=None):
        Expression.__init__(self, inner.value, begin_pos=begin_pos, end_pos=end_pos)
        self.inner = inner
        inner.parent = self
        assert not isinstance(inner, FlattenedAxis)

    def __str__(self):
        return f"({self.inner})"

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return FlattenedAxis(self.inner.__deepcopy__(), self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, FlattenedAxis) and self.inner == other.inner

    def __hash__(self):
        return 8716123 + hash(self.inner)

    def nodes(self):
        yield self
        yield from self.inner.nodes()


class List(Expression):
    @staticmethod
    def create(children, begin_pos=None, end_pos=None):
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
            return List(children, begin_pos, end_pos)

    def __init__(self, children, begin_pos=None, end_pos=None):
        Expression.__init__(self, np.prod([c.value for c in children]).astype("int32"), begin_pos=begin_pos, end_pos=end_pos)
        self.children = children
        assert len(children) != 1
        for c in children:
            c.parent = self
            assert not isinstance(c, List)

    def __str__(self):
        return " ".join([str(c) for c in self.children])

    def __getitem__(self, i):
        return self.children[i]

    def __iter__(self):
        for c in self.children:
            yield from c

    def __deepcopy__(self):
        return List([c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, List) and self.children == other.children

    def __hash__(self):
        return 6563 + hash(tuple(self.children))

    def nodes(self):
        yield self
        for c in self.children:
            yield from c.nodes()


class Axis(Expression):
    @staticmethod
    def new_unnamed(value):
        name = f"unnamed.{uuid.uuid4().int}"
        return Axis(name, value)

    def __init__(self, name, value, begin_pos=None, end_pos=None):
        Expression.__init__(self, value, begin_pos=begin_pos, end_pos=end_pos)
        if not isinstance(name, str):
            raise TypeError(f"Expected str, got {type(name)}")
        if value is None:
            raise ValueError("Axis value cannot be None")
        self.name = name
        self._is_unnamed = self.name.startswith("unnamed.")

    def __str__(self):
        return self.name if not self._is_unnamed else str(self.value)

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return Axis(self.name, self.value, self.begin_pos, self.end_pos)

    def __eq__(self, other):
        if not isinstance(other, Axis):
            return False
        if self.value != other.value:
            return False
        return self.name == other.name

    def __hash__(self):
        return 9817234 + hash(self.name) + 3 * hash(self.value)

    def nodes(self):
        yield self


class ConcatenatedAxis(Expression):
    @staticmethod
    def create(children, begin_pos=None, end_pos=None):
        if len(children) == 1:
            return children[0]
        elif len(children) == 0:
            raise ValueError("ConcatenatedAxis must have at least one child")
        else:
            return ConcatenatedAxis(children, begin_pos, end_pos)

    def __init__(self, children, begin_pos=None, end_pos=None):
        if len(children) == 0:
            raise ValueError("ConcatenatedAxis must have at least one child")
        Expression.__init__(self, np.sum([c.value for c in children]).astype("int32"), begin_pos=begin_pos, end_pos=end_pos)
        self.children = children
        for c in children:
            if len(c) != 1:
                raise ValueError(f"ConcatenatedAxis can only be used on expressions of length 1, butgot expression '{c}'")
            c.parent = self

    def __str__(self):
        return "(" + " + ".join([str(c) for c in self.children]) + ")"

    def __iter__(self):
        yield self

    def __deepcopy__(self):
        return ConcatenatedAxis([c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, ConcatenatedAxis) and self.children == other.children

    def __hash__(self):
        return 123 + hash(tuple(self.children))

    def nodes(self):
        yield self
        for c in self.children:
            yield from c.nodes()


class Brackets(Expression):
    @staticmethod
    def create(inner, begin_pos=None, end_pos=None):
        if isinstance(inner, Brackets):
            return inner
        elif inner.ndim == 0:
            return List([])
        else:
            return Brackets(inner, begin_pos, end_pos)

    def __init__(self, inner, begin_pos=None, end_pos=None):
        Expression.__init__(self, inner.value, begin_pos=begin_pos, end_pos=end_pos)
        self.inner = inner
        inner.parent = self

    def __str__(self):
        return f"[{self.inner}]"

    def __iter__(self):
        yield from self.inner

    def __deepcopy__(self):
        return Brackets(self.inner.__deepcopy__(), self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Brackets) and self.inner == other.inner

    def __hash__(self):
        return 6433236 + hash(self.inner)

    def nodes(self):
        yield self
        yield from self.inner.nodes()

class Expression:
    def __init__(self, begin_pos, end_pos):
        self.begin_pos = begin_pos
        self.end_pos = end_pos
        self.parent = None

    @property
    def depth(self):
        if self.parent is None:
            return 0
        elif isinstance(self.parent, Ellipsis):
            return 1 + self.parent.depth
        else:
            return self.parent.depth


class FlattenedAxis(Expression):
    @staticmethod
    def create(inner, begin_pos=-1, end_pos=-1):
        if isinstance(inner, FlattenedAxis):
            return inner
        else:
            return FlattenedAxis(inner, begin_pos, end_pos)

    def __init__(self, inner, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.inner = inner
        inner.parent = self
        assert not isinstance(inner, FlattenedAxis)

    def nodes(self):
        yield self
        yield from self.inner.nodes()

    @property
    def children(self):
        return [self.inner]

    def __str__(self):
        return "(" + str(self.inner) + ")"

    def __deepcopy__(self):
        return FlattenedAxis(self.inner.__deepcopy__(), self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, FlattenedAxis) and self.inner == other.inner

    def __hash__(self):
        return 87123 + hash(self.inner)

    @property
    def ndim(self):
        return 1

    @property
    def value(self):
        return self.inner.value


class Brackets(Expression):
    @staticmethod
    def create(inner, begin_pos=-1, end_pos=-1):
        if isinstance(inner, Brackets):
            return inner
        elif inner.ndim == 0:
            return List([])
        else:
            return Brackets(inner, begin_pos, end_pos)

    def __init__(self, inner, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.inner = inner
        self.inner.parent = self
        assert inner.ndim != 0
        assert not isinstance(inner, Brackets)

    def nodes(self):
        yield self
        yield from self.inner.nodes()

    @property
    def children(self):
        return [self.inner]

    def __str__(self):
        return "[" + str(self.inner) + "]"

    def __deepcopy__(self):
        return Brackets(self.inner.__deepcopy__(), self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Brackets) and self.inner == other.inner

    def __hash__(self):
        return 91236 + hash(self.inner)

    @property
    def ndim(self):
        return self.inner.ndim

    @property
    def value(self):
        return self.inner.value


class Axis(Expression):
    @staticmethod
    def create(name, value=None, begin_pos=-1, end_pos=-1):
        return Axis(name, value, begin_pos, end_pos)

    def __init__(self, name, value, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        if not isinstance(name, str):
            raise TypeError("Axis name must be a string")
        if value is not None and not isinstance(value, int):
            raise TypeError("Axis value must be an integer or None")
        self.name = name
        self.value = value

    def nodes(self):
        yield self

    @property
    def children(self):
        return []

    def __str__(self):
        return self.name if self.value is None else str(self.value)

    def __deepcopy__(self):
        return Axis(self.name, self.value, self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Axis) and self.name == other.name and self.value == other.value

    def __hash__(self):
        return 12345 + hash(self.name) + 2 * hash(self.value)

    @property
    def ndim(self):
        return 1


class Ellipsis(Expression):
    anonymous_variable_name = ".anonymous_ellipsis_axis"

    @staticmethod
    def create(inner, begin_pos=-1, end_pos=-1, ellipsis_id=None):
        if inner.ndim == 0:
            return List([])
        else:
            return Ellipsis(inner, begin_pos, end_pos, ellipsis_id)

    def __init__(self, inner, begin_pos=-1, end_pos=-1, ellipsis_id=None):
        Expression.__init__(self, begin_pos, end_pos)
        if ellipsis_id is None:
            raise ValueError("ellipsis_id must be provided")
        self.inner = inner
        self.inner.parent = self
        self.ellipsis_id = ellipsis_id
        assert inner.ndim != 0

    def nodes(self):
        yield self
        yield from self.inner.nodes()

    @property
    def children(self):
        return [self.inner]

    def __str__(self):
        if isinstance(self.inner, Axis) and self.inner.name == Ellipsis.anonymous_variable_name:
            return "..."
        n = str(self.inner)
        if isinstance(self.inner, List) and len(self.inner.children) != 1:
            n = "{" + n + "}"
        return f"{n}..."

    def __deepcopy__(self):
        return Ellipsis(self.inner.__deepcopy__(), self.begin_pos, self.end_pos, self.ellipsis_id)

    def __eq__(self, other):
        return isinstance(other, Ellipsis) and self.inner == other.inner

    def __hash__(self):
        return 34567 + hash(self.inner)

    @property
    def ndim(self):
        if self.inner.ndim == 0:
            return 0
        else:
            return None

    @property
    def value(self):
        if self.inner.value == 1:
            return 1
        elif self.inner.value == 0:
            return 0
        else:
            return None


class ConcatenatedAxis(Expression):
    @staticmethod
    def create(children, begin_pos=-1, end_pos=-1):
        if len(children) == 1:
            return children[0]
        elif len(children) == 0:
            raise ValueError("ConcatenatedAxis must have at least one child")
        else:
            return ConcatenatedAxis(children, begin_pos, end_pos)

    def __init__(self, children, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.children = children
        assert len(children) > 1
        for child in self.children:
            child.parent = self
            assert child.ndim == 1

    def nodes(self):
        yield self
        for child in self.children:
            yield from child.nodes()

    def __str__(self):
        return "(" + " + ".join([str(c) for c in self.children]) + ")"

    def __deepcopy__(self):
        return ConcatenatedAxis([c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, ConcatenatedAxis) and self.children == other.children

    def __hash__(self):
        return 234 + hash(tuple(self.children))

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


class List(Expression):
    @staticmethod
    def create(children, begin_pos=-1, end_pos=-1):
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

    def __init__(self, children, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.children = children
        assert len(children) != 1
        for child in self.children:
            child.parent = self
            assert not isinstance(child, List)

    def nodes(self):
        yield self
        for child in self.children:
            yield from child.nodes()

    def __str__(self):
        return " ".join([str(c) for c in self.children])

    def __deepcopy__(self):
        return List([c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, List) and self.children == other.children

    def __hash__(self):
        return 2333 + hash(tuple(self.children))

    @property
    def ndim(self):
        child_ndims = [c.ndim for c in self.children]
        if any(e is None for e in child_ndims):
            return None
        else:
            return sum(child_ndims)

    @property
    def value(self):
        values = [c.value for c in self.children]
        if any(v is None for v in values):
            return None
        else:
            return np.prod(values)


class Args(Expression):
    def create(children, begin_pos=-1, end_pos=-1):
        return Args(children, begin_pos, end_pos)

    def __init__(self, children, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.children = children
        for child in self.children:
            assert not isinstance(child, Args)
            child.parent = self

    @property
    def ndim(self):
        return None

    @property
    def value(self):
        return None

    def nodes(self):
        yield self
        for child in self.children:
            yield from child.nodes()

    def __str__(self):
        return ", ".join([str(c) for c in self.children])

    def __deepcopy__(self):
        return Args([c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Args) and self.children == other.children

    def __hash__(self):
        return 233314 + hash(tuple(self.children))


class Op(Expression):
    @staticmethod
    def create(children, begin_pos=-1, end_pos=-1):
        return Op(children, begin_pos, end_pos)

    def __init__(self, children, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        assert len(children) >= 1
        self.children = children
        for child in self.children:
            child.parent = self

    @property
    def ndim(self):
        return None

    @property
    def value(self):
        return None

    def nodes(self):
        yield self
        for child in self.children:
            yield from child.nodes()

    def __str__(self):
        return " -> ".join([str(c) for c in self.children])

    def __deepcopy__(self):
        return Op([c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Op) and self.children == other.children

    def __hash__(self):
        return 961121 + hash(tuple(self.children))

import einx
from collections import defaultdict

class Node:
    def __init__(self, ellipses, name=None):
        self.parent = None
        self.ellipses = ellipses
        self.name = name if not name is None else f"__{self.__class__.__name__}__{id(self)}"

    def __repr__(self):
        return str(self)

    @property
    def variables(self):
        return [x for x in self.traverse() if isinstance(x, Variable)]

    @property
    def depth(self):
        return len(self.ellipses)

class Variable(Node):
    def __init__(self, name, ellipses):
        if name.isnumeric():
            self.value = int(name)
            name = f"__constantdim{id(self)}({self.value})"
        else:
            if "__constantdim" in name or "." in name:
                raise ValueError(f"Parameter names cannot contain '__constantdim' or '.', found '{name}'")
            self.value = None
        Node.__init__(self, ellipses, name=name)

    def __str__(self):
        return self.name if self.value is None else str(self.value)

    def traverse(self):
        yield self

    def copy(self):
        variable = Variable.__new__(Variable)
        variable.value = self.value
        variable.name = self.name
        variable.ellipses = self.ellipses
        return variable

    def __eq__(self, other):
        return other.__class__ == Variable and str(self) == str(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

class Ellipsis(Node):
    def __init__(self, inner, ellipses, expansion_id):
        Node.__init__(self, ellipses)
        self.inner = inner
        self.inner.parent = self
        self.expansion_id = id(self) if expansion_id is None else expansion_id

    def __str__(self):
        return str(self.inner) + "..."

    def traverse(self):
        yield self
        for x in self.inner.traverse():
            yield x

    def copy(self):
        return Ellipsis(self.inner, self.ellipses, self.expansion_id)

    def __eq__(self, other):
        return other.__class__ == Ellipsis and self.inner == other.inner

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

class Group(Node):
    def __init__(self, children, ellipses, front, back):
        Node.__init__(self, ellipses)
        self.children = children
        for child in self.children:
            child.parent = self
        self.front = front
        self.back = back

    def __str__(self):
        return self.front + " ".join([str(c) for c in self.children]) + self.back

    def traverse(self):
        yield self
        for x in self.children:
            for y in x.traverse():
                yield y

    def copy(self):
        return Group([c.copy() for c in self.children], self.ellipses, self.front, self.back)

    def __eq__(self, other):
        return other.__class__ == Group and self.front == other.front and self.back == other.back and \
            len(self.children) == len(other.children) and all(c1 == c2 for c1, c2 in zip(self.children, other.children))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

class Choice(Node):
    def __init__(self, choices, ellipses, separator):
        Node.__init__(self, ellipses)
        self.choices = choices
        for child in self.choices:
            if isinstance(child, list):
                for child2 in child:
                    child2.parent = self
            else:
                child.parent = self
        self.separator = separator

    def __str__(self):
        return f"{self.separator}".join([to_string(c) for c in self.choices])

    def traverse(self):
        yield self
        for x in self.choices:
            if isinstance(x, list):
                for y in x:
                    for z in y.traverse():
                        yield z
            else:
                for y in x.traverse():
                    yield y

    def copy(self):
        return Choice(copy(self.choices), self.ellipses, self.separator)

    def __eq__(self, other):
        return other.__class__ == Choice and self.separator == other.separator or \
            len(self.choices) == len(other.choices) and all(c1 == c2 for c1, c2 in zip(self.choices, other.choices))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

class Root(Group):
    def __init__(self, children):
        Group.__init__(self, children, ellipses=[], front="", back="")
        self.parent = None

    def copy(self):
        return Root([c.copy() for c in self.children])

    def __eq__(self, other):
        return other.__class__ == Root and self.front == other.front and self.back == other.back and \
            len(self.children) == len(other.children) and all(c1 == c2 for c1, c2 in zip(self.children, other.children))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(self))

def parse(expression):
    if isinstance(expression, Root):
        return expression
    root = Root(_parse(expression, ellipses=[]))

    # Check that ellipsis depths match
    variable_rank = defaultdict(set)
    for variable in root.traverse():
        if isinstance(variable, Variable):
            variable_rank[variable.name].add(len(variable.ellipses))
    for k in variable_rank:
        if len(variable_rank[k]) > 1:
            raise ValueError(f"Got conflicting ellipsis depths for parameter {k}")

    return root

_parentheses = [("(", ")"), ("[", "]")]
_disallowed_strings = [",", "->"]
_choice_separators = ["|"]
def _parse(expression, ellipses):
    if any(s in expression for s in _disallowed_strings):
        raise ValueError(f"Expression cannot contain the following substrings: {_disallowed_strings}")

    # Split string
    elements = [""]
    stack = []
    for c in expression:
        if c in _choice_separators and len(stack) == 0:
            elements = elements + [c, ""]
            continue
        for front, back in _parentheses:
            if c == front:
                stack.append(front)
                elements[-1] += c
                break
        else:
            for front, back in _parentheses:
                if c == back:
                    if len(stack) == 0 or stack[-1] != front:
                        raise ValueError("Invalid parentheses")
                    stack.pop()
                    elements[-1] += c
                    break
            else:
                if len(stack) == 0 and c == " ":
                    if len(elements[-1]) > 0:
                        elements = elements + [""]
                else:
                    elements[-1] += c

    if len(stack) > 0:
        raise ValueError("Invalid parentheses")
    expression = elements
    expression = [e.strip() for e in expression]
    expression = [e for e in expression if len(e) > 0]

    # Parse lists to expression objects
    all_separators = set(child for child in expression if child in _choice_separators)
    if len(all_separators) > 1:
        raise ValueError("Cannot use different choice separators on the same level of an expression")
    elif len(all_separators) == 1:
        children = []
        current_child = []
        for child in expression:
            if child in _choice_separators:
                children.append(current_child)
                current_child = []
            else:
                current_child.append(child)
        children.append(current_child)
        children = [" ".join(child).strip() for child in children]

        separator = list(all_separators)[0]
        choice = Choice([_parse(child, ellipses=ellipses) for child in children], ellipses=ellipses, separator=separator)
        children = [choice]
    elif len(expression) == 1:
        expression = expression[0]
        if expression in _choice_separators:
            children = [_Separator(expression)]
        elif expression.endswith("..."):
            ellipsis = Ellipsis.__new__(Ellipsis)
            inner = _parse(expression[:-3], ellipses=ellipses + [ellipsis])
            if len(inner) == 0:
                if not einx.anonymous_ellipsis_name is None:
                    inner = [Variable(name=einx.anonymous_ellipsis_name, ellipses=ellipses + [ellipsis])]
                else:
                    raise ValueError("Anonymous ellipsis is not allowed")
            ellipsis.__init__(inner[0], ellipses=ellipses, expansion_id=None)
            children = [ellipsis]
        else:
            for front, back in _parentheses:
                if expression[0] == front:
                    assert expression[-1] == back
                    group = Group(_parse(expression[1:-1], ellipses=ellipses), ellipses=ellipses, front=front, back=back)
                    children = [group]
                    break
            else:
                for front, back in _parentheses:
                    assert not front in expression and not back in expression
                children = [Variable(name=expression, ellipses=ellipses)]
    else:
        children = [x for d in expression for x in _parse(d, ellipses=ellipses)]

    return children

def to_string(node):
    if isinstance(node, list):
        return " ".join([to_string(child) for child in node])
    else:
        return str(node)

def copy(node):
    if isinstance(node, list):
        return [copy(child) for child in node]
    else:
        return node.copy()

def is_parent_of(parent, child):
    if isinstance(child, Group) and child.front == "" and child.back == "":
        return is_parent_of(parent, child.children)

    if (isinstance(child, list) and len(child) == 0):
        return True
    elif parent == child:
        return True
    elif isinstance(parent, Group):
        return is_parent_of(parent.children, child)
    elif isinstance(parent, Choice):
        return is_parent_of(parent.choices, child)
    elif isinstance(parent, list):
        # Sublists of children
        for i in range(len(parent) - len(child) + 1):
            if (i > 0 or i + len(child) < len(parent)) and is_parent_of(parent[i:i + len(child)], child):
                return True
    return False

def remove(expr, pred):
    def traverse(expr, ellipses):
        if isinstance(expr, list):
            result = []
            for expr in expr:
                result.extend(traverse(expr, ellipses=ellipses))
            return result

        if pred(expr):
            return []
        if isinstance(expr, Variable):
            return [expr.copy()]
        elif isinstance(expr, Ellipsis):
            ellipsis = Ellipsis.__new__(Ellipsis)
            inner = traverse(expr.inner, ellipses=ellipses + [ellipsis])
            if len(inner) == 0:
                return []
            assert len(inner) == 1
            ellipsis.__init__(inner[0], ellipses=ellipses, expansion_id=expr.expansion_id)
            return [ellipsis]
        elif isinstance(expr, Group):
            return [Group(traverse(expr.children, ellipses=ellipses), ellipses, expr.front, expr.back)]
        elif isinstance(expr, Choice):
            return [Choice([traverse(choice, ellipses=ellipses) for choice in expr.choices], ellipses, expr.separator)]
        else:
            assert False

    return Root(traverse(expr.children, ellipses=[]))

def prune_group(expr, pred):
    def traverse(expr, ellipses):
        if isinstance(expr, list):
            result = []
            for expr in expr:
                result.extend(traverse(expr, ellipses=ellipses))
            return result

        if isinstance(expr, Variable):
            return [expr.copy()]
        elif isinstance(expr, Ellipsis):
            ellipsis = Ellipsis.__new__(Ellipsis)
            inner = traverse(expr.inner, ellipses=ellipses + [ellipsis])
            if len(inner) == 0:
                if on_empty_ellipsis == "one":
                    inner = [Variable(name="1", ellipses=ellipses + [ellipsis])]
                else:
                    return []
            assert len(inner) == 1
            ellipsis.__init__(inner[0], ellipses=ellipses, expansion_id=expr.expansion_id)
            return [ellipsis]
        elif isinstance(expr, Group):
            children = traverse(expr.children, ellipses=ellipses)
            if pred(expr):
                return children
            else:
                return [Group(children, ellipses, expr.front, expr.back)]
        elif isinstance(expr, Choice):
            return [Choice([traverse(choice, ellipses=ellipses) for choice in expr.choices], ellipses, expr.separator)]
        else:
            assert False

    return Root(traverse(expr.children, ellipses=[]))

def make_choice(expr, pred, index, num_choices):
    def traverse(expr, ellipses):
        if isinstance(expr, list):
            result = []
            for expr in expr:
                result.extend(traverse(expr, ellipses=ellipses))
            return result

        if isinstance(expr, Variable):
            return [expr.copy()]
        elif isinstance(expr, Ellipsis):
            ellipsis = Ellipsis.__new__(Ellipsis)
            inner = traverse(expr.inner, ellipses=ellipses + [ellipsis])
            if len(inner) == 0:
                if on_empty_ellipsis == "one":
                    inner = [Variable(name="1", ellipses=ellipses + [ellipsis])]
                else:
                    return []
            assert len(inner) == 1
            ellipsis.__init__(inner[0], ellipses=ellipses, expansion_id=expr.expansion_id)
            return [ellipsis]
        elif isinstance(expr, Group):
            return [Group(traverse(expr.children, ellipses=ellipses), ellipses, expr.front, expr.back)]
        elif isinstance(expr, Choice):
            if pred(expr):
                if len(expr.choices) != num_choices:
                    raise ValueError(f"Found expression with {len(expr.choices)} choices, but expected {num_choices}")
                return traverse(expr.choices[index], ellipses=ellipses)
            else:
                return [Choice([traverse(choice, ellipses=ellipses) for choice in expr.choices], ellipses, expr.separator)]
        else:
            assert False

    return Root(traverse(expr.children, ellipses=[]))

def concatenate(exprs):
    children = []
    for expr in exprs:
        if not isinstance(expr, Root):
            raise ValueError("Can only concatenate Root expressions")
        children.extend(expr.copy().children)
    return Root(children)
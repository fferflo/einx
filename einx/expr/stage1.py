from collections import defaultdict
import re
import uuid
import einx


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


class Composition(Expression):
    def __init__(self, inner, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.inner = inner
        self.inner.parent = self

    def all(self):
        yield self
        yield from self.inner.all()

    def __str__(self):
        return "(" + str(self.inner) + ")"

    def __deepcopy__(self):
        return Composition(self.inner.__deepcopy__(), self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Composition) and self.inner == other.inner

    def __hash__(self):
        return 87123 + hash(self.inner)

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        return self.inner[i]

    def expansion(self):
        return 1

    @property
    def direct_children(self):
        yield self.inner


class Marker(Expression):
    @staticmethod
    def maybe(inner, *args, **kwargs):
        if isinstance(inner, List) and len(inner) == 0:
            return inner
        else:
            return Marker(inner, *args, **kwargs)

    def __init__(self, inner, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.inner = inner
        self.inner.parent = self
        assert not (isinstance(inner, List) and len(inner) == 0)

    def all(self):
        yield self
        yield from self.inner.all()

    def __str__(self):
        return "[" + str(self.inner) + "]"

    def __deepcopy__(self):
        return Marker(self.inner.__deepcopy__(), self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Marker) and self.inner == other.inner

    def __hash__(self):
        return 91236 + hash(self.inner)

    def expansion(self):
        return self.inner.expansion()

    @property
    def direct_children(self):
        yield self.inner


class NamedAxis(Expression):
    def __init__(self, name, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.name = name

    def all(self):
        yield self

    def __str__(self):
        return self.name

    def __deepcopy__(self):
        return NamedAxis(self.name, self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, NamedAxis) and self.name == other.name

    def __hash__(self):
        return 12345 + hash(self.name)

    def expansion(self):
        return 1

    @property
    def direct_children(self):
        yield from ()


class UnnamedAxis(Expression):
    def __init__(self, value, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.value = value

    def all(self):
        yield self

    def __str__(self):
        return str(self.value)

    def __deepcopy__(self):
        return UnnamedAxis(self.value, self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, UnnamedAxis) and self.value == other.value

    def __hash__(self):
        return 67890 + hash(self.value)

    def expansion(self):
        return 1

    @property
    def direct_children(self):
        yield from ()


class Ellipsis(Expression):
    anonymous_variable_name = "_anonymous_ellipsis_axis"

    def maybe(inner, *args, **kwargs):
        if isinstance(inner, List) and len(inner) == 0:
            return inner
        else:
            return Ellipsis(inner, *args, **kwargs)

    def __init__(self, inner, begin_pos=-1, end_pos=-1, ellipsis_id=None):
        Expression.__init__(self, begin_pos, end_pos)
        self.inner = inner
        self.inner.parent = self
        self.ellipsis_id = uuid.uuid4().int if ellipsis_id is None else ellipsis_id
        assert not (isinstance(inner, List) and len(inner) == 0)

    def all(self):
        yield self
        yield from self.inner.all()

    def __str__(self):
        n = str(self.inner)
        if isinstance(self.inner, List) and len(self.inner.children) > 1:
            n = "{" + n + "}"
        return n + _ellipsis

    def __deepcopy__(self):
        return Ellipsis(self.inner.__deepcopy__(), self.begin_pos, self.end_pos, self.ellipsis_id)

    def __eq__(self, other):
        return isinstance(other, Ellipsis) and self.inner == other.inner

    def __hash__(self):
        return 34567 + hash(self.inner)

    def expansion(self):
        if self.inner.expansion() == 0:
            return 0
        else:
            return None

    @property
    def direct_children(self):
        yield self.inner


class Concatenation(Expression):
    def maybe(l, *args, **kwargs):
        if len(l) == 1:
            return l[0]
        else:
            return Concatenation(l, *args, **kwargs)

    def __init__(self, children, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.children = children
        for child in self.children:
            child.parent = self

    def all(self):
        yield self
        for child in self.children:
            yield from child.all()

    def __str__(self):
        return " + ".join([str(c) for c in self.children])

    def __deepcopy__(self):
        return Concatenation(
            [c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos
        )

    def __eq__(self, other):
        return isinstance(other, Concatenation) and self.children == other.children

    def __hash__(self):
        return 234 + hash(tuple(self.children))

    def __len__(self):
        return len(self.children)

    def __getitem__(self, i):
        return self.children[i]

    def expansion(self):
        return 1

    @property
    def direct_children(self):
        yield from self.children


class List(Expression):
    @staticmethod
    def maybe(l, *args, **kwargs):
        if len(l) == 1:
            return l[0]
        else:
            return List(l, *args, **kwargs)

    def __init__(self, children, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.children = children
        for child in self.children:
            child.parent = self

    def all(self):
        yield self
        for child in self.children:
            yield from child.all()

    def __str__(self):
        return " ".join([str(c) for c in self.children])

    def __deepcopy__(self):
        return List([c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, List) and self.children == other.children

    def __hash__(self):
        return 2333 + hash(tuple(self.children))

    def __len__(self):
        return len(self.children)

    def __getitem__(self, i):
        return self.children[i]

    def expansion(self):
        child_expansions = [c.expansion() for c in self.children]
        if any(e is None for e in child_expansions):
            return None
        else:
            return sum(child_expansions)

    @property
    def direct_children(self):
        yield from self.children


class Args(Expression):
    @staticmethod
    def maybe(*args, **kwargs):
        return Args(*args, **kwargs)

    def __init__(self, children, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        self.children = children
        for child in self.children:
            assert not isinstance(child, Args)
            child.parent = self

    def all(self):
        yield self
        for child in self.children:
            yield from child.all()

    def __str__(self):
        return ", ".join([str(c) for c in self.children])

    def __deepcopy__(self):
        return Args([c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Args) and self.children == other.children

    def __hash__(self):
        return 233314 + hash(tuple(self.children))

    def __getitem__(self, i):
        return self.children[i]

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.children)


class Op(Expression):
    def __init__(self, children, begin_pos=-1, end_pos=-1):
        Expression.__init__(self, begin_pos, end_pos)
        assert len(children) >= 1
        self.children = children
        for child in self.children:
            child.parent = self

    def all(self):
        yield self
        for child in self.children:
            yield from child.all()

    def __str__(self):
        return " -> ".join([str(c) for c in self.children])

    def __deepcopy__(self):
        return Op([c.__deepcopy__() for c in self.children], self.begin_pos, self.end_pos)

    def __eq__(self, other):
        return isinstance(other, Op) and self.children == other.children

    def __hash__(self):
        return 961121 + hash(tuple(self.children))

    def __getitem__(self, i):
        return self.children[i]

    def __len__(self):
        return len(self.children)

    def __iter__(self):
        return iter(self.children)


class Token:
    def __init__(self, pos, text):
        self.begin_pos = pos
        self.end_pos = pos + len(text)
        self.text = text

    def __str__(self):
        return self.text

    def __repr__(self):
        return f'Token("{self.text}")'


class TokenList:
    def __init__(self, tokens, pos):
        if isinstance(tokens, TokenList) and len(tokens) == 1 and isinstance(tokens[0], TokenList):
            tokens = tokens[0].tokens
        self.tokens = tokens
        self.text = "".join([t.text for t in self.tokens])
        self.begin_pos = pos
        if len(self.tokens) > 0:
            assert self.tokens[0].begin_pos == pos
            self.end_pos = self.tokens[-1].end_pos
        else:
            self.end_pos = pos

    def __str__(self):
        return "".join([str(t) for t in self.tokens])

    def __repr__(self):
        return f'TokenList("{self}")'


_parentheses = {
    "(": ")",
    "[": "]",
}
_delimiters_front = set(_parentheses.keys())
_delimiters_back = set(_parentheses.values())
_nary_ops = ["->", "|", ",", "+", " "]
_ellipsis = "..."
_axis_name = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_literals = _nary_ops + list(_delimiters_front) + list(_delimiters_back) + [_ellipsis]


def _delimiter_to_name(delimiter):
    if delimiter in ["(", ")"]:
        return "parenthesis"
    elif delimiter in ["[", "]"]:
        return "bracket"
    else:
        raise ValueError(f"Invalid delimiter: {delimiter}")


def parse_op(text):
    signature = einx.expr.CallSignature(text)

    # ##### Lexer: Convert string into list of (string-)tokens #####
    tokens = []
    start_pos = 0

    def next_token(end_pos):
        nonlocal start_pos
        if start_pos != end_pos:
            token_text = text[start_pos:end_pos]
            if (
                token_text not in _literals
                and token_text not in _nary_ops
                and not _axis_name.fullmatch(token_text)
                and not token_text.isdigit()
            ):
                raise einx.SyntaxError(
                    text,
                    range(start_pos, end_pos),
                    f"The expression '{token_text}' is not allowed:",
                    """The following expressions are allowed:
- Literals: -> + , [ ] ( ) ... whitespace
- Named axes: Must match the regex [a-zA-Z_][a-zA-Z0-9_]*
- Unnamed axes: Must match the regex [0-9]+
""",
                )
            tokens.append(Token(start_pos, token_text))
            start_pos = end_pos

    pos = 0
    while pos < len(text):
        for l in _literals:
            if text[pos:].startswith(l):
                next_token(pos)
                next_token(pos + len(l))
                pos += len(l)
                break
        else:
            pos += 1
    next_token(pos)

    # ##### Parser: Remove duplicate whitespaces #####
    tokens2 = []
    last_was_whitespace = False
    for token in tokens:
        if token.text == " ":
            if last_was_whitespace:
                continue
            last_was_whitespace = True
        else:
            last_was_whitespace = False
        tokens2.append(token)
    tokens = tokens2

    # ##### Parser: Convert list of tokens to tree of tokens #####
    stack = [[]]
    for token in tokens:
        if token.text in _delimiters_front:
            stack.append([])
            stack[-1].append(token)
        elif token.text in _delimiters_back:
            if len(stack) == 1 or _parentheses[stack[-1][0].text] != token.text:
                raise einx.SyntaxError(
                    text,
                    range(token.begin_pos, token.end_pos),
                    f"Found a closing {_delimiter_to_name(token.text)} that is not opened:",
                )
            stack[-1].append(token)
            begin_pos = stack[-1][0].begin_pos
            group = stack.pop()
            stack[-1].append(TokenList(group, begin_pos))
        else:
            stack[-1].append(token)
    if len(stack) > 1:
        raise einx.SyntaxError(
            text,
            range(stack[-1][0].begin_pos, stack[-1][0].end_pos),
            f"Found an opening {_delimiter_to_name(stack[-1][0].text)} that is not closed:",
        )
    expression = TokenList(stack[0], 0)

    # ##### Parser: Convert tokens to expressions #####
    def parse(in_tokens):
        assert isinstance(in_tokens, TokenList)
        begin_pos = in_tokens.begin_pos
        end_pos = in_tokens.end_pos
        in_tokens = in_tokens.tokens

        # Ignore starting and trailing whitespace
        while len(in_tokens) > 0 and in_tokens[0].text == " ":
            in_tokens.pop(0)
        while len(in_tokens) > 0 and in_tokens[-1].text == " ":
            in_tokens.pop(-1)

        # Empty expression
        if len(in_tokens) == 0:
            return List([], begin_pos, end_pos)
        elif len(in_tokens) == 1 and isinstance(in_tokens[0], TokenList):
            return parse(in_tokens[0])

        begin_pos = in_tokens[0].begin_pos
        end_pos = in_tokens[-1].end_pos

        # Parentheses
        if in_tokens[0].text in _delimiters_front:
            assert len(in_tokens) >= 2 and in_tokens[-1].text in _delimiters_back
            if in_tokens[0].text == "(":
                op = Composition
            elif in_tokens[0].text == "[":
                op = Marker.maybe
            else:
                raise AssertionError()
            inner = parse(TokenList(in_tokens[1:-1], in_tokens[1].begin_pos))
            return op(
                inner,
                begin_pos,
                end_pos,
            )

        # N-ary operators
        for nary_op in _nary_ops:
            if any(t.text == nary_op for t in in_tokens):
                # Split expression into operands
                operands = []
                current_operand_tokens = []
                for token in in_tokens:
                    if token.text == nary_op:
                        operands.append(
                            TokenList(
                                current_operand_tokens,
                                current_operand_tokens[0].begin_pos
                                if len(current_operand_tokens) > 0
                                else token.begin_pos,
                            )
                        )
                        current_operand_tokens = []
                    else:
                        current_operand_tokens.append(token)
                operands.append(
                    TokenList(
                        current_operand_tokens,
                        current_operand_tokens[0].begin_pos
                        if len(current_operand_tokens) > 0
                        else token.end_pos,
                    )
                )
                if nary_op == " ":
                    # Ignore empty operands
                    operands = [t for t in operands if len(t.tokens) > 0]

                # Create operands
                operands = [parse(operand) for operand in operands]

                # Create expression
                if nary_op == " ":
                    op = List.maybe
                elif nary_op in {"->", "|"}:
                    op = Op
                elif nary_op == ",":
                    op = Args
                elif nary_op == "+":
                    op = Concatenation
                    invalid_operands = [
                        o
                        for o in operands
                        if not isinstance(o, (NamedAxis, UnnamedAxis, Composition))
                    ]
                    if len(invalid_operands) > 0:
                        pos = []
                        for operand in invalid_operands:
                            pos.extend(range(operand.begin_pos, operand.end_pos))
                        for t in in_tokens:
                            if t.text == "+":
                                pos.extend(range(t.begin_pos, t.end_pos))
                        raise einx.SyntaxError(
                            text,
                            pos,
                            "Only named axes, unnamed axes, and compositions are allowed as "
                            "operands of a concatenation operator ('+').",
                        )
                else:
                    raise AssertionError()
                return op(operands, begin_pos, end_pos)

        # Ellipsis
        if in_tokens[-1].text == _ellipsis and len(in_tokens) <= 2:
            if len(in_tokens) == 1:
                operand = NamedAxis(
                    Ellipsis.anonymous_variable_name,
                    in_tokens[0].begin_pos,
                    in_tokens[0].begin_pos,
                )
            else:
                assert len(in_tokens) == 2
                operand = parse(TokenList(in_tokens[:1], in_tokens[0].begin_pos))
            return Ellipsis(
                operand,
                in_tokens[0].begin_pos,
                in_tokens[-1].end_pos,
            )

        # Axis
        if len(in_tokens) == 1:
            value = in_tokens[0].text.strip()
            if value.isdigit():
                return UnnamedAxis(int(value), in_tokens[0].begin_pos, in_tokens[0].end_pos)
            else:
                assert _axis_name.fullmatch(in_tokens[0].text), (
                    f"Invalid axis name: {in_tokens[0].text}"
                )
                return NamedAxis(value, in_tokens[0].begin_pos, in_tokens[0].end_pos)

        message = (
            f"The expression '{text[in_tokens[0].begin_pos : in_tokens[-1].end_pos]}' is not valid."
        )
        if len(in_tokens) > 1:
            message += " Are you maybe missing a whitespace?"
        raise einx.SyntaxError(text, range(in_tokens[0].begin_pos, in_tokens[-1].end_pos), message)

        raise AssertionError()

    expression = parse(expression)

    # ##### Move up and merge Op #####
    def move_up(expr):
        if isinstance(expr, (NamedAxis, UnnamedAxis)):
            return Op([expr.__deepcopy__()])
        elif isinstance(expr, Composition):
            op = move_up(expr.inner)
            return Op(
                [Composition(arglist, expr.begin_pos, expr.end_pos) for arglist in op.children],
                op.begin_pos,
                op.end_pos,
            )
        elif isinstance(expr, Marker):
            op = move_up(expr.inner)
            return Op(
                [Marker.maybe(arglist, expr.begin_pos, expr.end_pos) for arglist in op.children],
                op.begin_pos,
                op.end_pos,
            )
        elif isinstance(expr, Ellipsis):
            op = move_up(expr.inner)
            return Op(
                [
                    Ellipsis.maybe(arglist, expr.begin_pos, expr.end_pos, expr.ellipsis_id)
                    for arglist in op.children
                ],
                op.begin_pos,
                op.end_pos,
            )

        elif isinstance(expr, (List, Concatenation, Args)):
            _class = type(expr)
            children = [move_up(c) for c in expr.children]
            new_children = []

            nums = {len(c) for c in children if len(c) != 1}
            if len(nums) > 1:
                raise einx.SyntaxError(
                    text,
                    signature.get_pos_for_literal("->"),
                    "All '->' operators must appear at the same level of the expression tree.",
                )
            num = nums.pop() if len(nums) > 0 else 1

            new_arglists = []
            for idx in range(num):
                new_children = []
                for op in children:
                    if len(op) == 1:
                        new_children.append(op[0])
                    else:
                        new_children.append(op[idx])
                new_arglists.append(_class.maybe(new_children, expr.begin_pos, expr.end_pos))

            return Op(new_arglists, expr.begin_pos, expr.end_pos)

        elif isinstance(expr, Op):
            return Op(
                [arglist for child in expr.children for arglist in move_up(child).children],
                expr.begin_pos,
                expr.end_pos,
            )

        else:
            raise AssertionError(f"Invalid expression type {type(expr)}")

    expression = move_up(expression)

    # ##### Move up and merge Args #####
    def move_up(expr):
        if isinstance(expr, (NamedAxis, UnnamedAxis)):
            return Args([expr.__deepcopy__()])
        elif isinstance(expr, Composition):
            args = move_up(expr.inner)
            return Args(
                [Composition(arg, expr.begin_pos, expr.end_pos) for arg in args.children],
                args.begin_pos,
                args.end_pos,
            )
        elif isinstance(expr, Marker):
            args = move_up(expr.inner)
            return Args(
                [Marker.maybe(arg, expr.begin_pos, expr.end_pos) for arg in args.children],
                args.begin_pos,
                args.end_pos,
            )
        elif isinstance(expr, Ellipsis):
            args = move_up(expr.inner)
            return Args(
                [
                    Ellipsis.maybe(arg, expr.begin_pos, expr.end_pos, expr.ellipsis_id)
                    for arg in args.children
                ],
                args.begin_pos,
                args.end_pos,
            )

        elif isinstance(expr, (List, Concatenation)):
            _class = type(expr)
            children = [move_up(c) for c in expr.children]
            new_children = []

            nums = {len(c) for c in children if len(c) != 1}
            if len(nums) > 1:
                raise einx.SyntaxError(
                    text,
                    signature.get_pos_for_literal("->"),
                    "All ',' operators must appear at the same level of the expression tree.",
                )
            num = nums.pop() if len(nums) > 0 else 1

            new_args = []
            for idx in range(num):
                new_children = []
                for args in children:
                    if len(args) == 1:
                        new_children.append(args[0])
                    else:
                        new_children.append(args[idx])
                new_args.append(_class.maybe(new_children, expr.begin_pos, expr.end_pos))

            return Args(new_args, expr.begin_pos, expr.end_pos)

        elif isinstance(expr, Args):
            return Args(
                [arg for child in expr.children for arg in move_up(child).children],
                expr.begin_pos,
                expr.end_pos,
            )

        else:
            raise AssertionError()

    assert isinstance(expression, Op)
    expression = Op(
        [move_up(c) for c in expression.children], expression.begin_pos, expression.end_pos
    )

    # ##### Semantic checks #####

    # Op cannot have more than two children
    if len(expression.children) > 2:
        raise einx.SyntaxError(
            text,
            signature.get_pos_for_literal("->"),
            "The expression must not contain more than one '->' operator.",
        )

    # Semantic check: Axis names can only be used once per expression
    def traverse(expr, key, axes_by_key):
        if isinstance(expr, list):
            for expr in expr:
                traverse(expr, key, axes_by_key)
        elif isinstance(expr, NamedAxis):
            axes_by_key[(key + (expr.name,))].append(expr)
        elif isinstance(expr, UnnamedAxis):
            pass
        elif isinstance(expr, Composition):
            traverse(expr.inner, key, axes_by_key)
        elif isinstance(expr, List):
            traverse(expr.children, key, axes_by_key)
        elif isinstance(expr, Concatenation):
            for i, c in enumerate(expr.children):
                traverse(c, key + ((id(expr), i),), axes_by_key)
        elif isinstance(expr, Marker):
            traverse(expr.inner, key, axes_by_key)
        elif isinstance(expr, Ellipsis):
            traverse(expr.inner, key, axes_by_key)
        else:
            raise TypeError(f"Invalid expression type {type(expr)}")

    def check(root):
        axes_by_key = defaultdict(list)
        traverse(root, (), axes_by_key)
        for key in list(axes_by_key.keys()):
            exprs = []
            for i in range(len(key) + 1):
                exprs.extend(axes_by_key[key[:i]])
            if len(exprs) > 1:
                raise einx.SyntaxError(
                    text,
                    signature.get_pos_for_exprs(exprs),
                    "Each axis name must only be used once per operand, but the axis "
                    f"'{exprs[0].name}' is used multiple times in '{root}'.",
                )

    for arglist in expression.children:
        for arg in arglist.children:
            check(arg)

    # Axes may only appear with brackets or without brackets, but not both.
    axis_names = {expr.name for expr in expression.all() if isinstance(expr, NamedAxis)}
    for axis_name in axis_names:
        marked = 0
        unmarked = 0
        for expr in expression.all():
            if isinstance(expr, NamedAxis) and expr.name == axis_name:
                if is_marked(expr):
                    marked += 1
                else:
                    unmarked += 1
        if marked > 0 and unmarked > 0:
            pos = []
            for expr in expression.all():
                if isinstance(expr, NamedAxis) and expr.name == axis_name:
                    pos.extend(range(expr.begin_pos, expr.end_pos))
                    parent = expr.parent
                    while parent is not None:
                        if isinstance(parent, Marker):
                            pos.extend([parent.begin_pos, parent.end_pos - 1])
                        parent = parent.parent
            raise einx.SyntaxError(
                text,
                pos,
                f"There are multiple occurrences of axis {axis_name} with inconsistent bracket "
                "usage:",
                post_message="An axis may only appear with brackets or without brackets, but not "
                "both.",
            )

    return expression


def parse_args(text):
    op = parse_op(text)
    signature = einx.expr.CallSignature(text)
    if len(op.children) != 1:
        raise einx.SyntaxError(
            text,
            signature.get_pos_for_literal("->"),
            "The expression must not contain an '->' operator.",
        )
    assert isinstance(op.children[0], Args)
    return op.children[0]


def parse_arg(text):
    if isinstance(text, Expression):
        return text
    args = parse_args(text)
    signature = einx.expr.CallSignature(text)
    if len(args.children) != 1:
        raise einx.SyntaxError(
            text,
            signature.get_pos_for_literal(","),
            "The expression must not contain a ',' operator.",
        )
    return args.children[0]


def expr_map(f):
    def outer(expr, *args, **kwargs):
        # Wrap the user function to return a list of expressions
        def f2(expr):
            t = f(expr, *args, **kwargs)
            if t is None:
                return None, expr_map.CONTINUE
            expr, signal = t

            if isinstance(expr, list) or expr is None:
                return expr, signal
            if isinstance(expr, List):
                return expr.children, signal
            elif isinstance(expr, Expression):
                return [expr], signal
            else:
                raise TypeError(f"Invalid return type {type(expr)}")

        return List.maybe(_expr_map(expr, f2))

    return outer


expr_map.CONTINUE = 1
expr_map.COPY_AND_STOP = 2
expr_map.REPLACE_AND_STOP = 3
expr_map.REPLACE_AND_CONTINUE = 4


def _expr_map(expr, f):
    exprs, signal = f(expr)
    if signal == expr_map.REPLACE_AND_STOP:
        assert isinstance(exprs, list)
        return exprs
    elif signal == expr_map.COPY_AND_STOP:
        return [expr.__deepcopy__()]
    elif signal == expr_map.REPLACE_AND_CONTINUE:
        return [c for expr in exprs for c in _expr_map(expr, f)]

    if isinstance(expr, NamedAxis):
        return [expr.__deepcopy__()]
    elif isinstance(expr, UnnamedAxis):
        return [expr.__deepcopy__()]
    elif isinstance(expr, Composition):
        return [Composition(List.maybe(_expr_map(expr.inner, f)))]
    elif isinstance(expr, List):
        return [c2 for c1 in expr.children for c2 in _expr_map(c1, f)]
    elif isinstance(expr, Concatenation):
        return [Concatenation([List.maybe(_expr_map(c, f)) for c in expr.children])]
    elif isinstance(expr, Marker):
        x = _expr_map(expr.inner, f)
        if len(x) == 0:
            # Drop empty marker
            return []
        else:
            return [Marker(List.maybe(x))]
    elif isinstance(expr, Ellipsis):
        return [Ellipsis(List.maybe(_expr_map(expr.inner, f)), ellipsis_id=expr.ellipsis_id)]
    else:
        raise TypeError(f"Invalid expression type {type(expr)}")


@expr_map
def demark(expr):
    if isinstance(expr, Marker):
        return expr.inner, expr_map.REPLACE_AND_CONTINUE


def any_parent_is(expr, pred, include_self=True):
    if not include_self:
        if expr.parent is None:
            return False
        expr = expr.parent
    while expr is not None:
        if pred(expr):
            return True
        expr = expr.parent
    return False


def is_marked(expr):
    return any_parent_is(expr, lambda expr: isinstance(expr, Marker))


def _get_marked(expr):
    if isinstance(expr, NamedAxis):
        return []
    elif isinstance(expr, UnnamedAxis):
        return []
    elif isinstance(expr, Ellipsis):
        inner = _get_marked(expr.inner)
        if len(inner) > 0:
            return [Ellipsis(List.maybe(inner), ellipsis_id=expr.ellipsis_id)]
        else:
            return []
    elif isinstance(expr, Marker):
        return [expr.inner.__deepcopy__()]
    elif isinstance(expr, Concatenation):
        return [Concatenation.maybe([x for c in expr.children for x in _get_marked(c)])]
    elif isinstance(expr, Composition):
        return [Composition(List.maybe(_get_marked(expr.inner)))]
    elif isinstance(expr, List):
        return [List.maybe([x for c in expr.children for x in _get_marked(c)])]
    else:
        raise TypeError(f"Invalid expression type {type(expr)}")


def get_marked(expr):
    return List.maybe(_get_marked(expr))


def get_unmarked(expr):
    return remove(expr, lambda expr: is_marked(expr))


@expr_map
def replace(expr, f):
    expr = f(expr)
    if expr is not None:
        return expr, expr_map.REPLACE_AND_STOP


@expr_map
def remove(expr, pred):
    if pred(expr):
        return [], expr_map.REPLACE_AND_STOP

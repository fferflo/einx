from collections import defaultdict
import re
from .. import ExpressionIndicator
from einx._src.frontend.errors import SyntaxError
from .tree import *
from .transform import *
import uuid


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


_parentheses = {"(": ")", "[": "]"}
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
    indicator = ExpressionIndicator(text)

    # ##### Lexer: Convert string into list of (string-)tokens #####
    tokens = []
    start_pos = 0

    def next_token(end_pos):
        nonlocal start_pos
        if start_pos != end_pos:
            token_text = text[start_pos:end_pos]
            if token_text not in _literals and token_text not in _nary_ops and not _axis_name.fullmatch(token_text) and not token_text.isdigit():
                raise SyntaxError(
                    text,
                    pos=range(start_pos, end_pos),
                    message=f"""The expression '{token_text}' is not allowed:\n%EXPR%\nThe following expressions are allowed:
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
                raise SyntaxError(
                    text, pos=range(token.begin_pos, token.end_pos), message=f"Found a closing {_delimiter_to_name(token.text)} that is not opened:\n%EXPR%"
                )
            stack[-1].append(token)
            begin_pos = stack[-1][0].begin_pos
            group = stack.pop()
            stack[-1].append(TokenList(group, begin_pos))
        else:
            stack[-1].append(token)
    if len(stack) > 1:
        raise SyntaxError(
            text,
            pos=range(stack[-1][0].begin_pos, stack[-1][0].end_pos),
            message=f"Found an opening {_delimiter_to_name(stack[-1][0].text)} that is not closed:\n%EXPR%",
        )
    expression = TokenList(stack[0], 0)

    # ##### Parser: Convert tokens to expressions #####
    def parse(in_tokens, is_parent_composition=False):
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
            return List.create([], begin_pos, end_pos)
        elif len(in_tokens) == 1 and isinstance(in_tokens[0], TokenList):
            return parse(in_tokens[0], is_parent_composition=is_parent_composition)

        begin_pos = in_tokens[0].begin_pos
        end_pos = in_tokens[-1].end_pos

        # Delimiters
        if in_tokens[0].text in _delimiters_front:
            assert len(in_tokens) >= 2 and in_tokens[-1].text in _delimiters_back
            inner = parse(TokenList(in_tokens[1:-1], in_tokens[1].begin_pos), is_parent_composition=in_tokens[0].text == "(")

            if in_tokens[0].text == "(":
                if isinstance(inner, ConcatenatedAxis):
                    return inner
                else:
                    return FlattenedAxis.create(inner, begin_pos, end_pos)
            elif in_tokens[0].text == "[":
                return Brackets.create(inner, begin_pos, end_pos)
            else:
                raise AssertionError()

        # N-ary operators
        for nary_op in _nary_ops:
            if any(t.text == nary_op for t in in_tokens):
                # Split expression into operands
                operands = []
                current_operand_tokens = []
                for token in in_tokens:
                    if token.text == nary_op:
                        operands.append(
                            TokenList(current_operand_tokens, current_operand_tokens[0].begin_pos if len(current_operand_tokens) > 0 else token.begin_pos)
                        )
                        current_operand_tokens = []
                    else:
                        current_operand_tokens.append(token)
                operands.append(TokenList(current_operand_tokens, current_operand_tokens[0].begin_pos if len(current_operand_tokens) > 0 else token.end_pos))
                if nary_op == " ":
                    # Ignore empty operands
                    operands = [t for t in operands if len(t.tokens) > 0]

                # Create operands
                operands = [parse(operand, is_parent_composition=False) for operand in operands]

                # Create expression
                if nary_op == " ":
                    return List.create(operands, begin_pos, end_pos)
                elif nary_op == "->":
                    return Op(operands, begin_pos, end_pos)
                elif nary_op == ",":
                    return Args(operands, begin_pos, end_pos)
                elif nary_op == "+":
                    invalid_operands = [o for o in operands if not isinstance(o, Axis | FlattenedAxis)]
                    if len(invalid_operands) > 0:
                        pos = []
                        for operand in invalid_operands:
                            pos.extend(range(operand.begin_pos, operand.end_pos))
                        for t in in_tokens:
                            if t.text == "+":
                                pos.extend(range(t.begin_pos, t.end_pos))
                        raise SyntaxError(
                            text,
                            pos=pos,
                            message="Only named axes, unnamed axes, and flattened axes are allowed as operands of a concatenation operator ('+').\n%EXPR%",
                        )
                    if not is_parent_composition:
                        raise SyntaxError(text, pos=range(begin_pos, end_pos), message="Concatenated axes must be wrapped in parentheses.\n%EXPR%")
                    return ConcatenatedAxis.create(operands, begin_pos, end_pos)
                else:
                    raise AssertionError()

        # Ellipsis
        if in_tokens[-1].text == _ellipsis and len(in_tokens) <= 2:
            if len(in_tokens) == 1:
                operand = Axis(Ellipsis.anonymous_variable_name, None, in_tokens[0].begin_pos, in_tokens[0].begin_pos)
            else:
                assert len(in_tokens) == 2
                operand = parse(TokenList(in_tokens[:1], in_tokens[0].begin_pos), is_parent_composition=False)
            return Ellipsis.create(operand, in_tokens[0].begin_pos, in_tokens[-1].end_pos, ellipsis_id=uuid.uuid4().int)

        # Axis
        if len(in_tokens) == 1:
            value = in_tokens[0].text.strip()
            if value.isdigit():
                name = f"unnamed.{uuid.uuid4().int}"
                return Axis(name, int(value), in_tokens[0].begin_pos, in_tokens[0].end_pos)
            else:
                assert _axis_name.fullmatch(in_tokens[0].text), f"Invalid axis name: {in_tokens[0].text}"
                return Axis(value, None, in_tokens[0].begin_pos, in_tokens[0].end_pos)

        message = f"The expression '{text[in_tokens[0].begin_pos : in_tokens[-1].end_pos]}' is not valid."
        if len(in_tokens) > 1:
            message += " Are you maybe missing a whitespace?"
        raise SyntaxError(text, pos=range(in_tokens[0].begin_pos, in_tokens[-1].end_pos), message=message + "\n%EXPR%")

        raise AssertionError()

    expression = parse(expression)

    # ##### Move up and merge Op #####
    def move_up(expr):
        if isinstance(expr, Axis):
            return Op([expr.__deepcopy__()])
        elif isinstance(expr, FlattenedAxis):
            op = move_up(expr.inner)
            return Op([FlattenedAxis.create(arglist, expr.begin_pos, expr.end_pos) for arglist in op.children], op.begin_pos, op.end_pos)
        elif isinstance(expr, Brackets):
            op = move_up(expr.inner)
            return Op([Brackets.create(arglist, expr.begin_pos, expr.end_pos) for arglist in op.children], op.begin_pos, op.end_pos)
        elif isinstance(expr, Ellipsis):
            op = move_up(expr.inner)
            return Op([Ellipsis.create(arglist, expr.begin_pos, expr.end_pos, expr.ellipsis_id) for arglist in op.children], op.begin_pos, op.end_pos)

        elif isinstance(expr, List | ConcatenatedAxis | Args):
            _class = type(expr)
            children = [move_up(c) for c in expr.children]
            new_children = []

            nums = {len(c.children) for c in children if len(c.children) != 1}
            if len(nums) > 1:
                raise SyntaxError(
                    text, pos=indicator.get_pos_for_literal("->"), message="All '->' operators must appear at the same level of the expression tree.\n%EXPR%"
                )
            num = nums.pop() if len(nums) > 0 else 1

            new_arglists = []
            for idx in range(num):
                new_children = []
                for op in children:
                    if len(op.children) == 1:
                        new_children.append(op.children[0])
                    else:
                        new_children.append(op.children[idx])
                new_arglists.append(_class.create(new_children, expr.begin_pos, expr.end_pos))

            return Op(new_arglists, expr.begin_pos, expr.end_pos)

        elif isinstance(expr, Op):
            return Op([arglist for child in expr.children for arglist in move_up(child).children], expr.begin_pos, expr.end_pos)

        else:
            raise AssertionError(f"Invalid expression type {type(expr)}")

    expression = move_up(expression)

    # ##### Move up and merge Args #####
    def move_up(expr):
        if isinstance(expr, Axis):
            return Args([expr.__deepcopy__()])
        elif isinstance(expr, FlattenedAxis):
            args = move_up(expr.inner)
            return Args([FlattenedAxis.create(arg, expr.begin_pos, expr.end_pos) for arg in args.children], args.begin_pos, args.end_pos)
        elif isinstance(expr, Brackets):
            args = move_up(expr.inner)
            return Args([Brackets.create(arg, expr.begin_pos, expr.end_pos) for arg in args.children], args.begin_pos, args.end_pos)
        elif isinstance(expr, Ellipsis):
            args = move_up(expr.inner)
            return Args([Ellipsis.create(arg, expr.begin_pos, expr.end_pos, expr.ellipsis_id) for arg in args.children], args.begin_pos, args.end_pos)

        elif isinstance(expr, List | ConcatenatedAxis):
            _class = type(expr)
            children = [move_up(c) for c in expr.children]
            new_children = []

            nums = {len(c.children) for c in children if len(c.children) != 1}
            if len(nums) > 1:
                raise SyntaxError(
                    text, pos=indicator.get_pos_for_literal("->"), message="All ',' operators must appear at the same level of the expression tree.\n%EXPR%"
                )
            num = nums.pop() if len(nums) > 0 else 1

            new_args = []
            for idx in range(num):
                new_children = []
                for args in children:
                    if len(args.children) == 1:
                        new_children.append(args.children[0])
                    else:
                        new_children.append(args.children[idx])
                new_args.append(_class.create(new_children, expr.begin_pos, expr.end_pos))

            return Args(new_args, expr.begin_pos, expr.end_pos)

        elif isinstance(expr, Args):
            return Args([arg for child in expr.children for arg in move_up(child).children], expr.begin_pos, expr.end_pos)

        else:
            raise AssertionError()

    assert isinstance(expression, Op)
    expression = Op([move_up(c) for c in expression.children], expression.begin_pos, expression.end_pos)

    # ##### Drop redundant brackets #####
    def traverse(expr, is_in_brackets):
        if isinstance(expr, list):
            return [traverse(e, is_in_brackets) for e in expr]
        elif isinstance(expr, Axis):
            return expr.__deepcopy__()
        elif isinstance(expr, FlattenedAxis):
            return FlattenedAxis.create(traverse(expr.inner, is_in_brackets), expr.begin_pos, expr.end_pos)
        elif isinstance(expr, List):
            return List.create([traverse(c, is_in_brackets) for c in expr.children], expr.begin_pos, expr.end_pos)
        elif isinstance(expr, ConcatenatedAxis):
            return ConcatenatedAxis.create([traverse(c, is_in_brackets) for c in expr.children], expr.begin_pos, expr.end_pos)
        elif isinstance(expr, Brackets):
            if is_in_brackets:
                return traverse(expr.inner, True)
            else:
                return Brackets.create(traverse(expr.inner, True), expr.begin_pos, expr.end_pos)
        elif isinstance(expr, Ellipsis):
            return Ellipsis.create(traverse(expr.inner, is_in_brackets), expr.begin_pos, expr.end_pos, expr.ellipsis_id)
        elif isinstance(expr, Op):
            return Op([traverse(c, is_in_brackets) for c in expr.children], expr.begin_pos, expr.end_pos)
        elif isinstance(expr, Args):
            return Args([traverse(c, is_in_brackets) for c in expr.children], expr.begin_pos, expr.end_pos)
        else:
            raise TypeError(f"Invalid expression type {type(expr)}")

    expression = traverse(expression, False)

    # ##### Semantic checks #####

    # Op cannot have more than two children
    if len(expression.children) > 2:
        raise SyntaxError(text, pos=indicator.get_pos_for_literal("->"), message="The expression must not contain more than one '->' operator.\n%EXPR%")

    # Axes may only appear with brackets or without brackets, but not both.
    axis_names = {expr.name for expr in expression.nodes() if isinstance(expr, Axis)}
    for axis_name in axis_names:
        marked = 0
        unmarked = 0
        for expr in expression.nodes():
            if isinstance(expr, Axis) and expr.name == axis_name:
                if is_in_brackets(expr):
                    marked += 1
                else:
                    unmarked += 1
        if marked > 0 and unmarked > 0:
            pos = []
            for expr in expression.nodes():
                if isinstance(expr, Axis) and expr.name == axis_name:
                    pos.extend(range(expr.begin_pos, expr.end_pos))
                    parent = expr.parent
                    while parent is not None:
                        if isinstance(parent, Brackets):
                            pos.extend([parent.begin_pos, parent.end_pos - 1])
                        parent = parent.parent
            raise SyntaxError(
                text,
                pos=pos,
                message=f"There are multiple occurrences of axis {axis_name} with inconsistent bracket "
                "usage:\n%EXPR%\nAn axis may only appear with brackets or without brackets, but not "
                "both.",
            )

    return expression


def parse_args(text):
    indicator = ExpressionIndicator(text)
    op = parse_op(text)
    if len(op.children) != 1:
        raise SyntaxError(text, pos=indicator.get_pos_for_literal("->"), message="The expression must not contain a '->' operator.\n%EXPR%")
    assert isinstance(op.children[0], Args)
    return op.children[0]


def parse_arg(text):
    if isinstance(text, Expression):
        return text
    indicator = ExpressionIndicator(text)
    args = parse_args(text)
    if len(args.children) != 1:
        raise SyntaxError(text, pos=indicator.get_pos_for_literal(","), message="The expression must not contain a ',' operator.\n%EXPR%")
    return args.children[0]

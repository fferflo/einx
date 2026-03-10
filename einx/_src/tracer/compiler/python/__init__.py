import einx._src.tracer as tracer
import numpy as np
from collections import defaultdict
import itertools
from einx._src.util import pytree


class Block:
    def __init__(self):
        self.statements = []

    def prepend(self, definition):
        self.statements.insert(0, definition)

    def prepend_after_comments(self, definition):
        for i, statement in enumerate(self.statements):
            if not isinstance(statement, CommentStatement):
                self.statements.insert(i, definition)
                break
        else:
            self.statements.append(definition)

    def append(self, definition):
        self.statements.append(definition)

    def to_code(self, value_to_code):
        lines = []
        for statement in self.statements:
            output = statement.to_code(value_to_code)
            if isinstance(output, str):
                lines.append(output)
            else:
                lines.extend(output)
        return lines


class Statement:
    def __init__(self, to_code, inputs, output_variables, block):
        self.to_code = to_code
        self.inputs = list(pytree.flatten(inputs))
        self.output_variables = list(pytree.flatten(output_variables))
        self.block = block

    @property
    def input_variables(self):
        return {var for input in self.inputs for var in input.used_variables}


class CommentStatement(Statement):
    def __init__(self, comment, block):
        def to_code(value_to_code):
            return f"# {comment}"

        Statement.__init__(self, to_code, inputs=[], output_variables=[], block=block)


def comment_statement(comment, block):
    # def to_code(value_to_code):
    #     return f"# {comment}"

    # return Statement(to_code, inputs=[], output_variables=[], block=block)
    return CommentStatement(comment, block)


class Expression:
    def __init__(self, inputs, block):
        self.inputs = inputs
        self.block = block


class Variable(Expression):
    def __init__(self, block, allow_reusing_name):
        Expression.__init__(self, inputs=[], block=block)
        self.allow_reusing_name = allow_reusing_name

    @property
    def used_variables(self):
        return [self]


class Literal(Expression):
    def __init__(self, code, block):
        Expression.__init__(self, inputs=[], block=block)
        self.code = code

    @property
    def used_variables(self):
        return []


class Inlined(Expression):
    def __init__(self, to_code, inputs, block):
        Expression.__init__(self, inputs=inputs, block=block)
        self.to_code = to_code

    @property
    def used_variables(self):
        return {var for input in self.inputs for var in input.used_variables}


class CodeObject:
    def __init__(self, object):
        self.cached_expressions = {}

        from .scope import get_scopes

        self.scopes = get_scopes(object)

        from .usage import get_usages

        self.max_usage_num = get_usages(object)

        self.scopeid_to_block = {id(scope): Block() for scope in self.scopes.values()}
        self.root_block = self.scopeid_to_block[id(self.scopes.root)]

    def _to_key(self, x):
        if isinstance(x, list):
            return (0,) + tuple(id(i) for i in x)
        elif isinstance(x, tuple):
            return (1,) + tuple(id(i) for i in x)
        elif isinstance(x, dict):
            return (2,) + tuple(sorted(id(i) for i in x.keys())) + tuple(sorted(id(i) for i in x.values()))
        else:
            return id(x)

    def to_code(self, value_to_code):
        return self.root_block.to_code(value_to_code)

    @property
    def variables(self):
        return [v for v in self.cached_expressions.values() if isinstance(v, Variable)]

    @property
    def blocks(self):
        return list(self.scopeid_to_block.values())

    def _get_scope_for(self, obj):
        return self.scopes[obj]

    def get_block_for(self, obj):
        scope = self._get_scope_for(obj)
        return self.scopeid_to_block[id(scope)]

    def add_variable_for(self, obj, allow_reusing_name=True):
        variable = Variable(block=self.get_block_for(obj), allow_reusing_name=allow_reusing_name)
        self[obj] = variable
        return variable

    def define(self, obj, expression, no_inline=False, force_inline=False):
        if self.max_usage_num[obj] > 1:
            no_inline = True

        if no_inline and force_inline:
            raise ValueError("Cannot have both no_inline and force_inline set to True.")

        inline = force_inline or not no_inline

        if inline:
            self[obj] = expression
        else:
            variable = self.add_variable_for(obj)

            def left_to_code(value_to_code):
                return value_to_code(variable)

            def to_code(value_to_code):
                left = left_to_code(value_to_code)
                right = expression.to_code(value_to_code)
                return f"{left} = {right}"

            block = self.get_block_for(obj)
            block.append(Statement(to_code, inputs=[expression], output_variables=[variable], block=block))

    def __contains__(self, obj):
        return self._to_key(obj) in self.cached_expressions

    def __getitem__(self, obj):
        return self.cached_expressions[self._to_key(obj)]

    def __setitem__(self, obj, expression):
        if not isinstance(obj, tracer.Tracer | list | tuple | dict | tracer.Graph):
            raise ValueError(f"Got invalid object {obj} of type {type(obj)}.")
        if not isinstance(expression, Expression):
            raise ValueError(f"Got invalid expression {expression} of type {type(expression)}.")
        key = self._to_key(obj)
        assert key not in self.cached_expressions
        self.cached_expressions[key] = expression

        if isinstance(obj, list | tuple):
            for k, v in enumerate(obj):

                def to_code(value_to_code, k=k):
                    return f"{value_to_code(expression)}[{k}]"

                self[v] = Inlined(to_code, inputs=[expression], block=expression.block)
        elif isinstance(obj, dict):
            for k, v in obj.items():

                def to_code(value_to_code, k=k):
                    return f"{value_to_code(expression)}[{k}]"

                self[v] = Inlined(to_code, inputs=[expression], block=expression.block)


def _debug_value_to_code(x):
    if isinstance(x, Variable):
        return "V"
    elif isinstance(x, Literal):
        return "L"
    elif isinstance(x, Inlined):
        return x.to_code(_debug_value_to_code)
    else:
        raise ValueError(f"Got {x} of type {type(x)}.")


def compile(object, return_code=False):
    name_hints = defaultdict(list)

    code = CodeObject(object)

    def _at(obj, key):
        obj = _get_expression_for(obj)
        slices = key
        if not isinstance(slices, tuple):
            slices = (slices,)
        assert isinstance(slices, tuple)

        def _get_expression_for_slice(s):
            if isinstance(s, slice):
                return slice(
                    _get_expression_for(s.start) if s.start is not None else None,
                    _get_expression_for(s.stop) if s.stop is not None else None,
                    _get_expression_for(s.step) if s.step is not None else None,
                )
            else:
                return _get_expression_for(s)

        slices = [_get_expression_for_slice(s) for s in slices]

        inputs = [obj]
        for s in slices:
            if isinstance(s, slice):
                if s.start is not None:
                    inputs.append(s.start)
                if s.stop is not None:
                    inputs.append(s.stop)
                if s.step is not None:
                    inputs.append(s.step)
            else:
                inputs.append(s)

        def to_code(value_to_code):
            if len(slices) == 0:
                key = "()"
            else:

                def slice_to_code(s):
                    if isinstance(s, slice):
                        x = ""
                        if s.start is not None:
                            x += value_to_code(s.start)
                        x += ":"
                        if s.stop is not None:
                            x += value_to_code(s.stop)
                        if s.step is not None:
                            x += ":" + value_to_code(s.step)
                        return x
                    else:
                        return value_to_code(s)

                key = ", ".join(slice_to_code(s) for s in slices)

            return f"{value_to_code(obj)}[{key}]"

        return to_code, inputs

    allow_inline_functions = [tracer.signature.python.builtins.isinstance, tracer.signature.python.builtins.tuple, tracer.signature.python.builtins.list]

    variableid_to_constant = {}

    def _eval_app(origin):
        if isinstance(origin, tracer.signature.python.Call):
            # ################## __call__ ##################
            function = _get_expression_for(origin.function)
            args = [_get_expression_for(i) for i in origin.args]
            kwargs = {k: _get_expression_for(v) for k, v in origin.kwargs.items()}

            def to_code(value_to_code):
                args_code = [value_to_code(i) for i in args] + [f"{k}={value_to_code(v)}" for k, v in kwargs.items()]
                return f"{value_to_code(function)}({', '.join(args_code)})"

            code.define(
                origin.output,
                Inlined(to_code, inputs=[function] + args + list(kwargs.values()), block=code.get_block_for(origin.output)),
                no_inline=not any(origin.function == f for f in allow_inline_functions),
            )

        elif isinstance(origin, tracer.signature.python.CallInplace):
            # ################## __call__ inplace ##################
            xs = _get_expression_for(origin.xs)
            function = _get_expression_for(origin.function)
            args = [_get_expression_for(i) for i in origin.args]
            kwargs = {k: _get_expression_for(v) for k, v in origin.kwargs.items()}

            def to_code(value_to_code):
                args_code = [value_to_code(i) for i in args] + [f"{k}={value_to_code(v)}" for k, v in kwargs.items()]
                return f"{value_to_code(function)}({', '.join(args_code)})"

            block = code.get_block_for(origin.output)
            block.append(Statement(to_code, inputs=[xs, function] + args + list(kwargs.values()), output_variables=[], block=block))

            def to_code(value_to_code):
                return value_to_code(xs)

            code.define(origin.output, Inlined(to_code, inputs=[xs], block=block), force_inline=True)

        elif isinstance(origin, tracer.signature.python.GetAttr):
            # ################## __getattr__ ##################
            obj = _get_expression_for(origin.obj)
            key = _get_expression_for(origin.key)
            if isinstance(key, Literal):

                def to_code(value_to_code):
                    assert key.code.startswith('"') and key.code.endswith('"')
                    return f"{value_to_code(obj)}.{key.code[1:-1]}"

            else:

                def to_code(value_to_code):
                    return f"getattr({value_to_code(obj)}, {value_to_code(key)})"

            code.define(origin.output, Inlined(to_code, inputs=[obj, key], block=code.get_block_for(origin.output)))

        elif isinstance(origin, tracer.signature.python.GetItem):
            # ################## __getitem__ ##################
            to_code, inputs = _at(origin.obj, origin.key)
            code.define(origin.output, Inlined(to_code, inputs=inputs, block=code.get_block_for(origin.output)))

        elif isinstance(origin, tracer.signature.python.UpdateItem):
            # ################## __setitem__ __additem__ ... ##################
            at_to_code, inputs = _at(origin.obj, origin.key)
            updates = _get_expression_for(origin.value)
            obj = _get_expression_for(origin.obj)

            def to_code(value_to_code):
                return f"{at_to_code(value_to_code)} {origin.op} {value_to_code(updates)}"

            block = code.get_block_for(origin.output)
            block.append(Statement(to_code, inputs=inputs + [updates, obj], output_variables=[], block=block))

            def to_code(value_to_code):
                return value_to_code(obj)

            code.define(origin.output, Inlined(to_code, inputs=[obj], block=block), force_inline=True)

            # TODO: should invalidate everything that comes before, including stuff before reshape!

        elif isinstance(origin, tracer.signature.python.Import):
            # ################## import ##################
            variable = code.add_variable_for(origin.output, allow_reusing_name=False)

            def to_code(value_to_code):
                if origin.from_ is None:
                    code = f"import {origin.import_}"
                else:
                    code = f"from {origin.from_} import {origin.import_}"
                name = value_to_code(variable)
                if name != origin.import_:
                    code += f" as {name}"
                return code

            if origin.as_ is not None:
                name_hints[id(variable)].append(origin.as_)
            elif "." not in origin.import_:
                name_hints[id(variable)].append(origin.import_)

            block = code.get_block_for(origin.output)
            block.prepend_after_comments(Statement(to_code, inputs=[], output_variables=[variable], block=block))

        elif isinstance(origin, tracer.signature.python.OperatorApplication):
            # ################## operator ##################
            operands = [_get_expression_for(i) for i in origin.operands]
            if len(operands) == 1:
                to_code = lambda value_to_code: f"{origin.operator}({value_to_code(operands[0])})"
            elif len(operands) == 2:
                to_code = lambda value_to_code: f"({value_to_code(operands[0])} {origin.operator} {value_to_code(operands[1])})"
            else:
                raise NotImplementedError(f"Don't know how to handle operator {origin.operator} with {len(operands)} operands.")

            code.define(origin.output, Inlined(to_code, inputs=operands, block=code.get_block_for(origin.output)))

        elif isinstance(origin, tracer.signature.python.Assert):
            # ################## assert ##################
            xs = _get_expression_for(origin.xs)
            condition = _get_expression_for(origin.condition)

            message = origin.message
            if message is not None:

                def to_code(value_to_code):
                    return f'assert {value_to_code(condition)}, "{message}"'

            else:

                def to_code(value_to_code):
                    return f"assert {value_to_code(condition)}"

            block = code.get_block_for(origin.output)
            block.append(Statement(to_code, inputs=[xs, condition], output_variables=[], block=block))

            def to_code(value_to_code):
                return value_to_code(xs)

            code.define(origin.output, Inlined(to_code, inputs=[xs], block=code.get_block_for(origin.output)), force_inline=True)

        elif isinstance(origin, tracer.signature.python.Builtin):
            # ################## builtin ##################
            name = origin.name
            to_code = lambda value_to_code: name  # TODO: check if name is in scope. Option 1: prevent this. Option 2: import builtins
            code.define(origin.output, Inlined(to_code, inputs=[], block=code.get_block_for(origin.output)))

        elif isinstance(origin, tracer.Cast):
            # ################## tracer.cast ##################
            input = _get_expression_for(origin.input)
            to_code = lambda value_to_code: value_to_code(input)
            code.define(origin.output, Inlined(to_code, inputs=[input], block=code.get_block_for(origin.output)), force_inline=True)

        elif isinstance(origin, tracer.signature.python.Constant):
            # ################## tracer.constant ##################
            variable = code.add_variable_for(origin.output, allow_reusing_name=False)
            assert id(variable) not in variableid_to_constant
            variableid_to_constant[id(variable)] = origin.value

            name_hints[id(variable)].append(f"const{len(variableid_to_constant)}")
            value_str = str(origin.value).replace("\n", " ")
            code.root_block.prepend(comment_statement(f"Constant const{len(variableid_to_constant)}: {value_str}", code.root_block))

        else:
            raise NotImplementedError(f"Don't know how to handle application {origin} with type {type(origin)}.")

    def _get_expression_for2(x):
        if x in code:
            return code[x]
        elif isinstance(x, tracer.Tracer):
            # ################## Tracer ##################
            assert x.origin is not None, f"Got {x} of type {type(x)} with no origin."
            _eval_app(x.origin)
            return code[x]
        elif isinstance(x, str):
            # ################## str ##################
            return Literal(f'"{x}"', block=code.root_block)
        elif isinstance(x, int | float | np.integer | np.floating | bool):
            # ################## Numeric ##################
            return Literal(str(x), block=code.root_block)
        elif x is None:
            # ################## None ##################
            return Literal("None", block=code.root_block)
        elif isinstance(x, list):
            # ################## list ##################
            values = [_get_expression_for(i) for i in x]
            if x in code:
                # Previous applications already resulted in an expression for the entire list
                return code[x]

            def to_code(value_to_code):
                return f"[{', '.join([value_to_code(i) for i in values])}]"

            return Inlined(to_code, inputs=values, block=code.get_block_for(x))
        elif isinstance(x, tuple):
            # ################## tuple ##################
            values = [_get_expression_for(i) for i in x]
            if x in code:
                # Previous applications already resulted in an expression for the entire tuple
                return code[x]
            comma = "," if len(values) == 1 else ""

            def to_code(value_to_code):
                return f"({', '.join([value_to_code(i) for i in values])}{comma})"

            return Inlined(to_code, inputs=values, block=code.get_block_for(x))
        elif isinstance(x, dict):
            # ################## dict ##################
            keys = [_get_expression_for(k) for k in x.keys()]
            values = [_get_expression_for(v) for v in x.values()]
            if x in code:
                # Previous applications already resulted in an expression for the entire dict
                return code[x]

            def to_code(value_to_code):
                return f"{{{', '.join([f'{value_to_code(k)}: {value_to_code(v)}' for k, v in zip(keys, values, strict=False)])}}}"

            return Inlined(to_code, inputs=keys + values, block=code.get_block_for(x))
        elif isinstance(x, tracer.Graph):
            # ################## def function(...): ##################
            graph = x
            outer_block = code.get_block_for(graph)
            inner_block = code.get_block_for(graph.output)

            output_variable = code.add_variable_for(graph)
            if graph.name is not None:
                name_hints[id(output_variable)].append(graph.name)

            # Graph input
            parameter_variables = [code.add_variable_for(input) for input in graph.inputs]

            # Graph output
            return_value = _get_expression_for(graph.output)
            to_code = lambda value_to_code: f"return {value_to_code(return_value)}"
            inner_block.append(Statement(to_code, inputs=[return_value], output_variables=[], block=inner_block))

            def to_code(value_to_code):
                function_name = value_to_code(output_variable)
                parameter_variable_names = [value_to_code(i) for i in parameter_variables]
                lines = [f"def {function_name}({', '.join(parameter_variable_names)}):"]
                for line in inner_block.to_code(value_to_code):
                    lines.append(f"    {line}")
                return lines

            outer_block.append(Statement(to_code, inputs=[], output_variables=[output_variable], block=outer_block))

            return output_variable
        else:
            raise NotImplementedError(f"Don't know how to handle object {x} with type {type(x)}.")

    currently_evaluating_ids = set()

    def _get_expression_for(x):
        assert id(x) not in currently_evaluating_ids
        currently_evaluating_ids.add(id(x))
        result = _get_expression_for2(x)
        currently_evaluating_ids.remove(id(x))
        return result

    object_expression = _get_expression_for(object)

    variables = code.variables
    statements = [statement for block in code.blocks for statement in block.statements]

    # Gather dependents per variable
    variableid_to_dependentstatements = {id(var): [] for var in variables}
    for statement in statements:
        for input_variable in statement.input_variables:
            variableid_to_dependentstatements[id(input_variable)].append(statement)

    # Find groups of variables that will be assigned the same name
    variableid_to_group = {id(var): [var] for var in variables}

    def fuse(var1, var2):
        varid1 = id(var1)
        varid2 = id(var2)
        group1 = variableid_to_group[varid1]
        group2 = variableid_to_group[varid2]
        if group1 is group2:
            return
        for var in group2:
            variableid_to_group[id(var)] = group1
            group1.append(var)

    for block in code.blocks:
        seen_statement_ids = set()
        for statement in block.statements:
            seen_statement_ids.add(id(statement))

            output_variables = {v for v in statement.output_variables if v.allow_reusing_name}
            if len(output_variables) == 1:
                output_variable = output_variables.pop()

                input_variables = statement.input_variables

                # Consider only input variables that allow reusing their name
                input_variables = {v for v in input_variables if v.allow_reusing_name}

                # Consider only input variables that are not used in any other block
                input_variables = {v for v in input_variables if all(v.block == statement.block for statement in variableid_to_dependentstatements[id(v)])}

                # Consider only input variables that are not used in any later statement
                input_variables = {
                    v for v in input_variables if all(id(statement) in seen_statement_ids for statement in variableid_to_dependentstatements[id(v)])
                }

                if len(input_variables) == 1:
                    input_variable = input_variables.pop()
                    if id(input_variable.block) == id(output_variable.block):
                        fuse(input_variable, output_variable)

    groups = []
    group_ids = set()
    for group in variableid_to_group.values():
        if id(group) not in group_ids:
            groups.append(group)
            group_ids.add(id(group))

    # Assign names to variables
    variableid_to_name = {}

    def names():
        chars = [chr(i) for i in range(ord("a"), ord("z") + 1)]
        length = 1
        while True:
            for name in itertools.product(chars, repeat=length):
                yield "".join(name)
            length += 1

    names = names()
    for group in groups:
        group_name_hints = [name_hints[id(var)] for var in group if id(var) in name_hints]
        group_name_hints = [name for names in group_name_hints for name in names]
        if len(group_name_hints) != 1:
            name = next(names)
        else:
            name = group_name_hints[0]
        for var in group:
            assert id(var) not in variableid_to_name
            variableid_to_name[id(var)] = name

    # Generate dictionary of constants
    name_to_constant = {variableid_to_name[varid]: constant for varid, constant in variableid_to_constant.items()}

    # Generate code
    def value_to_code(x):
        if isinstance(x, Variable):
            return variableid_to_name[id(x)]
        elif isinstance(x, Literal):
            return x.code
        elif isinstance(x, Inlined):
            return x.to_code(value_to_code)
        else:
            raise ValueError(f"Got {x} of type {type(x)}.")

    exec_code = code.to_code(value_to_code)
    exec_code = "\n".join(exec_code)  # Used to run the code
    eval_code = value_to_code(object_expression)  # Used to retrieve the result from the locals_globals dictionary

    locals_globals = {**name_to_constant}
    try:
        exec(exec_code, locals_globals, locals_globals)
        compiled_object = eval(eval_code, locals_globals, locals_globals)
    except Exception as e:
        message = "The code that was created for this operation failed to compile. The following code was generated:\n"
        for i, line in enumerate(exec_code.splitlines(), 1):
            message += f"{i:4}: {line}\n"
        message += "\nThe error was: "
        if hasattr(type(e), "__name__"):
            message += f"{type(e).__name__}"
        if len(str(e)) > 0:
            message += f": {e}"
        raise Exception(message) from e

    if return_code:
        return compiled_object, exec_code
    else:
        return compiled_object

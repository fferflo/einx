import einx
from .tracer import *
from .tensor import *
from .optimize import optimize
from functools import partial


class Variables:
    def __init__(self, parent=None):
        self.variables = {}
        self.parent = parent
        self.children = []
        if not self.parent is None:
            self.parent.children.append(self)

    def fork(self):
        return Variables(parent=self)

    def __contains__(self, name):
        return name in self.variables or (not self.parent is None and name in self.parent)

    def _is_free(self, name):
        if name in self.variables:
            return False
        if not self.parent is None and name in self.parent:
            return False
        for child in self.children:
            if name in child:
                return False
        return True

    def add(self, value, prefix=None, name=None):
        if prefix is None:
            assert name is not None
            if name in self:
                raise ValueError(f"Variable name '{name}' already exists")
            self.variables[name] = value
            return name
        else:
            assert name is None
            i = 0
            while not self._is_free(name := f"{prefix}{i}"):
                i += 1
            self.variables[name] = value
            return name

    def __getitem__(self, name):
        if name in self.variables:
            return self.variables[name]
        if not self.parent is None:
            return self.parent[name]
        raise ValueError(f"Variable '{name}' is not set")


class Block:
    def __init__(self, variables, parent):
        self.variables = variables
        self.parent = parent
        self.code = []  # lines and blocks

    def is_parent_of(self, other):
        return other.parent is self or (
            not other.parent is None and self.is_parent_of(other.parent)
        )

    @property
    def root_block(self):
        block = self
        while block.parent is not None:
            block = block.parent
        return block

    def is_root(self):
        return self.parent is None

    def get_lines_of_code(self):
        lines = []
        for child in self.code:
            if isinstance(child, str):
                lines.append(child)
            else:
                assert isinstance(child, Block)
                lines.extend(
                    f"    {line}" for line in child.get_lines_of_code()
                )  # TODO: param for indentation

        return lines

    def __str__(self):
        return "\n".join(self.get_lines_of_code())


def _remove_parentheses(s):
    assert isinstance(s, str)
    level = 0
    can_remove_parentheses = s[0] == "(" and s[-1] == ")"
    if can_remove_parentheses:
        for i, c in enumerate(s):
            if c == "(":
                level += 1
            elif c == ")":
                level -= 1
                if level == 0:
                    can_remove_parentheses = i == len(s) - 1
                    break
    if can_remove_parentheses:
        s = s[1:-1]
    return s


class Definition:
    def __init__(self, value, block, code):
        self._value = value
        self._block = block
        self._code = code
        self.overwritten = False

    @property
    def value(self):
        if self.overwritten:
            raise ValueError("Trying to access overwritten definition")
        return self._value

    @property
    def block(self):
        if self.overwritten:
            raise ValueError("Trying to access overwritten definition")
        return self._block

    @property
    def code(self):
        if self.overwritten:
            raise ValueError("Trying to access overwritten definition")
        return self._code

    @property
    def name(self):
        if self.overwritten:
            raise ValueError("Trying to access overwritten definition")
        if not self.is_variable():
            raise ValueError("Trying to access name of non-variable definition")
        return self._code

    def is_variable(self):
        if self.overwritten:
            raise ValueError("Trying to access overwritten definition")
        return not self._code is None and self._code.isidentifier()

    def is_pytree(self):
        if self.overwritten:
            raise ValueError("Trying to access overwritten definition")
        return isinstance(self.value, (tuple, list, dict))

    def is_overwritten(self):
        return self.overwritten

    def overwrite(self, new_value):
        if self.overwritten:
            raise ValueError("Trying to overwrite definition twice")
        if not self.is_variable():
            raise ValueError("Trying to overwrite non-variable definition")
        self.overwritten = True
        return Definition(new_value, self._block, code=self._code)


class CodeObject:
    def __init__(self, objects):
        self.root_block = Block(variables=Variables(), parent=None)
        self.definitions = {}  # obj-id: Definition
        self.constants = []
        self.usages = Usages(objects)
        self.names = einx.tree_util.tree_map(lambda x: self.get_definition_of(x).name, objects)
        self.code = str(self.root_block)

        for definition in self.constants:
            line = f"# {definition.name}: {str(type(definition.value))}"
            value_str = str(definition.value)
            if not "\n" in value_str:
                line += f" = {value_str}"
            self.code = line + "\n" + self.code

        locals_globals = {definition.name: definition.value for definition in self.constants}
        exec(self.code, locals_globals, locals_globals)
        self.output = einx.tree_util.tree_map(lambda name: locals_globals[name], self.names)

    def __str__(self):
        return self.code

    def join_blocks(self, blocks):
        blocks = list(blocks)
        if len(blocks) == 0:
            return self.root_block
        block = blocks[0]
        for block2 in blocks[1:]:
            if id(block) == id(block2):
                pass
            elif block.is_parent_of(block2):
                block = block2
            elif block2.is_parent_of(block):
                pass
            else:
                raise ValueError("Cannot join blocks")
        return block

    def execute_application(self, application):
        assert isinstance(application, Application)

        comment = f"  # {application.comment}" if not application.comment is None else ""

        # Find block at which to execute the application (i.e. where all dependencies are defined)
        in_defs = [self.get_definition_of(x) for x in application.dependencies]
        block = self.join_blocks([d.block for d in in_defs])

        use_dynamic_output_check = False
        if isinstance(application.op, Import):
            import_str = f"import {application.op.import_}"
            name = application.op.import_
            if not application.op.as_ is None:
                import_str = f"{import_str} as {application.op.as_}"
                name = application.op.as_
            if not application.op.from_ is None:
                import_str = f"from {application.op.from_} {import_str}"

            # Import only once
            if not any(
                isinstance(line, str)
                and (line == import_str or line.startswith(import_str + "  #"))
                for line in block.code
            ):
                # First import
                block.code.insert(0, import_str + comment)
                self.new_value_definition(application.output, block, name)
            else:
                # Subsequent import: Reuse existing definition
                self.definitions[id(application.output)] = self.definitions[
                    id(block.variables[name])
                ]
            return

        inline = None
        if isinstance(application.op, MemberAccess):
            inline = True  # Always inline
            obj = self.get_definition_of(application.args[0]).code
            member = application.args[1]
            right_str = f"{obj}.{member}"
        elif isinstance(application.op, Operator):
            if len(application.args) == 1:
                op = application.op.op
                arg = self.get_definition_of(application.args[0]).code
                right_str = f"({op}{arg})"
            elif len(application.args) == 2:
                op = application.op.op
                arg0 = self.get_definition_of(application.args[0]).code
                arg1 = self.get_definition_of(application.args[1]).code
                right_str = f"({arg0} {op} {arg1})"
            else:
                raise ValueError(f"Invalid number of arguments for operator '{application.op.op}'")
        elif isinstance(application.op, AssignAt):
            obj = self.get_definition_of(application.args[0]).code
            key = self.get_definition_of(application.args[1]).code
            op = application.op.op
            update = self.get_definition_of(application.args[2]).code
            right_str = f"({obj}[{_remove_parentheses(key)}] {op} {_remove_parentheses(update)})"
        elif isinstance(application.op, GetAt):
            obj = self.get_definition_of(application.args[0]).code

            slices = application.args[1]
            if not isinstance(slices, tuple):
                slices = (slices,)
            assert isinstance(slices, tuple)
            assert len(slices) > 0

            def slice_to_str(s):
                if isinstance(s, slice):
                    x = ""
                    if s.start is not None:
                        x += str(s.start)
                    x += ":"
                    if s.stop is not None:
                        x += str(s.stop)
                    if s.step is not None:
                        x += ":" + str(s.step)
                    return x
                else:
                    return _remove_parentheses(self.get_definition_of(s).code)

            slices = ", ".join(slice_to_str(s) for s in slices)

            right_str = f"{obj}[{slices}]"
        else:
            op = self.get_definition_of(application.op).code
            args = [self.get_definition_of(arg).code for arg in application.args] + [
                f"{k}={self.get_definition_of(v).code}" for k, v in application.kwargs.items()
            ]
            args = f"{', '.join(args)}"
            right_str = f"{op}({args})"
            use_dynamic_output_check = not isinstance(application.op, Tracer)

        inplace = len(application.inplace_updates) > 0

        if inline is None:
            # Otherwise: inline if the application string is short and output is used only once
            inline = (
                not use_dynamic_output_check
                and not inplace
                and len(right_str) < 20  # TODO: add parameter
                and len(self.usages.get(application.output)) == 1
            )

        if inline:
            assert not use_dynamic_output_check
            self.new_value_definition(application.output, block, right_str)
        else:
            if isinstance(application.output, (tuple, list)) and all(
                isinstance(x, Tracer) for x in application.output
            ):
                # Output: Unwrap list or tuple of tracers
                assert not inplace
                output_defs = [
                    self.new_variable_definition(x, block, prefix="x") for x in application.output
                ]
                left_str = " ".join([d.name + "," for d in output_defs])
                block.code.append(f"{left_str} = {_remove_parentheses(right_str)}" + comment)
            elif inplace:
                # Output: Same existing variable
                for tensor_in, tensor_out in application.inplace_updates:
                    in_definition = self.get_definition_of(tensor_in)
                    usages = self.usages.get(tensor_in)
                    assert (
                        in_definition.is_variable()  # Must be a variable
                        and in_definition.block is block  # Must be in the same block
                        and len(usages) == 1  # Must be used exactly once
                    )
                    self.overwrite_variable_definition(in_definition, tensor_out)
                block.code.append(_remove_parentheses(right_str) + comment)
            else:
                # Output: Single new variable for pytree of tracers
                left_str = self.new_variable_definition(application.output, block, prefix="x").name
                block.code.append(f"{left_str} = {_remove_parentheses(right_str)}" + comment)

        if use_dynamic_output_check:

            def check(output):
                definition = self.get_definition_of(output)
                if isinstance(definition.value, Tensor):
                    line = f"assert {definition.code}.shape == {self.get_definition_of(output.shape).code}"
                    block.code.append(line)

            einx.tree_util.tree_map(check, application.output)

    def _add_definition(self, definition):
        if id(definition.value) in self.definitions:
            raise ValueError(f"Trying to add definition for existing value")
        self.definitions[id(definition.value)] = definition

        # If value is a pytree, add definition for all leaves
        def store(x, key):
            if len(key) > 0:
                code = definition.code
                for k in key:
                    if isinstance(k, int):
                        code += f"[{k}]"
                    elif isinstance(k, str):
                        code += f'["{k}"]'
                    else:
                        assert False
                self.new_value_definition(x, definition.block, code)

        einx.tree_util.tree_map_with_key(store, definition.value)

    def new_variable_definition(self, value, block, *args, **kwargs):
        name = block.variables.add(value, *args, **kwargs)
        definition = Definition(value, block, name)
        self._add_definition(definition)
        return definition

    def new_value_definition(self, value, block, code):
        definition = Definition(value, block, code)
        self._add_definition(definition)
        if definition.is_variable():
            definition.block.variables.add(value, name=definition.name)
        return definition

    def new_empty_definition(self, value, block):
        definition = Definition(value, block, "!!!")  # This should never appear in the final code
        self._add_definition(definition)
        return definition

    def overwrite_variable_definition(self, old_definition, new_value):
        assert old_definition.is_variable() and not old_definition.is_pytree()
        self.definitions[id(new_value)] = old_definition.overwrite(new_value)

    def get_definition_of(self, x):
        if id(x) in self.definitions:
            definition = self.definitions[id(x)]
            if definition.is_overwritten():
                raise ValueError(f"Trying to access overwritten variable")
            return definition

        if isinstance(x, TracableFunction):
            if x.args is None:
                raise ValueError("Cannot define a function without args and/or kwargs")

            # TODO: assert that function has no sideeffects
            block = self.root_block

            if x.name is not None:
                definition = self.new_variable_definition(x, block, name=x.name)
            else:
                definition = self.new_variable_definition(x, block, prefix="op")

            function_block = Block(variables=block.variables.fork(), parent=block)

            # Define parameters
            arg_defs = [
                self.new_variable_definition(arg, function_block, prefix="i") for arg in x.args
            ]  # TODO: not using kwargs
            virtual_arg_defs = [
                self.new_empty_definition(virtual_arg, function_block)
                for virtual_arg in x.virtual_args
            ]
            argnames = [d.name for d in arg_defs]

            # Define function body
            output_def = self.get_definition_of(x.output)

            block.code.append(f"def {definition.name}({', '.join(argnames)}):")
            block.code.append(function_block)
            block.code.append(f"    return {output_def.code}")

            return definition

        elif isinstance(x, Tracer):
            if x.origin == "constant":
                return Definition(x, self.root_block, None)
            elif x.origin is None:
                raise ValueError(
                    f"Got a tracer without an origin and a concrete value with type {type(x)}"
                )
            elif isinstance(x.origin, Application):
                self.execute_application(x.origin)
                assert id(x) in self.definitions
                return self.definitions[id(x)]
            else:
                assert False, f"{type(x.origin)}"
        elif isinstance(x, str):
            return Definition(x, self.root_block, f'"{x}"')
        elif isinstance(x, tuple):
            x_defs = [self.get_definition_of(a) for a in x]
            code = "(" + ", ".join([d.code for d in x_defs]) + ("," if len(x) == 1 else "") + ")"
            return Definition(x, self.join_blocks([d.block for d in x_defs]), code)
        elif isinstance(x, list):
            x_defs = [self.get_definition_of(a) for a in x]
            code = "[" + ", ".join([d.code for d in x_defs]) + "]"
            return Definition(x, self.join_blocks([d.block for d in x_defs]), code)
        elif isinstance(x, dict):
            x_defs = {k: self.get_definition_of(v) for k, v in x.items()}
            code = "{" + ", ".join(f'"{k}": {v.code}' for k, v in x_defs.items()) + "}"
            return Definition(x, self.join_blocks([v.block for v in x_defs.values()]), code)
        elif isinstance(x, (int, float, np.integer, np.floating)):
            return Definition(x, self.root_block, str(x))
        elif isinstance(x, slice):
            if x.step is not None:
                code = f"slice({self.get_definition_of(x.start).code}, {self.get_definition_of(x.stop).code}, {self.get_definition_of(x.step).code})"
            elif x.stop is not None:
                code = f"slice({self.get_definition_of(x.start).code}, {self.get_definition_of(x.stop).code})"
            else:
                code = f"slice({self.get_definition_of(x.start).code})"
            return Definition(x, self.root_block, code)
        elif x is None:
            return Definition(x, self.root_block, "None")
        else:
            # Constant
            definition = self.new_variable_definition(x, self.root_block, prefix="const")
            self.constants.append(definition)
            return definition


class CompiledFunction:
    def __init__(self, function):
        function = optimize(function)

        code_object = CodeObject(function)
        self.code = str(code_object)
        self.op = code_object.output

    def __call__(self, *input_concrete):
        # TODO: assert that input_concrete are compatible with function.input?
        return self.op(*input_concrete)

    def __str__(self):
        return self.code

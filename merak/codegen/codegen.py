from ..parser import AST as ast
from . import huffIR as ir

# from ..analyzers.symbolTable import SymbolTable, SymbolVariable, SymbolConstant
from collections import deque


class CodeGenerator:
    def __init__(self) -> None:
        # self.symbolTable = SymbolTable()
        self.functions = dict()
        self.args = dict()
        self.actual_function = None
        self.stack = deque()

    def run(self, tree):
        return self.walkTree(tree)

    def walkTree(self, node):
        if node is None:
            return None

        if isinstance(node, ast.Definition):
            # Definition(contract, rest) -> None
            contract = self.walkTree(node.contract)
            rest = self.walkTree(node.rest)
            if rest is not None:
                raise NotImplementedError(
                    "More than one definition, not implemented yet"
                )
                rest.append(contract)
                return rest
            else:
                return [contract]
        elif isinstance(node, ast.Contract):
            # Contract(id, definition) -> None
            nodef = self.walkTree(node.definition)
            storage = []
            functions = []
            interfaces = []
            for definition in nodef:
                if isinstance(definition, ir.StorageSymbol):
                    storage.append(definition)
                elif isinstance(definition, ir.Function):
                    functions.append(definition)
                elif isinstance(definition, ir.Interface):
                    interfaces.append(definition)
            return ir.Contract(functions, storage, interfaces)
        elif isinstance(node, ast.GlobalVariablesDeclaration):
            # GlobalVariablesDeclaration(variable, rest) -> None
            var = self.walkTree(node.variable)
            rest = self.walkTree(node.rest)
            if rest is not None:
                rest.append(var)
                return rest
            else:
                return [var]
        elif isinstance(node, ast.GlobalVariable):
            # GlobalVariable(id, type) -> None
            # var = SymbolVariable(node.id, node.type, None, static=True)
            # elf.symbolTable.define(var)
            return ir.StorageSymbol(node)
        elif isinstance(node, ast.GlobalConstantsDeclaration):
            # GlobalConstantsDeclaration(constant, rest) -> None
            const = self.walkTree(node.constant)
            rest = self.walkTree(node.rest)
            if rest is not None:
                rest.append(const)
                return rest
            else:
                return [const]
        elif isinstance(node, ast.GlobalConstant):
            # GlobalConstant(id, type, value) -> None
            # const = SymbolVariable(node.id, node.type, node.value, static=True)
            # self.symbolTable.define(const)
            return ir.StorageSymbol(node)
        elif isinstance(node, ast.Functions):
            # Functions(function, rest) -> None
            func = self.walkTree(node.function)
            rest = self.walkTree(node.rest)
            if rest is not None:
                rest.append(func)
                return rest
            else:
                return [func]
        elif isinstance(node, ast.FunctionDefinition):
            # FunctionDefinition(id, args, returns, body) -> None
            id = node.id
            self.actual_function = id
            if id in self.functions:
                raise Exception("Function already defined (args)")

            args = self.walkTree(node.args)
            # add to self.args[id] to
            # identify when used inside CODE
            self.args[id] = args

            returns = self.walkTree(node.returns)
            code = self.walkTree(node.body)
            func = ir.Function(id, args, returns, code)
            self.functions[id] = func
            self.actual_function = None
            return func
        elif isinstance(node, ast.FunctionArgs):
            # FunctionArgs(id, type, rest) -> None
            rest = self.walkTree(node.rest)

            # trasformed into dict to easily "id in args.keys()"
            arg = {node.id: node.type}

            if rest is not None:
                rest = {**arg, **rest}
                return rest
            else:
                return arg
        elif isinstance(node, ast.FunctionReturn):
            # FunctionReturn(type, rest) -> None
            rest = self.walkTree(node.rest)
            if rest is not None:
                rest.append(node.type)
                return rest
            else:
                return [node.type]
        elif isinstance(node, ast.FunctionTypes):
            # FunctionTypes(type, rest) -> None
            rest = self.walkTree(node.rest)
            if rest is not None:
                rest.append(node.type)
                return rest
            else:
                return [node.type]
        elif isinstance(node, ast.BinaryOperation):
            # BinaryOperation(op, left, right) -> None
            lnode = self.walkTree(node.left)
            rnode = self.walkTree(node.right)
            ops = [lnode, rnode, node.op]
            return ops
        elif isinstance(node, ast.GroupExpression):
            # GroupExpression(expression) -> None
            return self.walkTree(node.expression)
        elif isinstance(node, ast.ReturnCode):
            # ReturnCode(expression) -> None
            ret = self.walkTree(node.expression)
            if self.is_value(ret):
                # 1) value on top of the stack
                # 2) 0x00 mstore
                # 3) 0x20 0x00 return
                return 1
            else:
                # 1) Solve expression
                # 2) value on top of the stack
                # 3) 0x00 mstore
                # 4) 0x20 0x00 return
                return 0
        elif isinstance(node, ast.AssignCode):
            # AssignCode(id, type, expression) -> None
            if node.type is None:
                """
                CASE: overriding old variable
                if symbol not in symbol table
                    error, var not defined
                else
                    if type in SymbolTable != type node:
                        raise ERROR

                    if expression is value
                        check type value is equal type node
                        update value in symbolTable
                        and ?
                    else
                        solve expression ?
                        check type value is equal type node
                        update value in symbolTable
                        and ?
                    return ?
                """
                return 1
            else:
                """
                CASE: declaring new var
                if symbol in symbolTable
                    error, already defined
                else
                    if expression is value
                        check type value is equal type node
                        define new symbol in symbolTable
                        and ?
                    else
                        solve expression ?
                        check type value is equal type node
                        define new symbol in symbolTable
                        and ?
                    return ?
                """
                return 0
        elif isinstance(node, ast.NameExpression):
            # NameExpression(name) -> None
            # TODO ADD CHECKS TO IDS
            return node.name
        elif isinstance(node, ast.NumberExpression):
            # NumberExpression(number) -> None
            # TODO ADD CHECKS TO NUMBERS
            return node.number
        elif isinstance(node, ast.Empty):
            # Empty() -> None
            return None
        else:
            raise Exception("AST node not recognized: ", node)

    def is_value(self, exp):
        return True

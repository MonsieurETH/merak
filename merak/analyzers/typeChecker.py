from unicodedata import name
from ..parser import AST as ast


class TypeChecker:
    def __init__(self, env) -> None:
        self.scope = scope

    def check(self, node):
        if node is None:
            return None

        if isinstance(node, ast.Definition):
            # Definition(contract, rest) -> None
            self.check(node.contract)
        elif isinstance(node, ast.Contract):
            # Contract(id, definition) -> None
            self.check(node.definition)
        elif isinstance(node, ast.GlobalVariablesDeclaration):
            # GlobalVariablesDeclaration(variable, rest) -> None
            self.check(node.variable)
            self.check(node.rest)
        elif isinstance(node, ast.GlobalVariable):
            # GlobalVariable(id, type) -> None
            pass
        elif isinstance(node, ast.GlobalConstantsDeclaration):
            # GlobalConstantsDeclaration(constant, rest) -> None
            self.check(node.constant)
            self.check(node.rest)
        elif isinstance(node, ast.GlobalConstant):
            # GlobalConstant(id, type, value) -> None
            pass
        elif isinstance(node, ast.StructDeclaration):
            # StructDeclaration(struct, rest) -> None
            self.check(node.struct)
            self.check(node.rest)
        elif isinstance(node, ast.Struct):
            # Struct(name, vars) -> None
            pass
        elif isinstance(node, ast.StructVars):
            # StructVars(id, type, rest) -> None
            self.check(node.rest)
        elif isinstance(node, ast.Function):
            # Functions(function, rest) -> None
            self.check(node.function)
            self.check(node.rest)
        elif isinstance(node, ast.FunctionDefinition):
            # FunctionDefinition(id, args, returns, body) -> None
            self.check(node.body)
        elif isinstance(node, ast.FunctionArgs):
            # FunctionArgs(id, type, rest) -> None
            pass
        elif isinstance(node, ast.FunctionReturn):
            # FunctionReturn(type, rest) -> None
            pass
        elif isinstance(node, ast.FunctionTypes):
            # FunctionTypes(type, rest) -> None
            pass
        elif isinstance(node, ast.BinaryOperation):
            # BinaryOperation(op, left, right) -> None
            pass
        elif isinstance(node, ast.GroupExpression):
            # GroupExpression(expression) -> None
            pass
        elif isinstance(node, ast.ReturnCode):
            # ReturnCode(expression) -> None
            pass
        elif isinstance(node, ast.VarDefinition):
            # VarDefinition(id, type, expression, rest) -> None
            pass
        elif isinstance(node, ast.VarAssigment):
            # VarAssigment(id, expression, rest) -> None
            return
        elif isinstance(node, ast.NameExpression):
           # NameExpression(name) -> None
           return
        elif isinstance(node, ast.NumberExpression):
           # NumberExpression(number) -> None
           return
        elif isinstance(node, ast.Empty):
            # Empty() -> None
            return
        else:
            raise Exception("AST node not recognized: ", node)


class BuiltInType:
    pass

class WordType(BuiltInType):
    pass

class UWordType(BuiltInType):
    pass

class BoolType(BuiltInType):
    pass

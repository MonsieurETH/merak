from ..parser import AST as ast


class ScopeBuilder:
    def build(self, node, scope=None):
        if scope is None:
            gs = GlobalScope()
            return self.build(node, scope=gs)

        if node is None:
            return scope

        if isinstance(node, ast.Definition):
            # Definition(contract, rest) -> None
            scope = self.build(node.contract, scope=scope)
        elif isinstance(node, ast.Contract):
            # Contract(id, definition) -> None
            scope = self.build(node.definition, scope=scope)
        elif isinstance(node, ast.GlobalVariablesDeclaration):
            # GlobalVariablesDeclaration(variable, rest) -> None
            scope = self.build(node.variable, scope=scope)
            scope = self.build(node.rest, scope=scope)
        elif isinstance(node, ast.GlobalVariable):
            # GlobalVariable(id, type) -> None
            sv = SymbolVariable(node.id, node.type, None, scope)
            scope.addSymbol(sv)
        elif isinstance(node, ast.GlobalConstantsDeclaration):
            # GlobalConstantsDeclaration(constant, rest) -> None
            scope = self.build(node.constant, scope=scope)
            scope = self.build(node.rest, scope=scope)
        elif isinstance(node, ast.GlobalConstant):
            # GlobalConstant(id, type, value) -> None
            sc = SymbolConstant(node.id, node.type, node.value, scope)
            scope.addSymbol(sc)
        elif isinstance(node, ast.StructDeclaration):
            # StructDeclaration(struct, rest) -> None
            scopeStruct = self.build(node.struct, scope=scope)
            scope.addChild(scopeStruct)
            scope = self.build(node.rest, scope=scope)
        elif isinstance(node, ast.Struct):
            # Struct(name, vars) -> None
            sc = StructScope(node.name, parent=scope)
            scope = self.build(node.vars, scope=sc)
        elif isinstance(node, ast.StructVars):
            # StructVars(id, type, rest) -> None
            sv = SymbolVariable(node.id, node.type, None, scope)
            scope.addSymbol(sv)
            scope = self.build(node.rest, scope=scope)
        elif isinstance(node, ast.Function):
            # Functions(function, rest) -> None
            scopefn = self.build(node.function, scope=scope)
            scope = self.build(node.rest, scope=scopefn)
        elif isinstance(node, ast.FunctionDefinition):
            # FunctionDefinition(id, args, returns, body) -> None
            fs = FunctionScope(node.id, parent=scope)
            fs = self.build(node.args, scope=fs)
            ls = LocalScope(parent=fs)
            ls = self.build(node.body, scope=ls)
            fs.addChild(ls)
            scope.addChild(fs)
        elif isinstance(node, ast.FunctionArgs):
            # FunctionArgs(id, type, rest) -> None
            fa = SymbolArgument(node.id, node.type, scope)
            scope.addSymbol(fa)
            scope = self.build(node.rest, scope=scope)
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
            #self.build(node.expression)
        elif isinstance(node, ast.VarDefinition):
            # VarDefinition(id, type, expression, rest) -> None
            sv = SymbolVariable(
                node.id, node.type, node.expression, scope
            )
            scope.addSymbol(sv)
            scope = self.build(node.rest, scope=scope)
        elif isinstance(node, ast.VarAssigment):
            # VarAssigment(id, expression, rest) -> None
            scope = self.build(node.rest, scope=scope)
        elif isinstance(node, ast.NameExpression):
            # NameExpression(name) -> None
            pass
        elif isinstance(node, ast.NumberExpression):
            # NumberExpression(number) -> None
            pass
        elif isinstance(node, ast.Empty):
            # Empty() -> None
            pass
        else:
            raise Exception("AST node not recognized: ", node)

        return scope


class Scope:
    def __init__(self, parent=None) -> None:
        self.parent = parent
        self.childs = []
        self.symbols = []

    def addSymbol(self, symbol):
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        else:
            raise Exception('Symbol already defined:', symbol )

    def addChild(self, child):
        self.childs.append(child)


class GlobalScope(Scope):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

class LocalScope(Scope):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)


class FunctionScope(Scope):
    def __init__(self, name, parent=None) -> None:
        super().__init__(parent)
        self.name = name


class StructScope(Scope):
    def __init__(self, name, parent=None) -> None:
        super().__init__(parent)
        self.name = name


class Symbol:
    def __init__(self, name, type, value, scope) -> None:
        self.name = name
        self.type = type
        self.value = value
        self.scope = scope

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self.name == other.name
        return False


class SymbolVariable(Symbol):
    def __init__(self, name, type, value, scope) -> None:
        super().__init__(name, type, value, scope)


class SymbolConstant(Symbol):
    def __init__(self, name, type, value, scope) -> None:
        super().__init__(name, type, value, scope)

class SymbolArgument(Symbol):
    def __init__(self, name, type, scope) -> None:
        super().__init__(name, type, None, scope)

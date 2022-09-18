from ..parser import AST as ast


class ScopeBuilder:
    def __init__(self) -> None:
        gs = GlobalScope()
        self.scopeTree = gs
        self.lastScope = gs

    def build(self, node):
        if node is None:
            return None

        if isinstance(node, ast.Definition):
            # Definition(contract, rest) -> None
            self.build(node.contract)
            return self.scopeTree
        elif isinstance(node, ast.Contract):
            # Contract(id, definition) -> None
            self.build(node.definition)
        elif isinstance(node, ast.GlobalVariablesDeclaration):
            # GlobalVariablesDeclaration(variable, rest) -> None
            self.build(node.variable)
            self.build(node.rest)
        elif isinstance(node, ast.GlobalVariable):
            # GlobalVariable(id, type) -> None
            sv = SymbolVariable(node.id, node.type, None)
            self.lastScope.addSymbol(sv)
        elif isinstance(node, ast.GlobalConstantsDeclaration):
            # GlobalConstantsDeclaration(constant, rest) -> None
            self.build(node.constant)
            self.build(node.rest)
        elif isinstance(node, ast.GlobalConstant):
            # GlobalConstant(id, type, value) -> None
            sc = SymbolConstant(node.id, node.type, node.value)
            self.lastScope.addSymbol(sc)
        elif isinstance(node, ast.StructDeclaration):
            # StructDeclaration(struct, rest) -> None
            self.build(node.struct)
            self.build(node.rest)
        elif isinstance(node, ast.Struct):
            # Struct(name, vars) -> None
            tmp = self.lastScope
            sc = StructScope(node.name)
            self.lastScope = sc
            self.build(node.vars)
            tmp.addChild(sc)
            self.lastScope = tmp
        elif isinstance(node, ast.StructVars):
            # StructVars(id, type, rest) -> None
            sv = SymbolVariable(node.id, node.type, None)
            self.lastScope.addSymbol(sv)
            self.build(node.rest)
        elif isinstance(node, ast.Function):
            # Functions(function, rest) -> None
            tmp = self.lastScope
            self.build(node.function)
            self.lastScope = tmp
            self.build(node.rest)
        elif isinstance(node, ast.FunctionDefinition):
            # FunctionDefinition(id, args, returns, body) -> None
            args = self.build(node.args)
            fs = FunctionScope(node.id)
            if args is not None:
                for arg in args:
                    fs.addSymbol(arg)
            self.lastScope.addChild(fs)
            self.lastScope = fs

            vars = self.build(node.body)
            ls = LocalScope()
            for var in vars:
                ls.addSymbol(var)
            self.lastScope.addChild(ls)
            self.lastScope = ls
        elif isinstance(node, ast.FunctionArgs):
            # FunctionArgs(id, type, rest) -> None
            fa = SymbolArgument(node.id, node.type)
            rest = self.build(node.rest)
            if rest is not None:
                rest.append(fa)
                return rest
            else:
                return [fa]
        elif isinstance(node, ast.FunctionReturn):
           # FunctionReturn(type, rest) -> None
           return
        elif isinstance(node, ast.FunctionTypes):
           # FunctionTypes(type, rest) -> None
           return
        elif isinstance(node, ast.BinaryOperation):
           # BinaryOperation(op, left, right) -> None
           return
        elif isinstance(node, ast.GroupExpression):
           # GroupExpression(expression) -> None
           return
        elif isinstance(node, ast.ReturnCode):
          # ReturnCode(expression) -> None
            return []
            #self.build(node.expression)
        elif isinstance(node, ast.VarDefinition):
            # VarDefinition(id, type, expression) -> None
            sv = SymbolVariable(node.id, node.type, node.expression)
            #self.lastScope.addSymbol(sv)
            return [sv]
        elif isinstance(node, ast.VarAssigment):
            # VarAssigment(id, expression) -> None
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


class Scope:
    def __init__(self) -> None:
        self.symbols = []
        self.childs = []

    def addSymbol(self, symbol):
        self.symbols.append(symbol)

    def addChild(self, child):
        self.childs.append(child)


class GlobalScope(Scope):
    def __init__(self) -> None:
        super().__init__()


class LocalScope(Scope):
    def __init__(self) -> None:
        super().__init__()


class FunctionScope(Scope):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name


class StructScope(Scope):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name


class Symbol:
    def __init__(self, name, type, value) -> None:
        self.name = name
        self.type = type
        self.value = value


class SymbolVariable(Symbol):
    def __init__(self, name, type, value) -> None:
        super().__init__(name, type, value)


class SymbolConstant(Symbol):
    def __init__(self, name, type, value) -> None:
        super().__init__(name, type, value)


class SymbolArgument(Symbol):
    def __init__(self, name, type) -> None:
        super().__init__(name, type, None)

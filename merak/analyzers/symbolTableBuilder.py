from ..parser import AST as ast


class SymbolTableBuilder:

    def __init__(self) -> None:
        self.symbols = {}

    def build(self, node):
        scope = self._build_scope(node)
        self._check_types(node, scope=scope)
        return SymbolTable(scope)

    def _build_scope(self, node, scope=None):
        if scope is None:
            gs = GlobalScope()
            return self._build_scope(node, scope=gs)

        if node is None:
            return scope

        if isinstance(node, ast.Definition):
            # Definition(contract, rest) -> None
            scope = self._build_scope(node.contract, scope=scope)
        elif isinstance(node, ast.Contract):
            # Contract(id, definition) -> None
            scope = self._build_scope(node.definition, scope=scope)
        elif isinstance(node, ast.GlobalVariablesDeclaration):
            # GlobalVariablesDeclaration(variable, rest) -> None
            scope = self._build_scope(node.variable, scope=scope)
            scope = self._build_scope(node.rest, scope=scope)
        elif isinstance(node, ast.GlobalVariable):
            # GlobalVariable(id, type) -> None
            sv = SymbolVariable(node.id, node.type, None, scope)
            scope.addSymbol(sv)
        elif isinstance(node, ast.GlobalConstantsDeclaration):
            # GlobalConstantsDeclaration(constant, rest) -> None
            scope = self._build_scope(node.constant, scope=scope)
            scope = self._build_scope(node.rest, scope=scope)
        elif isinstance(node, ast.GlobalConstant):
            # GlobalConstant(id, type, value) -> None
            sc = SymbolConstant(node.id, node.type, node.value, scope)
            scope.addSymbol(sc)
        elif isinstance(node, ast.StructDeclaration):
            # StructDeclaration(struct, rest) -> None
            scopeStruct = self._build_scope(node.struct, scope=scope)
            scope.addChild(scopeStruct)
            scope = self._build_scope(node.rest, scope=scope)
        elif isinstance(node, ast.Struct):
            # Struct(name, vars) -> None
            sc = StructScope(node.name, parent=scope)
            scope = self._build_scope(node.vars, scope=sc)
        elif isinstance(node, ast.StructVars):
            # StructVars(id, type, rest) -> None
            sv = SymbolVariable(node.id, node.type, None, scope)
            scope.addSymbol(sv)
            scope = self._build_scope(node.rest, scope=scope)
        elif isinstance(node, ast.Function):
            # Functions(function, rest) -> None
            scopefn = self._build_scope(node.function, scope=scope)
            scope = self._build_scope(node.rest, scope=scopefn)
        elif isinstance(node, ast.FunctionDefinition):
            # FunctionDefinition(id, args, returns, body) -> None
            fs = FunctionScope(node.id, parent=scope)
            fs = self._build_scope(node.args, scope=fs)
            fs = self._build_scope(node.returns, scope=fs)
            ls = LocalScope(parent=fs)
            ls = self._build_scope(node.body, scope=ls)
            fs.addChild(ls)
            scope.addChild(fs)
        elif isinstance(node, ast.FunctionArgs):
            # FunctionArgs(id, type, rest) -> None
            fa = SymbolArgument(node.id, node.type, scope)
            scope.addSymbol(fa)
            scope = self._build_scope(node.rest, scope=scope)
        elif isinstance(node, ast.FunctionReturn):
            # FunctionReturn(type, rest) -> None
            scope.addReturns(node.type)
            scope = self._build_scope(node.rest, scope=scope)
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
            sv = SymbolVariable(
                node.id, node.type, node.expression, scope
            )
            scope.addSymbol(sv)
            scope = self._build_scope(node.rest, scope=scope)
        elif isinstance(node, ast.VarAssigment):
            # VarAssigment(id, expression, rest) -> None
            scope = self._build_scope(node.rest, scope=scope)
        elif isinstance(node, ast.NameExpression):
            # NameExpression(name) -> None
            pass
        elif isinstance(node, ast.NumberExpression):
            # NumberExpression(sign, number) -> None
            pass
        elif isinstance(node, ast.Empty):
            # Empty() -> None
            pass
        else:
            raise Exception("AST node not recognized: ", node)

        return scope

    def _check_types(self, node, scope=None):
        if scope is None:
            raise Exception('Undefined scope')

        if node is None:
            return scope

        if isinstance(node, ast.Definition):
            # Definition(contract, rest) -> None
            self._check_types(node.contract, scope=scope)
        elif isinstance(node, ast.Contract):
            # Contract(id, definition) -> None
            self._check_types(node.definition, scope=scope)
        elif isinstance(node, ast.GlobalVariablesDeclaration):
            # GlobalVariablesDeclaration(variable, rest) -> None
            self._check_types(node.variable, scope=scope)
            self._check_types(node.rest, scope=scope)
        elif isinstance(node, ast.GlobalConstantsDeclaration):
            # GlobalConstantsDeclaration(constant, rest) -> None
            self._check_types(node.constant, scope=scope)
            self._check_types(node.rest, scope=scope)
        elif isinstance(node, ast.GlobalConstant):
            # GlobalConstant(id, type, value) -> None
            if not self.is_valid_type(node.type, node.value):
                raise Exception(f'Value of {node.id} is not {node.type}')
        elif isinstance(node, ast.StructDeclaration):
            # StructDeclaration(struct, rest) -> None
            self._check_types(node.rest, scope=scope)
        elif isinstance(node, ast.Function):
            # Functions(function, rest) -> None
            self._check_types(node.function, scope=scope)
            self._check_types(node.rest, scope=scope)
        elif isinstance(node, ast.FunctionDefinition):
            # FunctionDefinition(id, args, returns, body) -> None
            fs = scope._get_child_by_name(node.id)
            ls = fs.childs[0]
            value = self._check_types(node.body, scope=ls)
        elif isinstance(node, ast.BinaryOperation):
            # BinaryOperation(op, left, right) -> None
            left_value = self._check_types(node.left, scope=scope)
            right_value = self._check_types(node.right, scope=scope)
            return self.solve_binary_type(node.op, left_value, right_value)
        elif isinstance(node, ast.GroupExpression):
            # GroupExpression(expression) -> None
            return self._check_types(node.expression)
        elif isinstance(node, ast.ReturnCode):
            # ReturnCode(expression) -> None
            value = self._check_types(node.expression, scope=scope)
            returns = scope.parent.returns
            for ret in returns:
                if not self.is_valid_type(ret, value):
                    raise Exception(f'Return type in function {scope.parent.name} is not correct')
        elif isinstance(node, ast.VarDefinition):
            # VarDefinition(id, type, expression, rest) -> None
            value = self._check_types(node.expression, scope=scope)
            symbol = scope._get_symbol_by_name(node.id)
            if not self.is_valid_type(symbol, value):
                raise Exception(f'Value of {node.id} is not {symbol.type}')
            self._check_types(node.rest, scope=scope)
        elif isinstance(node, ast.VarAssigment):
            # VarAssigment(id, expression, rest) -> None
            value = self._check_types(node.expression, scope=scope)
            symbol = scope._get_symbol_by_name(node.id)
            if not self.is_valid_type(symbol, value):
                raise Exception(f'Value of {node.id} is not {value}')
            self._check_types(node.rest, scope=scope)
        elif isinstance(node, ast.NameExpression):
            # NameExpression(name) -> None
            symbol = scope._get_symbol_by_name(node.name)
            return symbol
        elif isinstance(node, ast.NumberExpression):
            # NumberExpression(sign, number) -> None
            if node.sign == '-':
                return UIntType
            else:
                return IntType

        return scope

    def is_valid_type(self, type, value):
        return True

    def solve_binary_type(self, op, left, right):
        return IntType

class SymbolTable:
    def __init__(self, scope) -> None:
        self.scope = scope

    def getScope(self):
        return self.scope

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

    def _get_symbol_by_name(self, name):
        scope = self
        while scope is not None:
            for symbol in scope.symbols:
                if symbol.name == name:
                    return symbol

            scope = scope.parent

        return None

    def _get_child_by_name(self, name):
        for child in self.childs:
            if hasattr(child, 'name') and child.name == name:
                return child

        return None


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
        self.returns = []

    def addReturns(self, child):
        self.returns.append(child)


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


class BuiltInType:
    pass

class IntType(BuiltInType):
    pass

class UIntType(BuiltInType):
    pass

class BoolType(BuiltInType):
    pass

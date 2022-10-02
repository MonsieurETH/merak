from ..parser import AST as ast

class SymbolTableBuilder:

    def __init__(self) -> None:
        self.symbols = {}

    def build(self, node):
        scope = self._build_scope(node)
        self._check_types(node, scope)
        return SymbolTable(scope)

    def _build_scope(self, node):
        gs = GlobalScope()
        visitor = ScopeVisitor()
        return node.build_scope(visitor, scope=gs)

    def _check_types(self, node, scope):
        if scope is None:
            raise Exception('Undefined scope')

        visitor = TypeVisitor()
        node.check_type(visitor, scope=scope)



class SymbolTable:
    def __init__(self, scope) -> None:
        self.scope = scope

    def getScope(self):
        return self.scope

class ScopeVisitor:

    def buildScopeDefinition(self, node: ast.Definition, scope):
        scope = node.contract.build_scope(self, scope=scope)
        return scope

    def buildScopeContract(self, node: ast.Contract, scope):
        scope = node.definition.build_scope(self, scope=scope)
        return scope

    def buildScopeFunction(self, node: ast.Function, scope):
        scope = node.function.build_scope(self, scope=scope)
        if node.rest is not None:
            scope = node.rest.build_scope(self, scope=scope)
        return scope

    def buildScopeGlobalVariableDefinition(self, node: ast.GlobalVariablesDeclaration, scope):
        scope = node.variable.build_scope(self, scope=scope)
        if node.rest is not None:
            scope = node.rest.build_scope(self, scope=scope)
        return scope

    def buildScopeGlobalVariable(self, node: ast.GlobalVariable, scope):
        sv = SymbolVariable(node.id, node.type, None, scope)
        scope.addSymbol(sv)
        return scope

    def buildScopeGlobalConstantsDeclaration(self, node: ast.GlobalConstantsDeclaration, scope):
        scope = node.constant.build_scope(self, scope=scope)
        if node.rest is not None:
            scope = node.rest.build_scope(self, scope=scope)
        return scope

    def buildScopeGlobalConstant(self, node: ast.GlobalConstant, scope):
        sc = SymbolConstant(node.id, node.type, node.value, scope)
        scope.addSymbol(sc)
        return scope

    def buildScopeStructDeclaration(self, node: ast.StructDeclaration, scope):
        scopeStruct = node.struct.build_scope(self, scope=scope)
        scope.addChild(scopeStruct)
        if node.rest is not None:
            scope = node.rest.build_scope(self, scope=scope)
        return scope

    def buildScopeStruct(self, node: ast.Struct, scope):
        sc = StructScope(node.name, parent=scope)
        scope = node.vars.build_scope(self, scope=sc)
        return scope

    def buildScopeStructVars(self, node: ast.StructVars, scope):
        sv = SymbolVariable(node.id, node.type, None, scope)
        scope.addSymbol(sv)
        if node.rest is not None:
            scope = node.rest.build_scope(self, scope=scope)
        return scope

    def buildScopeFunctionDefinition(self, node: ast.FunctionDefinition, scope):
        fs = FunctionScope(node.id, parent=scope)
        fs = node.args.build_scope(self, scope=fs)
        fs = node.returns.build_scope(self, scope=fs)
        ls = LocalScope(parent=fs)
        ls = node.body.build_scope(self, scope=ls)
        fs.addChild(ls)
        scope.addChild(fs)
        return scope

    def buildScopeFunctionArgs(self, node: ast.FunctionArgs, scope):
        fa = SymbolArgument(node.id, node.type, scope)
        scope.addSymbol(fa)
        if node.rest is not None:
            scope = node.rest.build_scope(self, scope=scope)
        return scope

    def buildScopeFunctionReturn(self, node: ast.FunctionReturn, scope):
        scope.addReturns(node.type)
        if node.rest is not None:
            scope = node.rest.build_scope(self, scope=scope)
        return scope

    def buildScopeFunctionTypes(self, node: ast.FunctionTypes, scope):
        return scope

    def buildScopeBinaryOperation(self, node: ast.BinaryOperation, scope):
        return scope

    def buildScopeGroupExpression(self, node: ast.GroupExpression, scope):
        return scope

    def buildScopeReturnCode(self, node: ast.ReturnCode, scope):
        return scope

    def buildScopeVarDefinition(self, node: ast.VarDefinition, scope):
        sv = SymbolVariable(
            node.id, node.type, node.expression, scope
        )
        scope.addSymbol(sv)
        if node.rest is not None:
            scope = node.rest.build_scope(self, scope=scope)
        return scope

    def buildScopeVarAssigment(self, node: ast.VarAssigment, scope):
        if node.rest is not None:
            scope = node.rest.build_scope(self, scope=scope)
        return scope

    def buildScopeNameExpression(self, node: ast.NameExpression, scope):
        return scope

    def buildScopeNumberExpression(self, node: ast.NumberExpression, scope):
        return scope

    def buildScopeEmpty(self, node: ast.Empty, scope):
        return scope

class TypeVisitor:

    def checkTypeDefinition(self, node: ast.Definition, scope):
        # Definition(contract, rest) -> None
        node.contract.check_type(self, scope=scope)

    def checkTypeContract(self, node: ast.Contract, scope):
        # Contract(id, definition) -> None
        node.definition.check_type(self, scope=scope)

    def checkTypeFunction(self, node: ast.Function, scope):
        # Function(function, rest) -> None
        node.function.check_type(self, scope=scope)
        if node.rest is not None:
            node.rest.check_type(self, scope=scope)

    def checkTypeGlobalVariableDefinition(self, node: ast.GlobalVariablesDeclaration, scope):
        # GlobalVariablesDeclaration(variable, rest) -> None
        if node.rest is not None:
            node.rest.check_type(self, scope=scope)

    def checkTypeGlobalConstantsDeclaration(self, node: ast.GlobalConstantsDeclaration, scope):
        # GlobalConstantsDeclaration(constant, rest) -> None
        node.constant.check_type(self, scope=scope)
        if node.rest is not None:
            node.rest.check_type(self, scope=scope)

    def checkTypeGlobalConstant(self, node: ast.GlobalConstant, scope):
        # GlobalConstant(id, type, value) -> None
        if not self.is_valid_type(node.type, node.value):
            raise Exception(f'Value of {node.id} is not {node.type}')

    def checkTypeStructDeclaration(self, node: ast.StructDeclaration, scope):
        # StructDeclaration(struct, rest) -> None
        if node.rest is not None:
            node.rest.check_type(self, scope=scope)

    def checkTypeFunctionDefinition(self, node: ast.FunctionDefinition, scope):
        # FunctionDefinition(id, args, returns, body) -> None
        fs = scope._get_child_by_name(node.id)
        ls = fs.childs[0]
        node.body.check_type(self, scope=ls)

    def checkTypeBinaryOperation(self, node: ast.BinaryOperation, scope):
        # BinaryOperation(op, left, right) -> None
        left_value = node.left.check_type(self, scope=scope)
        right_value = node.right.check_type(self, scope=scope)
        return self.solve_binary_type(node.op, left_value, right_value)

    def checkTypeGroupExpression(self, node: ast.GroupExpression, scope):
        # GroupExpression(expression) -> None
        return self.check_type(node.expression)

    def checkTypeReturnCode(self, node: ast.ReturnCode, scope):
        # ReturnCode(expression) -> None
        value = node.expression.check_type(self, scope=scope)
        returns = scope.parent.returns
        for ret in returns:
            if not self.is_valid_type(ret, value):
                raise Exception(f'Return type in function {scope.parent.name} is not correct')

    def checkTypeVarDefinition(self, node: ast.VarDefinition, scope):
        # VarDefinition(id, type, expression, rest) -> None
        value = node.expression.check_type(self, scope=scope)
        symbol = scope._get_symbol_by_name(node.id)
        if not self.is_valid_type(symbol, value):
            raise Exception(f'Value of {node.id} is not {symbol.type}')
        if node.rest is not None:
            node.rest.check_type(self, scope=scope)

    def checkTypeVarAssigment(self, node: ast.VarAssigment, scope):
        # VarAssigment(id, expression, rest) -> None
        value = node.expression.check_type(self, scope=scope)
        symbol = scope._get_symbol_by_name(node.id)
        if not self.is_valid_type(symbol, value):
            raise Exception(f'Value of {node.id} is not {value}')
        if node.rest is not None:
            node.rest.check_type(self, scope=scope)

    def checkTypeNameExpression(self, node: ast.NameExpression, scope):
        # NameExpression(name) -> None
        return scope._get_symbol_by_name(node.name)

    def checkTypeNumberExpression(self, node: ast.NumberExpression, scope):
        # NumberExpression(sign, number) -> None
        if node.sign == '-':
            return UIntType
        else:
            return IntType

    def checkTypeEmpty(self, node: ast.Empty, scope):
        return scope

    def is_valid_type(self, type, value):
        return True

    def solve_binary_type(self, op, left, right):
        return IntType

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

class Definition:
    def __init__(self, contract, rest) -> None:
        self.contract = contract
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeDefinition(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeDefinition(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenDefinition(self, scope)



class Contract:
    def __init__(self, id, definition) -> None:
        self.id = id
        self.definition = definition

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeContract(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeContract(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenContract(self, scope)


class Function:
    def __init__(self, function, rest) -> None:
        self.function = function
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeFunction(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeFunction(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenFunction(self, scope)


class GlobalVariablesDeclaration:
    def __init__(self, variable, rest) -> None:
        self.variable = variable
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeGlobalVariablesDeclaration(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeGlobalVariablesDeclaration(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenGlobalVariablesDeclaration(self, scope)


class GlobalVariable:
    def __init__(self, id, type) -> None:
        self.id = id
        self.type = type

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeGlobalVariable(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeGlobalVariable(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenGlobalVariable(self, scope)


class GlobalConstantsDeclaration:
    def __init__(self, constant, rest) -> None:
        self.constant = constant
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeGlobalConstantsDeclaration(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeGlobalConstantsDeclaration(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenGlobalConstantsDeclaration(self, scope)


class GlobalConstant:
    def __init__(self, id, type, value) -> None:
        self.id = id
        self.type = type
        self.value = value

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeGlobalConstant(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeGlobalConstant(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenGlobalConstant(self, scope)


class StructDeclaration:
    def __init__(self, struct, rest) -> None:
        self.struct = struct
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeStructDeclaration(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeStructDeclaration(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenStructDeclaration(self, scope)


class Struct:
    def __init__(self, name, vars) -> None:
        self.name = name
        self.vars = vars

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeStruct(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeStruct(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenStruct(self, scope)


class StructVars:
    def __init__(self, id, type, rest) -> None:
        self.id = id
        self.type = type
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeStructVars(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeStructVars(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenStructVars(self, scope)


class FunctionDefinition:
    def __init__(self, id, args, returns, types, body) -> None:
        self.id = id
        self.args = args
        self.returns = returns
        self.types = types
        self.body = body

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeFunctionDefinition(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeFunctionDefinition(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenFunctionDefinition(self, scope)


class FunctionArgs:
    def __init__(self, id, type, rest) -> None:
        self.id = id
        self.type = type
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeFunctionArgs(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeFunctionArgs(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenFunctionArgs(self, scope)


class FunctionReturn:
    def __init__(self, type, rest) -> None:
        self.type = type
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeFunctionReturn(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeFunctionReturn(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenFunctionReturn(self, scope)


class FunctionTypes:
    def __init__(self, type, rest) -> None:
        self.type = type
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeFunctionTypes(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeFunctionTypes(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenFunctionTypes(self, scope)


class BinaryOperation:
    def __init__(self, op, left, right) -> None:
        self.op = op
        self.left = left
        self.right = right

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeBinaryOperation(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeBinaryOperation(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenBinaryOperation(self, scope)

class StructCall:
    def __init__(self, id, args) -> None:
        self.id = id
        self.args = args

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeStructCall(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeStructCall(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenStructCall(self, scope)

class DictEntry:
    def __init__(self, key, value, rest) -> None:
        self.key = key
        self.value = value
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeDictEntry(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeDictEntry(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenDictEntry(self, scope)


class GroupExpression:
    def __init__(self, expression) -> None:
        self.expression = expression

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeGroupExpression(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeGroupExpression(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenGroupExpression(self, scope)


class ReturnCode:
    def __init__(self, expression) -> None:
        self.expression = expression

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeReturnCode(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeReturnCode(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenReturnCode(self, scope)


class VarDefinition:
    def __init__(self, id, type, expression, rest) -> None:
        self.id = id
        self.type = type
        self.expression = expression
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeVarDefinition(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeVarDefinition(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenVarDefinition(self, scope)

class VarAssigment:
    def __init__(self, id, expression, rest) -> None:
        self.id = id
        self.expression = expression
        self.rest = rest

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeVarAssigment(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeVarAssigment(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenVarAssigment(self, scope)

class NameExpression:
    def __init__(self, name) -> None:
        self.name = name

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeNameExpression(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeNameExpression(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenNameExpression(self, scope)


class NumberExpression:
    def __init__(self, sign, number) -> None:
        self.sign = sign
        self.number = number

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeNumberExpression(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeNumberExpression(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenNumberExpression(self, scope)

class Empty:

    def build_scope(self, visitor, scope=None):
        return visitor.buildScopeEmpty(self, scope)

    def check_type(self, visitor, scope=None):
        visitor.checkTypeEmpty(self, scope)

    def gen_code(self, visitor, scope=None):
        return visitor.codeGenEmpty(self, scope)


class Definition:
    def __init__(self, contract, rest) -> None:
        self.contract = contract
        self.rest = rest


class Contract:
    def __init__(self, id, definition) -> None:
        self.id = id
        self.definition = definition


class Function:
    def __init__(self, function, rest) -> None:
        self.function = function
        self.rest = rest


class GlobalVariablesDeclaration:
    def __init__(self, variable, rest) -> None:
        self.variable = variable
        self.rest = rest


class GlobalVariable:
    def __init__(self, id, type) -> None:
        self.id = id
        self.type = type


class GlobalConstantsDeclaration:
    def __init__(self, constant, rest) -> None:
        self.constant = constant
        self.rest = rest


class GlobalConstant:
    def __init__(self, id, type, value) -> None:
        self.id = id
        self.type = type
        self.value = value


class StructDeclaration:
    def __init__(self, struct, rest) -> None:
        self.struct = struct
        self.rest = rest


class Struct:
    def __init__(self, name, vars) -> None:
        self.name = name
        self.vars = vars


class StructVars:
    def __init__(self, id, type, rest) -> None:
        self.id = id
        self.type = type
        self.rest = rest


class FunctionDefinition:
    def __init__(self, id, args, returns, body) -> None:
        self.id = id
        self.args = args
        self.returns = returns
        self.body = body


class FunctionArgs:
    def __init__(self, id, type, rest) -> None:
        self.id = id
        self.type = type
        self.rest = rest


class FunctionReturn:
    def __init__(self, type, rest) -> None:
        self.type = type
        self.rest = rest


class FunctionTypes:
    def __init__(self, type, rest) -> None:
        self.type = type
        self.rest = rest


class BinaryOperation:
    def __init__(self, op, left, right) -> None:
        self.op = op
        self.left = left
        self.right = right


class GroupExpression:
    def __init__(self, expression) -> None:
        self.expression = expression


class ReturnCode:
    def __init__(self, expression) -> None:
        self.expression = expression


class VarDefinition:
    def __init__(self, id, type, expression) -> None:
        self.id = id
        self.type = type
        self.expression = expression

class VarAssigment:
    def __init__(self, id, expression) -> None:
        self.id = id
        self.expression = expression


class NameExpression:
    def __init__(self, name) -> None:
        self.name = name


class NumberExpression:
    def __init__(self, sign, number) -> None:
        self.sign = sign
        self.number = number


class Empty:
    pass

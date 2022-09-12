from sly import Parser

from .AST import *

from ..lexer import MerakLexer


class MerakParser(Parser):
    tokens = MerakLexer.tokens

    precedence = (
        ("left", PLUS, MINUS),
        ("left", TIMES, DIVIDE),
        ("right", UMINUS),
    )

    """
    impl contrateto {

        storage const supply: i256 = 10000;
        storage allowance: HashMap<u256, u256>;

        fn function1(param1: u256) -> (u256) view { // view|pure|payable|nonpayable
            return param1 + 1;
        };

        fn function2() -> (u256) {
            var1: u256 = 10;
            return allowance[var1]
        }
    }
    """

    """
    sourceInit:
        contractDefinition sourceInit
        | empty

    contractDefintion: CONTRACT ID LBRACE contractBody RBRACE

    contractBody:
        functionDefinition contractBody
        | globalVariableDeclaration contractBody
        | globalConstantDeclaration contractBody
        | empty

    functionDefinition:
        FUNC ID LPAREN functionArgs RPAREN functionTypes LBRACE functionCode RBRACE SEMICOLON
        | FUNC ID LPAREN functionArgs RPAREN RARROW LPAREN functionReturn RPAREN functionTypes LBRACE functionCode RBRACE SEMICOLON

    functionArgs:
        ID COLON type
        | ID COLON type COMMA functionArgs
        | empty

    functionTypes: VIEW | PURE | PAYABLE | NONPAYABLE | empty

    functionReturn: type COMMA functionReturn | type

    functionCode:
        ID COLON type ASSIGN expression SEMICOLON
        | RETURN expression SEMICOLON

    expression:
        expression PLUS expression
        | expression MINUS expression
        | expression TIMES expression
        | expression DIVIDE expression
        | LPAREN expression RPAREN
        | value

    value:
        NUMBER
        | UMINUS NUMBER

    type: WORD | UWORD

    globalVariableDeclaration: STORAGE ID COLON type SEMICOLON

    globalConstantDeclaration: STORAGE CONST ID COLON type ASSIGN value SEMICOLON
    """

    """"
    sourceInit:
        contractDefinition sourceInit
        | empty
    """

    @_("contractDefinition sourceInit", "empty")
    def sourceInit(self, p):
        if hasattr(p, "contractDefinition"):
            return Definition(p.contractDefinition, p.sourceInit)

        return Empty()

    """
    contractDefintion: CONTRACT ID LBRACE contractBody RBRACE
    """

    @_("CONTRACT ID LBRACE contractBody RBRACE")
    def contractDefinition(self, p):
        return Contract(p.ID, p.contractBody)

    """
    contractBody:
        functionDefinition contractBody
        | globalVariableDeclaration contractBody
        | globalConstantDeclaration contractBody
        | empty
    """

    @_(
        "functionDefinition contractBody",
        "globalVariableDeclaration contractBody",
        "globalConstantDeclaration contractBody",
        "empty",
    )
    def contractBody(self, p):
        if hasattr(p, "functionDefinition"):
            return Functions(p.functionDefinition, p.contractBody)
        if hasattr(p, "globalVariableDeclaration"):
            return GlobalVariablesDeclaration(
                p.globalVariableDeclaration, p.contractBody
            )
        if hasattr(p, "globalConstantDeclaration"):
            return GlobalConstantsDeclaration(
                p.globalConstantDeclaration, p.contractBody
            )
        return Empty()

    """
    globalVariableDeclaration: STATIC ID COLON type SEMICOLON
    """

    @_("STATIC ID COLON type SEMICOLON")
    def globalVariableDeclaration(self, p):
        return GlobalVariable(p.ID, p.type)

    """
    globalConstantDeclaration: STATIC CONST ID COLON type ASSIGN value SEMICOLON
    """

    @_("STATIC CONST ID COLON type ASSIGN value SEMICOLON")
    def globalConstantDeclaration(self, p):
        return GlobalConstant(p.ID, p.type, p.value)

    """
    functionDefinition:
        FUNC ID LPAREN functionArgs RPAREN functionTypes LBRACE functionCode RBRACE SEMICOLON
        | FUNC ID LPAREN functionArgs RPAREN RARROW LPAREN functionReturn RPAREN functionTypes LBRACE functionCode RBRACE SEMICOLON
    """

    @_(
        "FUNC ID LPAREN functionArgs RPAREN functionTypes LBRACE functionCode RBRACE SEMICOLON",
        "FUNC ID LPAREN functionArgs RPAREN RARROW LPAREN functionReturn RPAREN functionTypes LBRACE functionCode RBRACE SEMICOLON",
    )
    def functionDefinition(self, p):
        if hasattr(p, "functionReturn"):
            return FunctionDefinition(
                p.ID, p.functionArgs, p.functionReturn, p.functionCode
            )
        return FunctionDefinition(p.ID, p.functionArgs, None, p.functionCode)

    """
    functionArgs:
        ID COLON type
        | ID COLON type COMMA functionArgs
        | empty
    """

    @_("ID COLON type", "ID COLON type COMMA functionArgs", "empty")
    def functionArgs(self, p):
        if hasattr(p, "functionArgs"):
            return FunctionArgs(p.ID, p.type, p.functionArgs)
        if hasattr(p, "type"):
            return FunctionArgs(p.ID, p.type, None)

        return Empty()

    """
    functionTypes:
        VIEW functionTypes
        | PURE functionTypes
        | PAYABLE functionTypes
        | NONPAYABLE functionTypes
        | empty
    """

    @_(
        "VIEW functionTypes",
        "PURE functionTypes",
        "PAYABLE functionTypes",
        "NONPAYABLE functionTypes",
        "empty",
    )
    def functionTypes(self, p):
        if hasattr(p, "functionTypes"):
            return FunctionTypes(p[0], p.functionTypes)

        return Empty()

    """
    functionReturn: type COMMA functionReturn | type
    """

    @_("type COMMA functionReturn", "type")
    def functionReturn(self, p):
        if hasattr(p, "functionArgs"):
            return FunctionReturn(p.type, p.functionReturn)

        return FunctionReturn(p.type, None)

    """
    type: WORD | UWORD
    """

    @_("WORD", "UWORD")
    def type(self, p):
        return p[0]

    """
    functionCode:
        ID ASSIGN expression SEMICOLON
        | RETURN expression SEMICOLON
    """

    @_("RETURN expression SEMICOLON")
    def functionCode(self, p):
        return ReturnCode(p.expression)

    @_(
        "ID COLON type ASSIGN expression SEMICOLON",
        "ID ASSIGN expression SEMICOLON",
    )
    def functionCode(self, p):
        if hasattr(p, "type"):
            return AssignCode(p.ID, p.type, p.expression)

        return AssignCode(p.ID, None, p.expression)

    """
    expression:
        expression PLUS expression
        | expression MINUS expression
        | expression TIMES expression
        | expression DIVIDE expression
        | LPAREN expression RPAREN
        | value
    """

    @_(
        "expression PLUS expression",
        "expression MINUS expression",
        "expression TIMES expression",
        "expression DIVIDE expression",
    )
    def expression(self, p):
        if p[1] == "+":
            op = "add"
        elif p[1] == "-":
            op = "sub"
        elif p[1] == "*":
            op = "mul"
        elif p[1] == "/":
            op = "div"
        else:
            raise Exception(f"Binary operation not recognized: {p[1]}")
        return BinaryOperation(op, p.expression0, p.expression1)

    @_("LPAREN expression RPAREN")
    def expression(self, p):
        return GroupExpression(p.expression)

    @_("value")
    def expression(self, p):
        return p[0]

    """
    value:
        NUMBER
        | UMINUS NUMBER
    """

    @_("NUMBER", "UMINUS NUMBER")
    def value(self, p):
        if hasattr(p, "UMINUS"):
            return NumberExpression("-", p.NUMBER)

        return NumberExpression("+", p.NUMBER)

    @_("ID")
    def expression(self, p):
        return NameExpression(p.ID)

    @_("")
    def empty(self, _):
        return Empty()

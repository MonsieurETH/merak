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

        struct Data {
            A: u256;
            B: i256;
        }

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
        | structDeclaration contractBody
        | empty

    structDeclaration: STRUCT ID LBRACE structVars RBRACE

    structVars:
        ID COLON type SEMICOLON functionArgs
        | empty

    functionDefinition:
        FUNC ID LPAREN functionArgs RPAREN functionTypes LBRACE functionCode RBRACE
        | FUNC ID LPAREN functionArgs RPAREN RARROW LPAREN functionReturn RPAREN functionTypes LBRACE functionCode RBRACE

    functionArgs:
        ID COLON type
        | ID COLON type COMMA functionArgs
        | empty

    functionTypes: VIEW | PURE | PAYABLE | NONPAYABLE | empty

    functionReturn: type COMMA functionReturn | type

    functionCode:
        ID ASSIGN expression SEMICOLON functionCode
        | ID COLON type ASSIGN expression SEMICOLON functionCode
        | RETURN expression SEMICOLON

    expression:
        expression PLUS expression
        | expression MINUS expression
        | expression TIMES expression
        | expression DIVIDE expression
        | LPAREN expression RPAREN
        | value
        | call

    value:
        NUMBER
        | UMINUS NUMBER

    call:
        ID funcCall SEMICOLON
        | ID structCall SEMICOLON

    funcCall:
        DOT LPAREN varList RPAREN funcCall
        | empty

    structCall:
        DOT LPAREN LBRACE dict RBRACE RPAREN

    dict:
        ID COLON ID
        | ID COLON ID COMMA dict
        | empty

    varList:
        ID
        | ID COMMA varList
        | empty

    type: INT | UINT | BOOL

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
        | structDeclaration contractBody
        | empty
    """

    @_(
        "functionDefinition contractBody",
        "globalVariableDeclaration contractBody",
        "globalConstantDeclaration contractBody",
        "structDeclaration contractBody",
        "empty",
    )
    def contractBody(self, p):
        if hasattr(p, "functionDefinition"):
            return Function(p.functionDefinition, p.contractBody)
        if hasattr(p, "globalVariableDeclaration"):
            return GlobalVariablesDeclaration(
                p.globalVariableDeclaration, p.contractBody
            )
        if hasattr(p, "globalConstantDeclaration"):
            return GlobalConstantsDeclaration(
                p.globalConstantDeclaration, p.contractBody
            )
        if hasattr(p, "structDeclaration"):
            return StructDeclaration(p.structDeclaration, p.contractBody)
        return Empty()

    """
    globalVariableDeclaration: STORAGE ID COLON type SEMICOLON
    """

    @_("STORAGE ID COLON type SEMICOLON")
    def globalVariableDeclaration(self, p):
        return GlobalVariable(p.ID, p.type)

    """
    globalConstantDeclaration: STORAGE CONST ID COLON type ASSIGN value SEMICOLON
    """

    @_("STORAGE CONST ID COLON type ASSIGN value SEMICOLON")
    def globalConstantDeclaration(self, p):
        return GlobalConstant(p.ID, p.type, p.value)

    """
    functionDefinition:
        FUNC ID LPAREN functionArgs RPAREN functionTypes LBRACE functionCode RBRACE
        | FUNC ID LPAREN functionArgs RPAREN RARROW LPAREN functionReturn RPAREN functionTypes LBRACE functionCode RBRACE
    """

    @_(
        "FUNC ID LPAREN functionArgs RPAREN functionTypes LBRACE functionCode RBRACE",
        "FUNC ID LPAREN functionArgs RPAREN RARROW LPAREN functionReturn RPAREN functionTypes LBRACE functionCode RBRACE",
    )
    def functionDefinition(self, p):
        if hasattr(p, "functionReturn"):
            return FunctionDefinition(
                p.ID, p.functionArgs, p.functionReturn, p.functionTypes, p.functionCode
            )
        return FunctionDefinition(p.ID, p.functionArgs, None, p.functionTypes, p.functionCode)

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
    structDeclaration: STRUCT ID LBRACE structVars RBRACE
    """

    @_("STRUCT ID LBRACE structVars RBRACE")
    def structDeclaration(self, p):
        return Struct(p.ID, p.structVars)

    """
    structVars:
        ID COLON type SEMICOLON structVars
        | empty
    """

    @_("ID COLON type SEMICOLON structVars", "empty")
    def structVars(self, p):
        if hasattr(p, "structVars"):
            return StructVars(p.ID, p.type, p.structVars)
        if hasattr(p, "type"):
            return StructVars(p.ID, p.type, None)

        return Empty()

    """
    type: INT | UINT | BOOL
    """

    @_("INT", "UINT", "BOOL")
    def type(self, p):
        return p[0]

    """
    functionCode:
        ID ASSIGN expression SEMICOLON functionCode
        | ID COLON type ASSIGN expression SEMICOLON functionCode
        | RETURN expression SEMICOLON
    """

    @_(
        "ID COLON type ASSIGN expression SEMICOLON functionCode",
        "ID ASSIGN expression SEMICOLON functionCode",
        "empty"
    )
    def functionCode(self, p):
        if hasattr(p, "type"):
            return VarDefinition(p.ID, p.type, p.expression, p.functionCode)
        if hasattr(p, "expression"):
            return VarAssigment(p.ID, p.expression, p.functionCode)

        return Empty()

    @_("RETURN expression SEMICOLON")
    def functionCode(self, p):
        return ReturnCode(p.expression)

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

    @_("call")
    def expression(self, p):
        return p[0]

    '''call:
        ID funcCall SEMICOLON
        | ID structCall SEMICOLON
    '''
    @_("ID structCall SEMICOLON")
    def call(self, p):
        if hasattr(p, "structCall"):
            return StructCall(p.ID, p.structCall)

    '''
    structCall:
        DOT LPAREN LBRACE dict RBRACE RPAREN
    '''
    @_("DOT LPAREN LBRACE dict RBRACE RPAREN")
    def structCall(self, p):
        if hasattr(p, "structCall"):
            return p.dict

    '''
    dict:
        ID COLON ID
        | ID COLON ID COMMA dict
        | empty
    '''
    def dict(self, p):
        if hasattr(p, "dict"):
            return DictEntry(p.ID0, p.ID1, p.dict)
        if hasattr(p, "ID0"):
            return DictEntry(p.ID0, p.ID1, None)

        return Empty()
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
    def empty(self, p):
        return Empty()

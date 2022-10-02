from ..parser import AST as ast

from .huffIR import *

class CodeGenerator:

    def _code_gen(self, node, scope=None):
        cgVisitor = CodeGenVisitor()
        code = node.gen_code(cgVisitor, scope=scope)
        return code

class CodeGenVisitor:

    def codeGenDefinition(self, node: ast.Definition, scope):
        return node.contract.gen_code(self, scope=scope)

    def codeGenContract(self, node: ast.Contract, scope):
        data = node.definition.gen_code(self, scope=scope)
        return HuffContract(data)

    def codeGenFunction(self, node: ast.Function, scope):
        func = node.function.gen_code(self, scope=scope)
        rest = None
        if node.rest is not None:
            rest = node.rest.gen_code(self, scope=scope)
        if rest is not None:
            rest.append(func)
        else:
            rest = [func]
        return rest

    def codeGenGlobalVariableDefinition(self, node: ast.GlobalVariablesDeclaration, scope):
        var = node.variable.gen_code(self, scope=scope)
        rest = None
        if node.rest is not None:
            rest = node.rest.gen_code(self, scope=scope)
        if rest is not None:
            rest.append(var)
        else:
            rest = [var]
        return rest

    def codeGenGlobalVariable(self, node: ast.GlobalVariable, scope):
        return HuffGlobalVar(node.id, node.type)


    def codeGenGlobalConstantsDeclaration(self, node: ast.GlobalConstantsDeclaration, scope):
        const = node.variable.gen_code(self, scope=scope)
        rest = None
        if node.rest is not None:
            rest = node.rest.gen_code(self, scope=scope)
        if rest is not None:
            rest.append(const)
        else:
            rest = [const]
        return rest

    def codeGenGlobalConstant(self, node: ast.GlobalConstant, scope):
        casted_value = node.value
        return HuffGlobalConst(node.id, node.type, casted_value)

    def codeGenStructDeclaration(self, node: ast.StructDeclaration, scope):
        return node.rest.gen_code(self, scope=scope)

    def codeGenStruct(self, node: ast.Struct, scope):
        #????
        pass

    def codeGenStructVars(self, node: ast.StructVars, scope):
        #????
        pass

    def codeGenFunctionDefinition(self, node: ast.FunctionDefinition, scope):
        funcArgs = node.args.gen_code(self, scope=scope)
        funcReturns = node.returns.gen_code(self, scope=scope)
        funcTypes = node.types.gen_code(self, scope=scope)
        funcBody = node.body.gen_code(self, scope=scope)
        return HuffFunction(node.id, funcArgs, funcReturns, funcTypes, funcBody)

    def codeGenFunctionArgs(self, node: ast.FunctionArgs, scope):
        arg = node.id #cast_args(node.id, node.type)
        rest = None
        if node.rest is not None:
            rest = node.rest.gen_code(self, scope=scope)
        if rest is not None:
            rest.append(arg)
        else:
            rest = [arg]
        return rest

    def codeGenFunctionReturn(self, node: ast.FunctionReturn, scope):
        type = node.type #cast_return(node.type)
        rest = None
        if node.rest is not None:
            rest = node.rest.gen_code(self, scope=scope)
        if rest is not None:
            rest.append(type)
        else:
            rest = [type]
        return rest

    def codeGenFunctionTypes(self, node: ast.FunctionTypes, scope):
        type = node.type #validate_function_type(node.type)
        rest = None
        if node.rest is not None:
            rest = node.rest.gen_code(self, scope=scope)
        if rest is not None:
            rest.append(type)
        else:
            rest = [type]
        return rest

    def codeGenBinaryOperation(self, node: ast.BinaryOperation, scope):
        left = node.left.gen_code(self, scope=scope)
        right = node.right.gen_code(self, scope=scope)
        #op = node.op.gen_code(self, scope=scope)

        #return transform_op(op, left, right)

    def codeGenGroupExpression(self, node: ast.GroupExpression, scope):
        exp = node.expression.gen_code(self, scope=scope)
        return exp

    def codeGenReturnCode(self, node: ast.ReturnCode, scope):
        ret = node.expression.gen_code(self, scope=scope)
        #return parse_return(ret, scope)

    def codeGenVarDefinition(self, node: ast.VarDefinition, scope):
        value = node.expression.gen_code(self, scope=scope)
        #var = define_var(node.id, node.type, value, scope)
        rest = None
        if node.rest is not None:
            rest = node.rest.gen_code(self, scope=scope)
        #combine value with rest
        #return combined

    def codeGenVarAssigment(self, node: ast.VarAssigment, scope):
        value = node.expression.gen_code(self, scope=scope)
        #var = assign_var(node.id, node.type, value, scope)
        rest = None
        if node.rest is not None:
            rest = node.rest.gen_code(self, scope=scope)
        #combine value with rest
        #return combined

    def codeGenNameExpression(self, node: ast.NameExpression, scope):
        return node.name

    def codeGenNumberExpression(self, node: ast.NumberExpression, scope):
        #parse type based on sign
        return node.number

    def codeGenEmpty(self, node: ast.Empty, scope):
        return None
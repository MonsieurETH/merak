use std::ops::Neg;
use z3::ast::Ast;
use z3::ast::{Bool, Int, String};
use z3::{Config, Context, SatResult, Solver};

use thiserror::Error;

use crate::ast::*;
use crate::environment::*;

#[derive(Error, Debug, PartialEq)]
pub enum TypeCheckerError {
    #[error("Type mismatch in {0}")]
    TypeMismatch(std::string::String),
    #[error("Variable {0} already declared in scope")]
    VariableRedeclaration(std::string::String),
    #[error("Variable {0} not declared in scope")]
    VariableNotDeclared(std::string::String),
    #[error("State {0} not declared")]
    StateNotDeclared(std::string::String),
    #[error("{0} condition must be of type bool")]
    ConditionMustBeBool(std::string::String),
    #[error("Invalid return statement in function with no return type")]
    InvalidReturnStatement,
}

pub struct TypeChecker {
    pub environment: Environment,
    pub z3_context: Context,
}

#[derive(Debug, Clone)]
pub enum AstZ3<'a> {
    Bool(Bool<'a>),
    Int(Int<'a>),
    String(String<'a>),
}

// Checks to implement:
// [x] Basic type consistency
// [x] Variable redeclaration
// [x] Valid state definition
// [ ] All states rechable (become not implemented)
// [ ] Improved type consistency (solve literals, etc)
// [ ] Tests for all of the above and more

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            environment: Environment::new(),
            z3_context: Context::new(&Config::new()),
        }
    }

    pub fn check_program(&mut self, program: &Program) -> Result<Environment, TypeCheckerError> {
        for contract in program.contracts() {
            self.check_type_consistency_contract(contract)?;
        }

        for state_def in program.state_defs() {
            self.check_type_consistency_state_def(state_def)?;
        }

        Ok(self.environment.clone())
    }

    fn check_type_consistency_contract(
        &mut self,
        contract: &Contract,
    ) -> Result<(), TypeCheckerError> {
        for state in contract.states().iter() {
            self.environment.insert_state(state.clone());
        }

        for state_data in contract.state_data().iter() {
            let state_type = state_data.get_type();
            if let Some(expr) = state_data.value() {
                let expr_type = expr.get_type(&self.environment);
                let type_mismatch = match state_type {
                    Type::Int | Type::Bool | Type::String | Type::Address => {
                        &expr_type != state_type
                    }
                    _ => unreachable!("Error: Invalid type in state data declaration"),
                };
                if type_mismatch {
                    return Err(TypeCheckerError::TypeMismatch(
                        "state data declaration".to_string(),
                    ));
                }
            }

            self.environment.insert_symbol(Symbol::new(
                state_data.name().clone(),
                state_type.clone(),
                state_data.value().clone(),
                None,
                if state_data.is_constant() {
                    Scope::GlobalConst
                } else {
                    Scope::GlobalVar
                },
            ));
        }

        if let Some(constructor) = &contract.constructor() {
            self.environment.push_function("constructor".to_string());
            self.check_type_consistency_constructor(constructor)?;
        }

        Ok(())
    }

    fn check_type_consistency_state_def(
        &mut self,
        state_def: &ContractState,
    ) -> Result<(), TypeCheckerError> {
        if !self.environment.locate_state(state_def.name()) {
            return Err(TypeCheckerError::StateNotDeclared(state_def.name().clone()));
        }

        for function in state_def.functions() {
            self.check_type_consistency_function(function)?;
        }

        Ok(())
    }

    fn check_type_consistency_constructor(
        &mut self,
        constructor: &ContractConstructor,
    ) -> Result<(), TypeCheckerError> {
        for param in constructor.parameters() {
            let value = None;
            self.environment.insert_symbol(Symbol::new(
                param.name().clone(),
                param.get_type().clone(),
                value,
                param.constraint().clone(),
                Scope::Param,
            ));
        }

        for statement in constructor.statements() {
            self.check_type_consistency_statement(statement)?;
        }

        Ok(())
    }

    fn check_type_consistency_function(
        &mut self,
        function: &ContractFunction,
    ) -> Result<(), TypeCheckerError> {
        self.environment.push_function(function.name().clone());
        if let Some(return_ty) = function.return_type() {
            self.environment.set_scope_return_symbol(return_ty.clone());
        }

        for param in function.parameters() {
            let value = None;
            self.environment.insert_symbol(Symbol::new(
                param.name().clone(),
                param.get_type().clone(),
                value,
                param.constraint().clone(),
                Scope::Param,
            ));
        }

        for statement in function.statements() {
            self.check_type_consistency_statement(statement)?;
        }

        self.check_function_constraints(function);

        self.environment.pop_scope();

        Ok(())
    }

    fn check_type_consistency_statement(
        &mut self,
        statement: &Statement,
    ) -> Result<(), TypeCheckerError> {
        match statement {
            Statement::VarDeclaration(name, ty, expr) => {
                if let Some(inner_expr) = expr {
                    let expr_ty = inner_expr.get_type(&self.environment);
                    // TODO Maybe != is too strict
                    if ty != &expr_ty {
                        return Err(TypeCheckerError::TypeMismatch(
                            "variable declaration".to_string(),
                        ));
                    }
                }

                if self
                    .environment
                    .locate_symbol_in_scope(name, self.environment.active_scope())
                    .is_some()
                {
                    return Err(TypeCheckerError::VariableRedeclaration(name.clone()));
                }

                let constraint = None;
                self.environment.insert_symbol(Symbol::new(
                    name.clone(),
                    ty.clone(),
                    expr.clone(),
                    constraint,
                    Scope::LocalVar,
                ));
            }
            Statement::Return(expr) => match expr {
                Some(Expression::Identifier(name)) => match self.environment.locate_symbol(name) {
                    Some(symbol) => {
                        if let Some(return_ty) = self.environment.scope_return_type() {
                            if symbol.get_type() != return_ty.get_type() {
                                return Err(TypeCheckerError::TypeMismatch(
                                    "return statement".to_string(),
                                ));
                            }
                        }
                    }
                    None => {
                        return Err(TypeCheckerError::VariableNotDeclared(name.clone()));
                    }
                },
                Some(expr) => {
                    if let Some(return_ty) = self.environment.scope_return_type() {
                        if expr.get_type(&self.environment) != return_ty.get_type().clone() {
                            return Err(TypeCheckerError::TypeMismatch(
                                "return statement".to_string(),
                            ));
                        }
                    } else {
                        return Err(TypeCheckerError::InvalidReturnStatement);
                    }
                }
                None => {
                    if let Some(return_ty) = self.environment.scope_return_type() {
                        if return_ty.get_type() != Type::Nil {
                            return Err(TypeCheckerError::TypeMismatch(
                                "return statement".to_string(),
                            ));
                        }
                    }
                }
            },
            Statement::If(expr, true_stmts, false_stmt) => {
                if expr.get_type(&self.environment) != Type::Bool {
                    return Err(TypeCheckerError::ConditionMustBeBool("If".to_string()));
                }

                for stmt in true_stmts {
                    self.check_type_consistency_statement(stmt)?;
                }

                if let Some(false_stmt) = false_stmt {
                    for fstmt in false_stmt {
                        self.check_type_consistency_statement(fstmt)?;
                    }
                }
            }
            Statement::While(expr, stmts) => {
                if expr.get_type(&self.environment) != Type::Bool {
                    return Err(TypeCheckerError::ConditionMustBeBool("While".to_string()));
                }

                for stmt in stmts {
                    self.check_type_consistency_statement(stmt)?;
                }
            }
            Statement::VarAssignment(name, expr) => {
                let symbol = self
                    .environment
                    .locate_symbol(name)
                    .expect("Error: Variable not declared");

                if symbol.get_type() != expr.get_type(&self.environment) {
                    return Err(TypeCheckerError::TypeMismatch(
                        "variable assignment".to_string(),
                    ));
                }
            }
            Statement::ConstDeclaration(name, ty, expr) => {
                // TODO Maybe != is too strict
                if ty != &expr.get_type(&self.environment) {
                    return Err(TypeCheckerError::TypeMismatch(
                        "constant declaration".to_string(),
                    ));
                }

                if self
                    .environment
                    .locate_symbol_in_scope(name, self.environment.active_scope())
                    .is_some()
                {
                    return Err(TypeCheckerError::VariableRedeclaration(name.clone()));
                }

                let constraint = None;
                self.environment.insert_symbol(Symbol::new(
                    name.clone(),
                    ty.clone(),
                    Some(expr.clone()),
                    constraint,
                    Scope::LocalConst,
                ));
            }
            _ => {}
        }

        Ok(())
    }

    fn check_function_constraints<T: Block + Function>(&self, function: &T) {
        let mut constrains = Vec::new();
        for params in function.parameters() {
            if let Some(constraint) = params.constraint() {
                let z3expr = self.expr_to_z3(constraint);
                constrains.push(z3expr);
            }
        }

        for stmt in function.statements() {
            constrains.push(self.stmt_to_z3(stmt));
        }

        if let Some(return_type) = function.return_type() {
            if let Some(constraint) = return_type.constraint() {
                let z3expr = self.expr_to_z3(constraint);
                constrains.push(z3expr);
            }
        }

        let solver = Solver::new(&self.z3_context);
        println!("{:?}", constrains);
        for z3ast in constrains {
            match z3ast {
                AstZ3::Bool(b) => {
                    solver.assert(&b);
                }
                _ => unreachable!("Only bools are supported"),
            }
        }

        let cloned = solver.clone();
        assert_eq!(
            cloned.check(),
            SatResult::Sat,
            "Unsatisfiable constraints in function"
        );

        //constrains
    }

    fn handle_declaration(&self, name: &str, ty: Type, opt_expr: &Option<Expression>) -> AstZ3 {
        match opt_expr {
            Some(expr) => {
                let z3expr = self.expr_to_z3(expr);
                match (ty, z3expr) {
                    (Type::Bool, AstZ3::Bool(b)) => {
                        let var = Bool::new_const(&self.z3_context, name.to_string());
                        AstZ3::Bool(var._eq(&b))
                    }
                    (Type::Int, AstZ3::Int(i)) => {
                        let var = Int::new_const(&self.z3_context, name.to_string());
                        AstZ3::Bool(var._eq(&i))
                    }
                    (Type::String, AstZ3::String(s)) | (Type::Address, AstZ3::String(s)) => {
                        let var = String::new_const(&self.z3_context, name.to_string());
                        AstZ3::Bool(var._eq(&s))
                    }
                    (_, _) => unimplemented!("Type mismatch or unsupported type"),
                }
            }
            None => AstZ3::Bool(Bool::from_bool(&self.z3_context, true)),
        }
    }

    fn stmt_to_z3(&self, stmt: &Statement) -> AstZ3 {
        match stmt {
            Statement::VarDeclaration(name, ty, opt_expr) => {
                self.handle_declaration(name, ty.to_owned(), opt_expr)
            }
            Statement::ConstDeclaration(name, ty, expr) => {
                self.handle_declaration(name, ty.to_owned(), &Some(expr.clone()))
            }
            Statement::VarAssignment(name, expr) => {
                let z3expr = self.expr_to_z3(expr);
                let symbol = self
                    .environment
                    .locate_symbol(name)
                    .expect("Symbol not found");

                match (symbol.get_type(), z3expr) {
                    (Type::Bool, AstZ3::Bool(b)) => {
                        let var = Bool::new_const(&self.z3_context, name.to_string());
                        AstZ3::Bool(var._eq(&b))
                    }
                    (Type::Int, AstZ3::Int(i)) => {
                        let var = Int::new_const(&self.z3_context, name.to_string());
                        AstZ3::Bool(var._eq(&i))
                    }
                    (Type::String, AstZ3::String(s)) | (Type::Address, AstZ3::String(s)) => {
                        let var = String::new_const(&self.z3_context, name.to_string());
                        AstZ3::Bool(var._eq(&s))
                    }
                    _ => unimplemented!("Type mismatch or unsupported type"),
                }
            }
            Statement::If(expr, true_branch, false_branch) => {
                let condition = self.expr_to_z3(expr);
                if let AstZ3::Bool(b) = condition {
                    let mut true_stmts = Bool::from_bool(&self.z3_context, true);
                    for tstmt in true_branch {
                        match self.stmt_to_z3(tstmt) {
                            AstZ3::Bool(b) => {
                                true_stmts = Bool::and(&self.z3_context, &[&true_stmts, &b]);
                            }
                            _ => unimplemented!("Type mismatch"),
                        }
                    }

                    let mut false_stmts = Bool::from_bool(&self.z3_context, true);
                    if let Some(false_branch) = false_branch {
                        for fstmt in false_branch {
                            match self.stmt_to_z3(fstmt) {
                                AstZ3::Bool(b) => {
                                    false_stmts = Bool::and(&self.z3_context, &[&false_stmts, &b]);
                                }
                                _ => unimplemented!("Type mismatch"),
                            }
                        }
                    }

                    AstZ3::Bool(Bool::ite(&b, &true_stmts, &false_stmts))
                } else {
                    unimplemented!("Type mismatch");
                }
            }
            Statement::While(expr, stmts) => {
                //if let Some(constraint) = while_stmts.get_constraint() {
                //    constrains.push(constraint);
                //}
                AstZ3::Bool(Bool::from_bool(&self.z3_context, true))
            }
            Statement::Return(return_stmt) => match return_stmt {
                Some(Expression::Identifier(_)) => {
                    AstZ3::Bool(Bool::from_bool(&self.z3_context, true))
                }
                Some(expr) => {
                    let z3expr = self.expr_to_z3(expr);
                    //z3expr
                    unimplemented!(
                        "Return statement with expression not implemented, result was {:?}",
                        z3expr
                    )
                }
                None => AstZ3::Bool(Bool::from_bool(&self.z3_context, true)),
            },
            Statement::Expression(expr) => self.expr_to_z3(expr),
        }
    }

    fn expr_to_z3(&self, expr: &Expression) -> AstZ3 {
        match expr {
            Expression::BinaryOp(left_expr, bin_op, right_expr) => {
                let left_ast = self.expr_to_z3(left_expr);
                let right_ast = self.expr_to_z3(right_expr);
                println!("{:?} {:?} {:?}", left_ast, bin_op, right_ast);

                match (left_ast, right_ast) {
                    (AstZ3::Bool(left), AstZ3::Bool(right)) => match bin_op {
                        BinaryOperator::And => {
                            AstZ3::Bool(Bool::and(&self.z3_context, &[&left, &right]))
                        }
                        BinaryOperator::Or => {
                            AstZ3::Bool(Bool::or(&self.z3_context, &[&left, &right]))
                        }
                        BinaryOperator::Eq => AstZ3::Bool(left._eq(&right)),
                        BinaryOperator::Ne => AstZ3::Bool(!(left._eq(&right))),
                        //BinaryOperator::Lt => AstZ3::Bool(left.lt(&right)),
                        //BinaryOperator::Le => AstZ3::Bool(left.le(&right)),
                        //BinaryOperator::Gt => AstZ3::Bool(left.gt(&right)),
                        //BinaryOperator::Ge => AstZ3::Bool(left.ge(&right)),
                        _ => unimplemented!("Binary operation not supported for bool types"),
                    },
                    (AstZ3::Int(left), AstZ3::Int(right)) => match bin_op {
                        BinaryOperator::Add => AstZ3::Int(left + right),
                        BinaryOperator::Sub => AstZ3::Int(left - right),
                        BinaryOperator::Mul => AstZ3::Int(left * right),
                        BinaryOperator::Div => AstZ3::Int(left / right),
                        BinaryOperator::Mod => AstZ3::Int(left % right),
                        //BinaryOperator::Exp => left.pow(right),
                        BinaryOperator::Eq => AstZ3::Bool(left._eq(&right)),
                        BinaryOperator::Ne => AstZ3::Bool(!(left._eq(&right))),
                        BinaryOperator::Lt => AstZ3::Bool(left.lt(&right)),
                        BinaryOperator::Le => AstZ3::Bool(left.le(&right)),
                        BinaryOperator::Gt => AstZ3::Bool(left.gt(&right)),
                        BinaryOperator::Ge => AstZ3::Bool(left.ge(&right)),
                        _ => unimplemented!("Binary operation not supported for int types"),
                    },
                    _ => unimplemented!("Binary operation not supported for types"),
                }
            }
            Expression::UnaryOp(op, expr) => {
                let solved = self.expr_to_z3(expr);
                match solved {
                    AstZ3::Bool(b) => match op {
                        UnaryOperator::Not => AstZ3::Bool(b.not()),
                        _ => unimplemented!("Unary operation NEG not supported for bool types"),
                    },
                    AstZ3::Int(i) => match op {
                        UnaryOperator::Neg => AstZ3::Int(i.neg()),
                        _ => unimplemented!("Unary operation NOT not supported for non-bool types"),
                    },
                    _ => unimplemented!("Unary operation not supported for non-bool types"),
                }
            }
            Expression::Literal(literal) => match literal {
                Literal::Bool(b) => AstZ3::Bool(Bool::from_bool(&self.z3_context, *b)),
                Literal::Int(i) => AstZ3::Int(Int::from_i64(&self.z3_context, *i)),
                Literal::String(s) => {
                    AstZ3::String(String::from_str(&self.z3_context, &s).unwrap())
                }
                Literal::Address(a) => {
                    AstZ3::String(String::from_str(&self.z3_context, &a).unwrap())
                }
            },
            Expression::Identifier(name) => {
                let symbol = self.environment.locate_symbol(&name).unwrap();
                match symbol.get_type() {
                    Type::Bool => AstZ3::Bool(Bool::new_const(&self.z3_context, name.clone())),
                    Type::Int => AstZ3::Int(Int::new_const(&self.z3_context, name.clone())),
                    Type::String => {
                        AstZ3::String(String::new_const(&self.z3_context, name.clone()))
                    }
                    Type::Address => {
                        AstZ3::String(String::new_const(&self.z3_context, name.clone()))
                    }
                    _ => unimplemented!("Nil?"),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn test_states() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                
            }
            ",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let env = tc.check_program(&ast).unwrap();

        assert_eq!(env.states(), vec!["Open", "Full", "Locked"]);
    }

    #[test]
    fn test_state_vars() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                state var x: int;
                state var y: int = 10;
                state const z: int = y + 2;

            }
            ",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let env = tc.check_program(&ast).unwrap();

        let symbol_x = env.locate_symbol("x").unwrap();
        let symbol_y = env.locate_symbol("y").unwrap();
        let symbol_z = env.locate_symbol("z").unwrap();

        assert_eq!(symbol_x.get_type(), Type::Int);
        assert_eq!(symbol_y.get_type(), Type::Int);
        assert_eq!(symbol_z.get_type(), Type::Int);

        assert_eq!(symbol_x.scope(), Scope::GlobalVar);
        assert_eq!(symbol_y.scope(), Scope::GlobalVar);
        assert_eq!(symbol_z.scope(), Scope::GlobalConst);

        assert_eq!(symbol_x.value(), None);
        assert_eq!(
            symbol_y.value(),
            Some(Expression::Literal(Literal::Int(10)))
        );
        assert_eq!(
            symbol_z.value(),
            Some(Expression::BinaryOp(
                Box::new(Expression::Identifier("y".to_string())),
                BinaryOperator::Add,
                Box::new(Expression::Literal(Literal::Int(2)))
            ))
        );
    }

    #[test]
    fn test_invalid_state_vars() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                state var x: int;
                state var y: int = 10;
                state var z: bool = y + 2;

            }
            ",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let err = tc.check_program(&ast).unwrap_err();

        assert_eq!(
            err,
            TypeCheckerError::TypeMismatch("state data declaration".to_string())
        );
    }

    #[test]
    fn test_constructor() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor({x: int}, {y: int}) {
                    x = y;
                }
            }
            ",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let env = tc.check_program(&ast).unwrap();

        let symbol_x = env.locate_symbol("x").unwrap();
        let symbol_y = env.locate_symbol("y").unwrap();

        assert_eq!(symbol_x.get_type(), Type::Int);
        assert_eq!(symbol_y.get_type(), Type::Int);

        assert_eq!(symbol_x.scope(), Scope::Param);
        assert_eq!(symbol_y.scope(), Scope::Param);
    }

    #[test]
    fn test_invalid_constructor() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {
                constructor({x: int}, {y: bool}) {
                    x = y;
                }
            }
            ",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let err = tc.check_program(&ast).unwrap_err();

        assert_eq!(
            err,
            TypeCheckerError::TypeMismatch("variable assignment".to_string())
        );
    }

    #[test]
    fn test_function() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {

            }

            Vault@Open(owner) {
                function deposit({x: int}, {y: int}) {
                    x = y;
                }
            }
            ",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let env = tc.check_program(&ast).unwrap();

        let symbol_x = env.locate_symbol_in_scope("x", 1).unwrap();
        let symbol_y = env.locate_symbol_in_scope("y", 1).unwrap();

        assert_eq!(symbol_x.get_type(), Type::Int);
        assert_eq!(symbol_y.get_type(), Type::Int);

        assert_eq!(symbol_x.scope(), Scope::Param);
        assert_eq!(symbol_y.scope(), Scope::Param);
    }

    #[test]
    fn test_invalid_function() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {

            }

            Vault@Open(owner) {
                function deposit({x: int}, {y: bool}) {
                    x = y;
                }
            }
            ",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let err = tc.check_program(&ast).unwrap_err();

        assert_eq!(
            err,
            TypeCheckerError::TypeMismatch("variable assignment".to_string())
        );
    }

    #[test]
    fn test_function_invalid_state() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {

            }

            Vault@Invalid(owner) {
                function deposit({x: int}, {y: int}) {
                    x = y;
                }
            }
            ",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let err = tc.check_program(&ast).unwrap_err();

        assert_eq!(
            err,
            TypeCheckerError::StateNotDeclared("Invalid".to_string())
        );
    }

    #[test]
    fn test_function_constraints() {
        let ast = parse(
            "
            contract Vault[Open, Full, Locked] {

            }

            Vault@Locked(owner) {
                function deposit({x: int | x > 10}, {y: int | y < 10}) -> {z: int | z >= x + y} {
                    var z: int = x + y;
                    return z;
                }
            }
            ",
        )
        .unwrap();

        let mut tc = TypeChecker::new();
        let env = tc.check_program(&ast).unwrap();

        let symbol_x = env.locate_symbol_in_scope("x", 1).unwrap();
        let symbol_y = env.locate_symbol_in_scope("y", 1).unwrap();
        let symbol_z = env.locate_symbol_in_scope("z", 1).unwrap();
        let symbol_ret = env.locate_return_type_in_scope(1).unwrap();

        assert_eq!(symbol_x.get_type(), Type::Int);
        assert_eq!(symbol_y.get_type(), Type::Int);
        assert_eq!(symbol_z.get_type(), Type::Int);
        assert_eq!(symbol_ret.get_type(), Type::Int);

        assert_eq!(symbol_x.scope(), Scope::Param);
        assert_eq!(symbol_y.scope(), Scope::Param);
        assert_eq!(symbol_z.scope(), Scope::LocalVar);
        //assert_eq!(symbol_ret.scope(), Scope::Return);

        let symbol_z_value = symbol_z.value().unwrap();
        assert_eq!(
            symbol_z_value,
            Expression::BinaryOp(
                Box::new(Expression::Identifier("x".to_string())),
                BinaryOperator::Add,
                Box::new(Expression::Identifier("y".to_string()))
            )
        );

        assert_eq!(symbol_ret.name(), "z");
        let symbol_ret_constraint = symbol_ret.constraint().clone().unwrap();
        assert_eq!(
            symbol_ret_constraint,
            Expression::BinaryOp(
                Box::new(Expression::Identifier("z".to_string())),
                BinaryOperator::Ge,
                Box::new(Expression::BinaryOp(
                    Box::new(Expression::Identifier("x".to_string())),
                    BinaryOperator::Add,
                    Box::new(Expression::Identifier("y".to_string()))
                ))
            )
        );

        let symbol_x_constraint = symbol_x.constraint().unwrap();
        assert_eq!(
            symbol_x_constraint,
            Expression::BinaryOp(
                Box::new(Expression::Identifier("x".to_string())),
                BinaryOperator::Gt,
                Box::new(Expression::Literal(Literal::Int(10)))
            )
        );

        let symbol_y_constraint = symbol_y.constraint().unwrap();
        assert_eq!(
            symbol_y_constraint,
            Expression::BinaryOp(
                Box::new(Expression::Identifier("y".to_string())),
                BinaryOperator::Lt,
                Box::new(Expression::Literal(Literal::Int(10)))
            )
        );
    }
}

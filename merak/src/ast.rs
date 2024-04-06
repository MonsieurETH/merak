use std::collections::HashMap;

use crate::environment::Environment;

#[derive(Debug)]
pub struct Program {
    contract: Vec<Contract>,
    states_def: Vec<ContractState>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            contract: Vec::new(),
            states_def: Vec::new(),
        }
    }

    pub fn add_contract(&mut self, contract: Contract) {
        self.contract.push(contract);
    }

    pub fn contracts(&self) -> &Vec<Contract> {
        &self.contract
    }

    pub fn add_state_def(&mut self, state_def: ContractState) {
        self.states_def.push(state_def);
    }

    pub fn state_defs(&self) -> &Vec<ContractState> {
        &self.states_def
    }
}

#[derive(Debug)]
pub struct Contract {
    name: String,
    state_data: Vec<ContractStateData>,
    constructor: Option<ContractConstructor>,
    states: Vec<String>,
}

impl Contract {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            state_data: Vec::new(),
            constructor: None,
            states: Vec::new(),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn add_state(&mut self, state: String) {
        self.states.push(state);
    }

    pub fn states(&self) -> &Vec<String> {
        &self.states
    }

    pub fn add_state_data(&mut self, state_data: ContractStateData) {
        self.state_data.push(state_data);
    }

    pub fn state_data(&self) -> &Vec<ContractStateData> {
        &self.state_data
    }

    pub fn add_constructor(&mut self, constructor: ContractConstructor) {
        self.constructor = Some(constructor);
    }

    pub fn constructor(&self) -> &Option<ContractConstructor> {
        &self.constructor
    }
}

#[derive(Debug)]
pub struct ContractStateData {
    name: String,
    ty: Type,
    constant: bool,
    value: Option<Expression>,
}

impl ContractStateData {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            ty: Type::Nil,
            constant: false,
            value: None,
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn set_type(&mut self, ty: Type) {
        self.ty = ty;
    }

    pub fn get_type(&self) -> &Type {
        &self.ty
    }

    pub fn set_constant(&mut self, constant: bool) {
        self.constant = constant;
    }

    pub fn is_constant(&self) -> bool {
        self.constant
    }

    pub fn set_value(&mut self, value: Expression) {
        self.value = Some(value);
    }

    pub fn value(&self) -> &Option<Expression> {
        &self.value
    }
}

#[derive(Debug)]
pub struct ContractState {
    name: String,
    contract: String,
    convenor: String,
    functions: Vec<ContractFunction>,
}

impl ContractState {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            contract: String::new(),
            convenor: String::new(),
            functions: Vec::new(),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn set_contract(&mut self, contract: String) {
        self.contract = contract;
    }

    pub fn contract(&self) -> &String {
        &self.contract
    }

    pub fn set_convenor(&mut self, convenor: String) {
        self.convenor = convenor;
    }

    pub fn convenor(&self) -> &String {
        &self.convenor
    }

    pub fn add_function(&mut self, function: ContractFunction) {
        self.functions.push(function);
    }

    pub fn functions(&self) -> &Vec<ContractFunction> {
        &self.functions
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Nil,
    Int,
    Bool,
    String,
    Address,
}

impl From<&str> for Type {
    fn from(s: &str) -> Self {
        match s {
            "int" => Self::Int,
            "bool" => Self::Bool,
            "string" => Self::String,
            "address" => Self::Address,
            _ => Self::Nil,
        }
    }
}

pub trait Function {
    fn parameters(&self) -> &Vec<FunctionParameter>;
    fn return_type(&self) -> &Option<FunctionParameter>;
}

pub trait Block {
    fn statements(&self) -> &Vec<Statement>;
    fn add_statements(&mut self, statements: &mut Vec<Statement>);
}

impl Block for Vec<Statement> {
    fn statements(&self) -> &Vec<Statement> {
        self
    }

    fn add_statements(&mut self, statements: &mut Vec<Statement>) {
        self.append(statements)
    }
}

#[derive(Debug)]
pub struct ContractConstructor {
    parameters: Vec<FunctionParameter>,
    body: Vec<Statement>,
    return_type: Option<FunctionParameter>,
}

impl Function for ContractConstructor {
    fn parameters(&self) -> &Vec<FunctionParameter> {
        &self.parameters
    }

    fn return_type(&self) -> &Option<FunctionParameter> {
        &self.return_type
    }
}

impl Block for ContractConstructor {
    fn statements(&self) -> &Vec<Statement> {
        &self.body
    }

    fn add_statements(&mut self, statements: &mut Vec<Statement>) {
        self.body.append(statements);
    }
}

impl ContractConstructor {
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
            body: Vec::new(),
            return_type: None,
        }
    }

    pub fn add_parameter(&mut self, parameter: FunctionParameter) {
        self.parameters.push(parameter);
    }
}

#[derive(Debug)]
pub struct ContractFunction {
    name: String,
    parameters: Vec<FunctionParameter>,
    visibility: Visibility,
    modifiers: Vec<Modifier>,
    return_type: Option<FunctionParameter>,
    body: Vec<Statement>,
}

impl Function for ContractFunction {
    fn parameters(&self) -> &Vec<FunctionParameter> {
        &self.parameters
    }

    fn return_type(&self) -> &Option<FunctionParameter> {
        &self.return_type
    }
}

impl Block for ContractFunction {
    fn statements(&self) -> &Vec<Statement> {
        &self.body
    }
    fn add_statements(&mut self, statements: &mut Vec<Statement>) {
        self.body.append(statements);
    }
}

impl ContractFunction {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            parameters: Vec::new(),
            visibility: Visibility::Function,
            modifiers: Vec::new(),
            return_type: None,
            body: Vec::new(),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn set_visibility(&mut self, visibility: Visibility) {
        self.visibility = visibility;
    }

    pub fn visibility(&self) -> &Visibility {
        &self.visibility
    }

    pub fn add_modifier(&mut self, modifier: Modifier) {
        self.modifiers.push(modifier);
    }

    pub fn modifiers(&self) -> &Vec<Modifier> {
        &self.modifiers
    }

    pub fn add_parameter(&mut self, parameter: FunctionParameter) {
        self.parameters.push(parameter);
    }

    pub fn set_return_type(&mut self, return_type: FunctionParameter) {
        self.return_type = Some(return_type);
    }
}

#[derive(Debug, PartialEq)]
pub enum Visibility {
    Entrypoint,
    Function,
}

impl From<&str> for Visibility {
    fn from(s: &str) -> Self {
        match s {
            "entrypoint" => Self::Entrypoint,
            "function" => Self::Function,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Modifier {
    Stateful,
    Payable,
}

impl From<&str> for Modifier {
    fn from(s: &str) -> Self {
        match s {
            "stateful" => Self::Stateful,
            "payable" => Self::Payable,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct FunctionParameter {
    name: String,
    ty: Type,
    constraint: Option<Expression>,
}

impl FunctionParameter {
    pub fn new() -> Self {
        Self {
            name: String::new(),
            ty: Type::Nil,
            constraint: None,
        }
    }

    pub fn create(name: String, ty: Type, constraint: Option<Expression>) -> Self {
        Self {
            name,
            ty,
            constraint,
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    pub fn name(&self) -> &String {
        &self.name
    }

    pub fn set_type(&mut self, ty: Type) {
        self.ty = ty;
    }

    pub fn get_type(&self) -> Type {
        self.ty.clone()
    }

    pub fn set_constraint(&mut self, constraint: Expression) {
        self.constraint = Some(constraint);
    }

    pub fn constraint(&self) -> &Option<Expression> {
        &self.constraint
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Expression(Expression),
    VarAssignment(String, Expression),
    VarDeclaration(String, Type, Option<Expression>),
    ConstDeclaration(String, Type, Expression),
    Return(Option<Expression>),
    If(Expression, Vec<Statement>, Option<Vec<Statement>>),
    While(Expression, Vec<Statement>),
}

impl Statement {
    pub fn defines_variable(&self, var: &str) -> bool {
        match self {
            Self::Expression(_) => false,
            Self::VarAssignment(name, _) => name == var,
            Self::VarDeclaration(name, _, _) => name == var,
            Self::ConstDeclaration(name, _, _) => name == var,
            Self::Return(_) => false,
            Self::If(_, then_block, else_block) => {
                then_block.iter().any(|stmt| stmt.defines_variable(var))
                    || else_block
                        .as_ref()
                        .map(|block| block.iter().any(|stmt| stmt.defines_variable(var)))
                        .unwrap_or(false)
            }
            Self::While(_, block) => block.iter().any(|stmt| stmt.defines_variable(var)),
        }
    }

    pub fn used_vars(&self) -> Vec<String> {
        match self {
            Self::Expression(expr) => expr.get_vars(),
            Self::VarAssignment(_, expr) => expr.get_vars(),
            Self::VarDeclaration(_, _, Some(expr)) => expr.get_vars(),
            Self::VarDeclaration(_, _, None) => Vec::new(),
            Self::ConstDeclaration(_, _, expr) => expr.get_vars(),
            Self::Return(Some(expr)) => expr.get_vars(),
            Self::Return(None) => Vec::new(),
            Self::If(cond, then_block, else_block) => {
                let mut vars = cond.get_vars();
                for stmt in then_block {
                    vars.append(&mut stmt.used_vars());
                }
                if let Some(else_block) = else_block {
                    for stmt in else_block {
                        vars.append(&mut stmt.used_vars());
                    }
                }
                vars
            }
            Self::While(cond, block) => {
                let mut vars = cond.get_vars();
                for stmt in block {
                    vars.append(&mut stmt.used_vars());
                }
                vars
            }
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Expression {
    Literal(Literal),
    BinaryOp(Box<Expression>, BinaryOperator, Box<Expression>),
    UnaryOp(UnaryOperator, Box<Expression>),
    //FunctionCall(String, Vec<Expression>),
    Identifier(String),
}

impl Expression {
    pub fn binary_op(left: Expression, op: BinaryOperator, right: Expression) -> Self {
        Self::BinaryOp(Box::new(left), op, Box::new(right))
    }

    pub fn unary_op(op: UnaryOperator, expr: Expression) -> Self {
        Self::UnaryOp(op, Box::new(expr))
    }

    pub fn identifier(name: String) -> Self {
        Self::Identifier(name)
    }

    // Get all variables (identifiers) used in the expression
    pub fn get_vars(&self) -> Vec<String> {
        match self {
            Self::Literal(_) => Vec::new(),
            Self::BinaryOp(left, _, right) => {
                let mut vars = left.get_vars();
                vars.append(&mut right.get_vars());
                vars
            }
            Self::UnaryOp(_, expr) => expr.get_vars(),
            Self::Identifier(name) => vec![name.clone()],
        }
    }

    // Index variables in the expression using the provided mapping <var_name, index>
    pub fn index_vars(&self, vars: &HashMap<String, usize>) -> Self {
        match self {
            Self::Literal(literal) => Self::Literal(literal.clone()),
            Self::BinaryOp(left, op, right) => Self::BinaryOp(
                Box::new(left.index_vars(vars)),
                op.clone(),
                Box::new(right.index_vars(vars)),
            ),
            Self::UnaryOp(op, expr) => Self::UnaryOp(op.clone(), Box::new(expr.index_vars(vars))),
            Self::Identifier(name) => {
                if let Some(index) = vars.get(name) {
                    Self::Identifier(format!("{}_{}", name, index))
                } else {
                    unreachable!("Variable {} not found in vars", name)
                }
            }
        }
    }

    pub fn get_type(&self, environment: &Environment) -> Type {
        match self {
            Self::Literal(Literal::Int(_)) => Type::Int,
            Self::Literal(Literal::Bool(_)) => Type::Bool,
            Self::Literal(Literal::String(_)) => Type::String,
            Self::Literal(Literal::Address(_)) => Type::Address,
            Self::BinaryOp(left, op, right) => {
                let ty_left = left.get_type(environment);
                let ty_right = right.get_type(environment);
                // TODO basic type checking, need to be improved
                match op {
                    BinaryOperator::Add | BinaryOperator::Sub | BinaryOperator::Mul => {
                        if ty_left == Type::Int && ty_right == Type::Int {
                            Type::Int
                        } else {
                            Type::Nil
                        }
                    }
                    BinaryOperator::Div | BinaryOperator::Mod => {
                        if ty_left == Type::Int && ty_right == Type::Int {
                            Type::Int
                        } else {
                            Type::Nil
                        }
                    }
                    BinaryOperator::Eq | BinaryOperator::Ne => {
                        if ty_left == ty_right {
                            Type::Bool
                        } else {
                            Type::Nil
                        }
                    }
                    BinaryOperator::Lt
                    | BinaryOperator::Le
                    | BinaryOperator::Gt
                    | BinaryOperator::Ge => {
                        if ty_left == Type::Int && ty_right == Type::Int {
                            Type::Bool
                        } else {
                            Type::Nil
                        }
                    }
                    BinaryOperator::And | BinaryOperator::Or => {
                        if ty_left == Type::Bool && ty_right == Type::Bool {
                            Type::Bool
                        } else {
                            Type::Nil
                        }
                    }
                }
            }
            Self::UnaryOp(op, expr) => {
                let ty = expr.get_type(environment);
                match op {
                    UnaryOperator::Neg => {
                        if ty == Type::Int {
                            Type::Int
                        } else {
                            Type::Nil
                        }
                    }
                    UnaryOperator::Not => {
                        if ty == Type::Bool {
                            Type::Bool
                        } else {
                            Type::Nil
                        }
                    }
                }
            }
            Self::Identifier(name) => {
                if let Some(symbol) = environment.locate_symbol(name) {
                    symbol.get_type()
                } else {
                    Type::Nil
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
}

impl From<&str> for BinaryOperator {
    fn from(s: &str) -> Self {
        match s {
            "+" => Self::Add,
            "-" => Self::Sub,
            "*" => Self::Mul,
            "/" => Self::Div,
            "%" => Self::Mod,
            "==" => Self::Eq,
            "!=" => Self::Ne,
            "<" => Self::Lt,
            "<=" => Self::Le,
            ">" => Self::Gt,
            ">=" => Self::Ge,
            "&&" => Self::And,
            "||" => Self::Or,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum UnaryOperator {
    Neg,
    Not,
}

impl From<&str> for UnaryOperator {
    fn from(s: &str) -> Self {
        match s {
            "-" => Self::Neg,
            "!" => Self::Not,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, PartialEq, Clone, Eq, Hash)]
pub enum Literal {
    Int(i64),
    Bool(bool),
    String(String),
    Address(String),
}

impl From<&str> for Literal {
    fn from(s: &str) -> Self {
        if s.starts_with("0x") {
            Self::Address(s.to_string())
        } else if s == "true" {
            Self::Bool(true)
        } else if s == "false" {
            Self::Bool(false)
        } else if s.starts_with('"') && s.ends_with('"') {
            Self::String(s.to_string())
        } else {
            Self::Int(s.parse().unwrap())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_vars() {
        let expr = Expression::binary_op(
            Expression::identifier("a".to_string()),
            BinaryOperator::Add,
            Expression::binary_op(
                Expression::identifier("b".to_string()),
                BinaryOperator::Mul,
                Expression::identifier("c".to_string()),
            ),
        );
        assert_eq!(expr.get_vars(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_get_vars_2() {
        let expr = Expression::binary_op(
            Expression::binary_op(
                Expression::Literal(Literal::Int(1)),
                BinaryOperator::Add,
                Expression::unary_op(
                    UnaryOperator::Neg,
                    Expression::binary_op(
                        Expression::unary_op(
                            UnaryOperator::Neg,
                            Expression::Identifier("a".to_string()),
                        ),
                        BinaryOperator::Add,
                        Expression::Identifier("b".to_string()),
                    ),
                ),
            ),
            BinaryOperator::Mul,
            Expression::identifier("c".to_string()),
        );
        assert_eq!(expr.get_vars(), vec!["a", "b", "c"]);
    }
}

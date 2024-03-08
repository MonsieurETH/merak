use crate::ast::{Expression, FunctionParameter, Type};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Debug)]
pub struct Environment {
    symbol_tables: Vec<SymbolTable>,
    active_table: usize,
    states: Vec<String>,
    functions: HashMap<String, usize>,
}

impl Environment {
    pub fn new() -> Self {
        Self {
            symbol_tables: vec![SymbolTable::new(None)],
            active_table: 0,
            states: Vec::new(),
            functions: HashMap::new(),
        }
    }

    pub fn push_function(&mut self, name: String) {
        let index = self.push_scope();
        self.functions.insert(name, index);
    }

    pub fn push_scope(&mut self) -> usize {
        let new_pos = self.symbol_tables.len();
        self.symbol_tables
            .push(SymbolTable::new(Some(self.active_table)));
        self.active_table = new_pos;

        new_pos
    }

    pub fn pop_scope(&mut self) {
        let new_active = self.symbol_tables[self.active_table].enclosing.unwrap_or(0);
        self.active_table = new_active;
    }

    pub fn insert_symbol(&mut self, symbol: Symbol) {
        self.symbol_tables[self.active_table].insert(symbol.name.clone(), symbol);
    }

    pub fn locate_symbol(&self, name: &str) -> Option<&Symbol> {
        let symbol = self.symbol_tables[self.active_table].retrieve(name);
        if symbol.is_none() {
            let mut scope = self.symbol_tables[self.active_table].enclosing;
            while scope.is_some() {
                let symbol = self.symbol_tables[scope.unwrap()].retrieve(name);
                if symbol.is_some() {
                    return symbol;
                }
                scope = self.symbol_tables[scope.unwrap()].enclosing;
            }
            return None;
        }

        symbol
    }

    pub fn locate_symbol_in_scope(&self, name: &str, scope: usize) -> Option<&Symbol> {
        self.symbol_tables[scope].retrieve(name)
    }

    pub fn locate_return_type_in_scope(&self, scope: usize) -> Option<FunctionParameter> {
        self.symbol_tables[scope].return_type()
    }

    pub fn insert_state(&mut self, state: String) {
        self.states.push(state);
    }

    pub fn set_scope_return_symbol(&mut self, return_type: FunctionParameter) {
        self.symbol_tables[self.active_table].set_return_type(return_type);
    }

    pub fn active_scope(&self) -> usize {
        self.active_table
    }

    pub fn scope_return_type(&self) -> Option<FunctionParameter> {
        self.symbol_tables[self.active_table].return_type()
    }

    pub fn locate_state(&self, state: &str) -> bool {
        self.states.contains(&state.to_string())
    }

    pub fn states(&self) -> Vec<String> {
        self.states.clone()
    }
}

#[derive(Clone, Debug)]
pub struct SymbolTable {
    table: BTreeMap<String, Symbol>,
    enclosing: Option<usize>,
    // This shouldn't be called FunctionParameter, maybe NamedParameter, NamedType or something like that
    return_type: Option<FunctionParameter>,
}

impl SymbolTable {
    pub fn new(enclosing: Option<usize>) -> Self {
        Self {
            table: BTreeMap::new(),
            enclosing,
            return_type: None,
        }
    }

    pub fn insert(&mut self, name: String, symbol: Symbol) {
        self.table.insert(name, symbol);
    }

    pub fn retrieve(&self, name: &str) -> Option<&Symbol> {
        self.table.get(name)
    }

    pub fn enclosing(&self) -> Option<usize> {
        self.enclosing
    }

    pub fn return_type(&self) -> Option<FunctionParameter> {
        self.return_type.clone()
    }

    pub fn set_return_type(&mut self, return_type: FunctionParameter) {
        self.return_type = Some(return_type);
    }
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub symbol_type: Type,
    pub value: Option<Expression>,
    pub constraint: Option<Expression>,
    pub scope: Scope,
}

impl Symbol {
    pub fn new(
        name: String,
        symbol_type: Type,
        value: Option<Expression>,
        constraint: Option<Expression>,
        scope: Scope,
    ) -> Self {
        Self {
            name,
            symbol_type,
            value,
            constraint,
            scope,
        }
    }

    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn get_type(&self) -> Type {
        self.symbol_type.clone()
    }

    pub fn constraint(&self) -> Option<Expression> {
        self.constraint.clone()
    }

    pub fn value(&self) -> Option<Expression> {
        self.value.clone()
    }

    pub fn scope(&self) -> Scope {
        self.scope.clone()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Scope {
    Param,
    LocalVar,
    LocalConst,
    GlobalVar,
    GlobalConst,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment() {
        let mut env = Environment::new();
        let symbol = Symbol::new(
            "x".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(10))),
            None,
            Scope::LocalVar,
        );
        env.insert_symbol(symbol.clone());
        let symbol = env.locate_symbol("x").unwrap();
        assert_eq!(symbol.name(), "x");
        assert_eq!(symbol.get_type(), Type::Int);
        assert_eq!(
            symbol.value().unwrap(),
            Expression::Literal(crate::ast::Literal::Int(10))
        );
    }

    #[test]
    fn test_environment_scope() {
        let mut env = Environment::new();
        let symbol = Symbol::new(
            "x".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(10))),
            None,
            Scope::LocalVar,
        );
        env.insert_symbol(symbol.clone());

        env.push_scope();
        let symbol = Symbol::new(
            "x".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(20))),
            None,
            Scope::LocalVar,
        );
        env.insert_symbol(symbol.clone());

        let symbol = env.locate_symbol("x").unwrap();
        assert_eq!(symbol.name(), "x");
        assert_eq!(symbol.get_type(), Type::Int);
        assert_eq!(
            symbol.value().unwrap(),
            Expression::Literal(crate::ast::Literal::Int(20))
        );
        env.pop_scope();

        let symbol = env.locate_symbol("x").unwrap();
        assert_eq!(symbol.name(), "x");
        assert_eq!(symbol.get_type(), Type::Int);
        assert_eq!(
            symbol.value().unwrap(),
            Expression::Literal(crate::ast::Literal::Int(10))
        );
    }

    #[test]
    fn test_environment_scope_return_type() {
        let mut env = Environment::new();
        let symbol = Symbol::new(
            "y".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(10))),
            None,
            Scope::LocalVar,
        );
        env.insert_symbol(symbol.clone());

        env.push_scope();
        let symbol = Symbol::new(
            "x".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(20))),
            None,
            Scope::LocalVar,
        );
        env.insert_symbol(symbol.clone());

        env.set_scope_return_symbol(FunctionParameter::create(
            "x".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(20))),
        ));

        let return_type = env.scope_return_type().unwrap();
        assert_eq!(return_type.name(), "x");
        assert_eq!(return_type.get_type(), Type::Int);
        assert_eq!(
            return_type.constraint().clone().unwrap(),
            Expression::Literal(crate::ast::Literal::Int(20))
        );
        env.pop_scope();
    }

    #[test]
    fn test_environment_state() {
        let mut env = Environment::new();
        env.insert_state("state1".to_string());
        env.insert_state("state2".to_string());
        assert_eq!(env.locate_state("state1"), true);
        assert_eq!(env.locate_state("state2"), true);
        assert_eq!(env.locate_state("state3"), false);
    }

    #[test]
    fn test_retrieve_from_upper_env() {
        let mut env = Environment::new();
        let symbol = Symbol::new(
            "x".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(10))),
            None,
            Scope::LocalVar,
        );
        env.insert_symbol(symbol.clone());

        env.push_scope();
        let symbol = Symbol::new(
            "y".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(20))),
            None,
            Scope::LocalVar,
        );
        env.insert_symbol(symbol.clone());

        let symbol = env.locate_symbol("x").unwrap();
        assert_eq!(symbol.name(), "x");
        assert_eq!(symbol.get_type(), Type::Int);
        assert_eq!(
            symbol.value().unwrap(),
            Expression::Literal(crate::ast::Literal::Int(10))
        );
    }

    #[test]
    fn test_pop_and_push_scope() {
        let mut env = Environment::new();
        let symbol = Symbol::new(
            "x".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(10))),
            None,
            Scope::LocalVar,
        );
        env.insert_symbol(symbol.clone());

        env.push_scope();
        let symbol = Symbol::new(
            "y".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(20))),
            None,
            Scope::LocalVar,
        );
        env.insert_symbol(symbol.clone());

        env.pop_scope();
        env.push_scope();
        let symbol = Symbol::new(
            "z".to_string(),
            Type::Int,
            Some(Expression::Literal(crate::ast::Literal::Int(30))),
            None,
            Scope::LocalVar,
        );
        env.insert_symbol(symbol.clone());

        env.push_scope();
        let symbol = env.locate_symbol("z").unwrap();
        assert_eq!(symbol.name(), "z");
        assert_eq!(symbol.get_type(), Type::Int);
        assert_eq!(
            symbol.value().unwrap(),
            Expression::Literal(crate::ast::Literal::Int(30))
        );

        let none_y = env.locate_symbol_in_scope("y", env.active_scope());
        assert!(none_y.is_none());

        env.pop_scope();
        env.pop_scope();
        let symbol_y = env.locate_symbol("y");
        assert!(symbol_y.is_none());
        let symbol_z = env.locate_symbol("z");
        assert!(symbol_z.is_none());

        env.push_scope();
        let symbol = env.locate_symbol("x").unwrap();
        assert_eq!(symbol.name(), "x");
        assert_eq!(symbol.get_type(), Type::Int);
        assert_eq!(
            symbol.value().unwrap(),
            Expression::Literal(crate::ast::Literal::Int(10))
        );
    }
}

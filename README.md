# Merak

**Attention: Merak is still in alpha development, and is not ready to be used in production (don't worry, it does not produce any bytecode yet). Breaking changes are expected!**

Merak is an experimental programming language for the EVM based on [Liquid Types](https://goto.ucsd.edu/~rjhala/liquid/liquid_types.pdf) (Logically Qualified Data Types) and [Typestate](https://ieeexplore.ieee.org/document/6312929), linear types to represent assets and a system of mandatory users and privileges.


The main goal of Merak, apart from being a side project that I find really fun, is to explore ideas that can improve the reliability of smart contracts in EVM. It's to be expected that the code will be messy, not idiomatically correct and not performant.

### Liquid types

Liquid types are an extension of traditional type systems that allow, through the use of logical predicates, to automatically specify and verify some semantic properties of the code. In addition to specifying the structure of the elements of a program (int, bool, etc.), liquid types allow, using a reduced logical language, to specify part of the semantics of a program. For example, using Merak's syntax, we could have a function that not only specifies the structure of its parameters and the value it returns, but also allows us to add its own semantics (yes, the example is not really awesome, I know).

```
function sum({x: int | x > 10}, {y: int | y < 10}) -> {z: int | z >= x + y} {
  var z: int = x + y;
  return z;
}
```

While it's true that writing liquid types seems much more cumbersome, it brings some interesting advantages:
1) It forces the programmer to think through each of his functions in more detail.
2) It makes it easier to share code with someone else, because you add part of the function specification into it.
3) And the most important advantage, it allows us to use theorem provers like Z3 to make sure that the function complies with this specification at compile time, warning the programmer of a number of errors that would not be detected otherwise.

Furthermore, the use of liquid types is optional and one could write this same function as follows:

```
function sum({x: int}, {y: int}) -> {z: int} {
  var z: int = x + y;
  return z;
}
```

### Typestate

The idea of typestate is to refine the concept of type by adding context to it. While the type of an object determines the set of operations that can be applied to it, typestate allows to determine a set of operations that are allowed or disallowed based on the context of the contract.

```
contract Vault[Open, Full, Locked] {
  state var owner: address;
  state var total: int = 0;
}

Vault@Open(owner) {
  function deposit({amount: int}) {
      total = amount;
      if total > 10 {
        become Full;
      }
  }

  function lock() {
    become Locked;
  }
}

Vault@Locked(owner) {
  function unlock() {
    if total < 10 {
      become Full;
    } else {
      become Open;
    }
  }
}
```


### Transformations & Optimizations already implemented
- [x] Control flow graph
- [x] SSA

- [X] Local value numbering
  - [x] Dead code elimination (DCE) 
  - [x] Copy propagation
  - [x] Constant propagation
  - [x] Common subexpression elimination (CSE) with commutativity  

### Contributing

Merak is a 100% experimental project and code contributions are not welcome at this stage. However, if you want to support this project in any way, you can buy me a beer by sending ETH to this address: 0x7FF4408Bf503Cdd3991771a18E8F8C364eACE215

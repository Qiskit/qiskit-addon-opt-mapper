# Qiskit addon: optimization modeling

### Table of contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Computational requirements](#computational-requirements)
* [Deprecation Policy](#deprecation-policy)
* [Contributing](#contributing)
* [License](#license)
* [References](#references)

----------------------------------------------------------------------------------------------------

### About

[Qiskit addons](https://quantum.cloud.ibm.com/docs/guides/addons) are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains the Qiskit addon for optimization modeling.
Quantum computers have the potential to solve combinatorial optimization problems [1].
These optimization problems can be formulated in an abstract model and then converted into a representation that a quantum computer can understand, for instance a Hamiltonian operator.

The optimization workflow first involves formulating the optimization problem in mathematical terms.
This requires defining the objective function to either maximize or minimize and adding any constraints that the decision variables must satisfy.
The variables may be continuous, integare, binary or spin-like.
Furthermore, the constraints on the variables are typically formulated as equalities and inequalities.
Second, this mathematical model is often reformulated into an unconstrained form by transforming the constraints into penalty terms.
Third, the decision variables may be converted into a desired format.
For example, integer variables may the transformed into binary variables so that the resulting model is either quadratic unconstrained binary optimization (QUBO) problem or a higher-order unconstrained binary optimization (HUBO) problem.
In this package the term higher-order is used to designed any polynomial or monomial with a degree higher than two.
Finally, the model is translated into a format that a quantum computer can understand.
Typically, this implies creating a Hamiltonian operator whose ground state corresponds to the solution of the original optimization problem.

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at [https://qiskit.github.io/qiskit-addon-opt-mapper](https://qiskit.github.io/qiskit-addon-opt-mapper)

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible:

```bash
pip install qiskit-addon-opt-mapper
```

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Computational requirements

The most computationally expensive part of the addon is the computation of the objective function for large combinatorial optimization problems.

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
[release notes](https://qiskit.github.io/qiskit-addon-opt-mapper/release-notes.html).

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-addon-opt-mapper).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-addon-opt-mapper/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-addon-opt-mapper/issues/new/choose) for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)

### References

[1] Abbas, et al. [Challenges and opportunities in quantum optimization](https://www.nature.com/articles/s42254-024-00770-9), Nat. Rev. Physics **6**, 718-735 (2024).

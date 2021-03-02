# Sample Flux Layer

This example demonstrates the creation of a sample Flux layer.

```@meta
CurrentModule = Qaintellect
end
```

#### Example
This example shows test code that uses the `Flux.gradient()` call to calculate the gradients of the `Circuit` object.

```@example
using Flux
using Qaintessent
using Qaintellect
using LinearAlgebra
using IterTools: ncycle

# construct parametrized circuit
rx = RxGate(2π*rand())
ry = RyGate(2π*rand())

cgc = [
    circuit_gate(1, rx),
    circuit_gate(1, ry),
]

# using Pauli-Z matrix as observable
meas = [MeasurementOperator([1 0; 0 -1], (1,))]

c = Circuit{1}(cgc, meas)

# We set the initial input as the $\lvert 0 \rangle$ state, and the target expectation value of the measurement as $-1$.
ψ = ComplexF64[1, 0]
e = -1

# create loss function: note that circuit `c` is applied to `x`
loss(x, y) = Flux.mse(c(x), y)

# gather parameters from Circuit
paras = Flux.params(c)
@show(paras)

# define optimizer
opt = ADAM(0.5)

# set up data for training; using `ncycle()` to repeatedly feed the input quantum state into the training algorithm
data = ncycle([(ψ, e)], 128)

# define evaluation function
evalcb() = @show(loss(ψ, e))

# example: compute gradients
grads = gradient(() -> loss(ψ, e), paras)
grads[ry.θ]

Flux.train!(loss, paras, data, opt, cb=Flux.throttle(evalcb, 0.01))

# verify output
@show apply(c, ψ)
```

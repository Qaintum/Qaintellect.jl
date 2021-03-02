# Flux Integration

Qaintellect.jl integrates the various utilities of Qaintessent.jl into Flux.jl. This allows the creation of trainable layers, which can integrated into ML or optimization algorithms.

```@meta
CurrentModule = Qaintellect
DocTestSetup = quote
    function ngradient(f, xs::AbstractArray...)
        grads = zero.(xs)
        for (x, Δ) in zip(xs, grads), i in 1:length(x)
            δ = sqrt(eps())
            tmp = x[i]
            x[i] = tmp - δ/2
            y1 = f(xs...)
            x[i] = tmp + δ/2
            y2 = f(xs...)
            x[i] = tmp
            Δ[i] = (y2-y1)/δ
            if eltype(x) <: Complex
                # derivative with respect to imaginary part
                x[i] = tmp - im*δ/2
                y1 = f(xs...)
                x[i] = tmp + im*δ/2
                y2 = f(xs...)
                x[i] = tmp
                Δ[i] += im*(y2-y1)/δ
            end
        end
        return grads
    end
end
```

#### Example
This example shows test code that uses the `Flux.gradient()` call to calculate the gradients of the `Circuit` object.

```jldoctest
using Flux
using Qaintessent
using Qaintellect
using LinearAlgebra

# construct parametrized circuit
N = 4
rz = RzGate(1.5π)
ps = PhaseShiftGate(0.3)
ry = RyGate(√2)
n = randn(Float64, 3)
n /= norm(n)
rg = RotationGate(0.2π, n)
cgc = [
    circuit_gate(3, HadamardGate()),
    circuit_gate(2, rz, (1, 4)),
    circuit_gate(2, 3, SwapGate()),
    circuit_gate(3, ps),
    circuit_gate(3, rg),
    circuit_gate(1, ry),
]

# measurement operators
meas = [MeasurementOperator(Matrix{Float64}(I, 2^N, 2^N), Tuple(1:N)), MeasurementOperator(Hermitian(randn(ComplexF64, 2^N, 2^N)), Tuple(1:N))]
c = Circuit{N}(cgc, meas)

# input quantum state
ψ = randn(ComplexF64, 2^N)

# fictitious gradients of cost function with respect to output quantum state after applying circuit gates
Δ = 0.1*randn(ComplexF64, 2^N)

# Flux will call pullback function with argument Δ
grads = Flux.gradient(() -> real(dot(Δ, apply(c.moments, ψ))), Flux.Params([rz.θ, ps.ϕ, ry.θ, rg.nθ]))

# arguments used implicitly via references
# ngradient calculates gradients via finite difference, adapted from https://github.com/FluxML/Zygote.jl/blob/master/test/gradcheck.jl 
f(args...) = real(dot(Δ, apply(c.moments, ψ)))
all(isapprox.(ngradient(f, rz.θ, ps.ϕ, ry.θ, rg.nθ),
    (grads[rz.θ], grads[ps.ϕ], grads[ry.θ], grads[rg.nθ]), rtol=1e-5, atol=1e-5))

# output
true
```

# QAOA Example

This examples runs a simple Max-Cut algorithm.

```@meta
CurrentModule = Qaintellect
end
```

#### Example

```@eval
using Qaintessent
using Qaintellect
using LinearAlgebra
using Flux
using IterTools: ncycle
# visualization
using Plots
using LaTeXStrings

# number of vertices
n = 5

# graph edges corresponding to the above graph
edges = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)];

# generates QAOA layers
function qaoa_layers(p::Int, n::Int, edges::Vector{Tuple{Int,Int}})
    cgs = CircuitGate[]
    for _ in 1:p
        # C operator
        for e in edges
            # circuit gate uses 1-based indexing
            push!(cgs, circuit_gate(e[1] + 1, e[2] + 1, EntanglementZZGate(0.01*randn())))
        end
        # B operator
        for j in 1:n
           push!(cgs, circuit_gate(j, RxGate(0.01*randn())))
        end
    end
    return cgs
end

# create measurement operator representing C
Cmatrix = zeros(2^n, 2^n)
for edge in edges
    k1 = circuit_gate(edge[1] + 1, ZGate())
    k2 = circuit_gate(edge[2] + 1, ZGate())
    Cmatrix += 0.5*(I - sparse_matrix([k1, k2], n))
end

Cop = MeasurementOperator(Cmatrix, Tuple(1:n));

# example
circ = Circuit{n}(qaoa_layers(2, n, edges), [Cop])

# gather parameters from circuit
paras = Flux.params(circ)

# there is not actually any input data for training
data = ncycle([()], 500)

# define optimizer
opt = Descent(0.5)

# create equal superposition state
s_uni = fill(1/√(2^n) + 0.0im, 2^n);

# define evaluation function
evalcb() = @show(apply(circ, s_uni))

# perform minimization with the negated target function to achieve maximization
Flux.train!(() -> -apply(circ, s_uni)[1], paras, data, opt, cb=Flux.throttle(evalcb, 0.5));

# corresponding optimized quantum wavefunction
ψ1 = apply(circ.moments, s_uni)

tags = [join(reverse(digits(i, pad=n, base=2))) for i in 0:2^n-1];
bar(tags, abs2.(ψ1), xticks=:all, xrotation=45, ylabel=L"|\psi|^2", legend=false);
```

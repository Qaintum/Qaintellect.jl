@info("Ensuring example environment instantiated...")
import Pkg
Pkg.activate("../")
Pkg.instantiate()

@info("Loading Qaintessent, Qaintellect, Flux...")
using Qaintessent, Qaintellect, Flux, Zygote, LinearAlgebra

# construct parametrized circuit
N = 4
meas = MeasurementOps{N}([Matrix{Float64}(I, 2^N, 2^N), Hermitian(randn(ComplexF64, 2^N, 2^N))])
# input quantum state
ψ = randn(ComplexF64, 2^N)
ψ = ψ/norm(ψ)
# fictitious gradients of cost function with respect to circuit output
Δ = [0.3, -1.2]

rz = RzGate(1.5π)
ps = PhaseShiftGate(0.3)
ry = RyGate(√2)
n = randn(Float64, 3)
n = n/norm(n)
rg = RotationGate(0.2π, n)

cgc = CircuitGateChain{N}([
    single_qubit_circuit_gate(3, HadamardGate(), N),
    controlled_circuit_gate((1, 4), 2, rz, N),
    two_qubit_circuit_gate(2, 3, SwapGate(), N),
    single_qubit_circuit_gate(3, ps, N),
    single_qubit_circuit_gate(3, rg, N),
    single_qubit_circuit_gate(1, ry, N),
])
c = Circuit(cgc, meas)

param = Params([rz.θ, ps.ϕ, rg.nθ, ry.θ])
# Zygote assumes input gradients are [1, 1]
gs = gradient(param) do
    ψs = c(ψ)
    dot(Δ, ψs)
end

println("Gradient of rz.θ: " * string(gs[rz.θ]))
println("Gradient of ps.ϕ: " * string(gs[ps.ϕ]))
println("Gradient of rg.nθ: " * string(gs[rg.nθ]))
println("Gradient of ry.θ: " * string(gs[ry.θ]))

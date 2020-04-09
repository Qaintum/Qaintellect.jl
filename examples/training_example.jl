@info("Ensuring example environment instantiated...")
import Pkg
Pkg.activate("../")
Pkg.instantiate()

@info("Loading Qaintessent, Flux...")
using Qaintessent, Flux, Zygote, LinearAlgebra
using Flux.Optimise: update!
using IterTools: ncycle

function output(ψ, Δ)
    @assert length(ψ) == length(Δ)
    dot(ψ, Δ)
end

# construct parametrized circuit
N = 4
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
meas = MeasurementOps{N}([Matrix{Float64}(I, 2^N, 2^N), Hermitian(randn(ComplexF64, 2^N, 2^N))])

c = Circuit(cgc, meas)

# input quantum state
ψ = randn(ComplexF64, 2^N)
ψ = ψ/norm(ψ)

# fictitious gradients of cost function with respect to circuit output
Δ = [0.3, -1.2]
e = 0.65

# set up model
model(ψ) = output(c(ψ), Δ)

loss(x,y) = Flux.mse(model(x), y)
paras = Flux.params(c)
# Freeze parameter ry.θ
delete!(paras, ry.θ)
# Equivalent to delete!(paras, Qaintessent.get_trainable(ry))
data = ncycle([(ψ, e)], 150)
opt = Descent(1)
evalcb() = @show(loss(ψ, e))

println("Initial Circuit: ")
for gate in c.cgc
    println(gate)
end
println("Initial model evaluation: " * string(model(ψ)) * ", Target: " * string(e))

Flux.train!(loss, paras, data, opt, cb=Flux.throttle(evalcb, 0.01))

# Note that ry.θ remains unchanged
println("Final Circuit: ")
for gate in c.cgc
    println(gate)
end
println("Final model evaluation: " * string(model(ψ)) * ", Target: " * string(e))

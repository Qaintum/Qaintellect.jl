# `get_trainable` functions return the trainable variables of a given gate
function get_trainable(c::Circuit{M}) where {M}
    trainable = []
    for gate in c.cgc
        if !isnothing(get_trainable(gate))
            push!(trainable, get_trainable(gate))
        end
    end
    return trainable
end

function get_trainable(g::CircuitGate)
    get_trainable(g.gate)
end

function get_trainable(g::RxGate)
    g.θ
end

function get_trainable(g::RyGate)
    g.θ
end

function get_trainable(g::RzGate)
    g.θ
end

function get_trainable(g::RotationGate)
    g.nθ
end

function get_trainable(g::PhaseShiftGate)
    g.ϕ
end

function get_trainable(g::ControlledGate{M,N}) where {M,N}
    get_trainable(g.U)
end

function get_trainable(g::HadamardGate)
    nothing
end

function get_trainable(g::XGate)
    nothing
end

function get_trainable(g::YGate)
    nothing
end

function get_trainable(g::ZGate)
    nothing
end

function get_trainable(g::SwapGate)
    nothing
end

function get_trainable(g::SGate)
    nothing
end

function get_trainable(g::TGate)
    nothing
end

# Custom adjoint for Qaintessent apply
function store_gradients(cs::Zygote.Context, c::Circuit, ψ::AbstractVector, Δ::AbstractVector)
    grads = Qaintessent.gradients(c, ψ, Δ)
    for (k,v) in grads
        Zygote.accum_param(cs, k, v)
    end
    return nothing
end

@adjoint function Qaintessent.apply(c::Circuit, ψ::AbstractVector)
    cs = __context__
    Qaintessent.apply(c::Circuit, ψ::AbstractVector), Δ -> (store_gradients(cs, c, ψ, Δ), Δ)
end

Flux.trainable(c::Circuit) = get_trainable(c::Circuit)

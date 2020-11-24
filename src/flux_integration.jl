
# let Flux discover the trainable parameters

Flux.@functor RxGate
Flux.@functor RyGate
Flux.@functor RzGate

Flux.@functor RotationGate

Flux.@functor PhaseShiftGate

Flux.@functor EntanglementXXGate
Flux.@functor EntanglementYYGate
Flux.@functor EntanglementZZGate

Flux.@functor ControlledGate

Flux.@functor CircuitGate
Flux.@functor CircuitGateChain
Flux.@functor Moment
Flux.@functor MeasurementOps
Flux.@functor Circuit

function collect_gradients(cx::Zygote.Context, q, dq)
    # special cases: circuit gate chain
    # TODO: also support measurement operators
    if typeof(q) <: AbstractVector{<:AbstractCircuitGate} || typeof(q) <: AbstractVector{<:Moment}
        for i in 1:length(q)
            collect_gradients(cx, q[i], dq[i])
        end
    else
        # need to explicitly "accumulate parameters" for Flux Params([...]) to work
        Zygote.accum_param(cx, q, dq)
        for f in fieldnames(typeof(q))
            try
                collect_gradients(cx, getfield(q, f), getfield(dq, f))
            catch UndefRefError
            end
        end
    end
end


# custom adjoint
Zygote.@adjoint apply(cgc::CircuitGateChain{N}, ψ::AbstractVector{<:Complex}) where {N} = begin
    ψ1 = apply(cgc, ψ)
    ψ1, function(Δ)
        # backward pass through unitary gates
        dcgc, ψbar = Qaintessent.backward(cgc, ψ1, Δ)
        collect_gradients(__context__, cgc, dcgc)
        return (dcgc, ψbar)
    end
end


# custom adjoint
Zygote.@adjoint apply(c::Circuit{N}, ψ::AbstractVector) where {N} = begin
    apply(c, ψ), function(Δ)
        # TODO: don't recompute apply(c, ψ) here
        dc, ψbar = Qaintessent.gradients(c, ψ, Δ)
        collect_gradients(__context__, c, dc)
        return (dc, ψbar)
    end
end

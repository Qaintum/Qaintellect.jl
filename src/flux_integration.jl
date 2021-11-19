using Qaintessent.MaxCutWSQAOA
using Qaintessent.MaxKColSubgraphQAOA

# let Flux discover the trainable parameters

Flux.@functor RxGate
Flux.@functor RyGate
Flux.@functor RzGate
Flux.@functor RotationGate
Flux.@functor PhaseShiftGate
Flux.@functor ControlledGate
Flux.@functor EntanglementXXGate
Flux.@functor EntanglementYYGate
Flux.@functor EntanglementZZGate
Flux.@functor CircuitGate
Flux.@functor Moment
Flux.@functor MeasurementOperator
Flux.@functor Circuit
Flux.@functor ParityRingMixerGate
Flux.@functor RNearbyValuesMixerGate
Flux.@functor PartitionMixerGate
Flux.@functor MaxKColSubgraphPhaseSeparationGate
Flux.@functor MaxCutPhaseSeparationGate
Flux.@functor WSQAOAMixerGate (β,)
Flux.@functor RxMixerGate

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
Zygote.@adjoint apply(ψ::Vector{<:Complex}, moments::Vector{Moment}) = begin
    N = Qaintessent.intlog2(length(ψ))
    length(ψ) == 2^N || error("Vector length must be a power of 2")
    ψ1 = apply(ψ, moments)
    ψ1, function(Δ)
        # factor 1/2 due to convention for Wirtinger derivative with prefactor 1/2
        dmoments, ψbar = Qaintessent.backward(moments, ψ1, 0.5*Δ, N)
        ψbar .*= 2.0
        collect_gradients(__context__, moments, dmoments)
        return (ψbar, dmoments)
    end
end


# custom adjoint
Zygote.@adjoint apply(ψ::AbstractVector, c::Circuit{N}) where {N} = begin
    apply(ψ, c), function(Δ)
        # TODO: don't recompute apply(c, ψ) here
        dc, ψbar = Qaintessent.gradients(c, ψ, Δ)
        collect_gradients(__context__, c, dc)
        return (ψbar, dc)
    end
end


# custom adjoint for applying moments to a density matrix
Zygote.@adjoint apply(ρ::DensityMatrix, moments::Vector{Moment}) = begin
    ρ1 = apply(ρ, moments)
    ρ1, function(Δ)
        if !(Δ isa DensityMatrix)
            Δ = DensityMatrix(Δ[1], ρ.N)
        end
        dmoments, ρbar = Qaintessent.backward_density(moments, ρ1, Δ)
        collect_gradients(__context__, moments, dmoments)
        return (ρbar, dmoments)
    end
end


# custom adjoint for applying a circuit to a density matrix
Zygote.@adjoint apply(ρ::DensityMatrix, c::Circuit{N}) where {N} = begin
    @assert(ρ.N == N)
    apply(ρ, c), function(Δ)
        # TODO: don't recompute apply(c, ρ) here
        dc, ρbar = Qaintessent.gradients(c, ρ, Δ)
        collect_gradients(__context__, c, dc)
        return (ρbar, dc)
    end
end


# custom adjoint for DensityMatrix constructor; setting gradient with respect to integer `N` to zero. 
Zygote.@adjoint DensityMatrix(v::AbstractVector{<:Real}, N::Integer) = DensityMatrix(v, N), ρbar -> (ρbar.v, 0)

using Test
using TestSetExtensions
using LinearAlgebra
using Flux
using Qaintellect


# adapted from https://github.com/FluxML/Zygote.jl/blob/master/test/gradcheck.jl
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


@testset ExtendedTestSet "circuit gradients" begin

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

    @testset "circuit gate chain gradients" begin
        # fictitious gradients of cost function with respect to output quantum state after applying circuit gates
        Δ = 0.1*randn(ComplexF64, 2^N)

        # Flux will call pullback function with argument Δ
        grads = Flux.gradient(() -> real(dot(Δ, apply(ψ, c.moments))), Flux.Params([rz.θ, ps.ϕ, ry.θ, rg.nθ]))

        # arguments used implicitly via references
        f(args...) = real(dot(Δ, apply(ψ, c.moments)))
        @test all(isapprox.(ngradient(f, rz.θ, ps.ϕ, ry.θ, rg.nθ),
            (grads[rz.θ], grads[ps.ϕ], grads[ry.θ], grads[rg.nθ]), rtol=1e-5, atol=1e-5))
    end

    @testset "circuit gate chain gradients 2" begin
        Δ = 0.1*randn(Float64, 2^N)

        grads = Flux.gradient(() -> dot(Δ, abs2.(apply(ψ, c.moments))), Flux.Params([rz.θ, ps.ϕ, ry.θ, rg.nθ]))

        # arguments used implicitly via references; factor 2 due to convention for Wirtinger derivative with prefactor 1/2
        f(args...) = dot(Δ, abs2.(apply(ψ, c.moments)))
        @test all(isapprox.(ngradient(f, rz.θ, ps.ϕ, ry.θ, rg.nθ),
            (grads[rz.θ], grads[ps.ϕ], grads[ry.θ], grads[rg.nθ]), rtol=1e-5, atol=1e-5))
    end

    @testset "circuit with measurement gradients" begin
        # fictitious gradients of cost function with respect to circuit output
        Δ = [0.3, -1.2]

        grads = Flux.gradient(() -> dot(Δ, apply(ψ, c)), Flux.Params([rz.θ, ps.ϕ, ry.θ, rg.nθ]))

        # arguments used implicitly via references
        f(args...) = dot(Δ, apply(ψ, c))
        @test all(isapprox.(ngradient(f, rz.θ, ps.ϕ, ry.θ, rg.nθ),
            (grads[rz.θ], grads[ps.ϕ], grads[ry.θ], grads[rg.nθ]), rtol=1e-5, atol=1e-5))
    end

    @testset "moments chain gradients for density matrices" begin
        # fictitious gradients of cost function with respect to output density matrix
        Δ = DensityMatrix(0.1*randn(Float64, 256), 4)

        ρ = density_from_statevector(ψ)

        grads = Flux.gradient(() -> dot(Δ.v, apply(ρ, c.moments).v), Flux.Params([rz.θ, ps.ϕ, ry.θ, rg.nθ]))
        
        # arguments used implicitly via references
        f(args...) = dot(Δ.v, apply(ρ, c.moments).v)
        @test all(isapprox.(ngradient(f, rz.θ, ps.ϕ, ry.θ, rg.nθ),
            (grads[rz.θ], grads[ps.ϕ], grads[ry.θ], grads[rg.nθ]), rtol=1e-5, atol=1e-5))
    end


    @testset "circuit with measurement gradients for density matrices" begin
        # fictitious gradients of cost function with respect to circuit output
        Δ = [0.3, -1.2]
        ρ = density_from_statevector(ψ)

        grads = Flux.gradient(() -> dot(Δ, apply(ρ, c)), Flux.Params([rz.θ, ps.ϕ, ry.θ, rg.nθ]))

        # arguments used implicitly via references
        f(args...) = dot(Δ, apply(ρ, c))
        @test all(isapprox.(ngradient(f, rz.θ, ps.ϕ, ry.θ, rg.nθ),
            (grads[rz.θ], grads[ps.ϕ], grads[ry.θ], grads[rg.nθ]), rtol=1e-5, atol=1e-5))
    end

end

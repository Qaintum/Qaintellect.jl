# Utility function to count the properly colored edges in a graph coloring (i.e. endpoints have different colors)
function properly_colored_edges(graph::Graph, coloring::Vector{Int})
    length(coloring) == graph.n || throw(ArgumentError("Length of coloring must equal the number of vertices."))
    return count(coloring[a] != coloring[b] for (a, b) ∈ graph.edges)
end 

# Utility function to compute the probabilities of the outcomes represented by a wavefunction ψ, sorted by descending probability
function wavefunction_distribution(ψ::Vector{ComplexF64}; as_bitstrings::Bool = true,
        include_zero = false)::Union{Vector{Tuple{Int, Float64}}, Vector{Tuple{Vector{Int}, Float64}}}
    distribution = [(i-1, abs(amplitude)^2) for (i, amplitude) ∈ enumerate(ψ) if abs(amplitude) > 0 || include_zero]
    if as_bitstrings
        N = Int(log2(length(ψ)))
        distribution = [(digits(i, base=2, pad=N) |> reverse, p) for (i, p) ∈ distribution]
    end
    
    return sort(distribution, by=(t -> -t[2]))
end

# Utility function to compute the probabilities of the outcome wavefunction of a circuit applied to ψ, sorted by descending probability
function output_distribution(circ::Circuit, ψ::Vector{ComplexF64}; as_bitstrings::Bool = true,
        include_zero = false)::Union{Vector{Tuple{Int, Float64}}, Vector{Tuple{Vector{Int}, Float64}}}
    ψ_out = apply(ψ, circ.moments)
    return wavefunction_distribution(ψ_out, as_bitstrings=as_bitstrings, include_zero=include_zero)
end

# Utility function to compute the probabilities of the output colorings of a circuit applied to ψ, sorted by descending probability
function output_colorings_distribution(circ::Circuit{N}, n::Int;
        include_zero = false)::Vector{Tuple{Dict{Int, Vector{Int}}, Float64}} where {N}
    κ = Int(N / n)
    ψ = ψ_initial(n, κ)
    ψ_out = apply(ψ, circ.moments)
    distribution = wavefunction_distribution(ψ_out, as_bitstrings=false, include_zero=include_zero)

    return [(decode_basis_state(state, n, κ), p) for (state, p) ∈ distribution]
end

# Utility function to compute the probabilities of the output colorings of a circuit applied to ψ, sorted by descending probability
function output_colorings_distribution_scored(circ::Circuit{N}, graph::Graph;
        include_zero = false)::Vector{Tuple{Vector{Int}, Int, Float64}} where {N}
    dist = output_colorings_distribution(circ, graph.n, include_zero = include_zero)
    
    coloring_dict_to_list(dict) = [length(dict[i]) == 1 ? dict[i][1] : 
        throw(ArgumentError("At least one vertex has not one unique color.")) for i ∈ 1:graph.n]

    return [(coloring_dict_to_list(coloring), properly_colored_edges(graph, coloring_dict_to_list(coloring)), p) for (coloring, p) ∈ dist]
end

# Decodes the coloring represented by a single computational basis state, represented as integer
function decode_basis_state(basis_state::Int, n::Int, κ::Int)::Dict{Int, Vector{Int}}
    (n > 0 && κ > 0) || throw(DomainError("Parameters n and κ must be positive integers."))

    N = n * κ
    bits = digits(basis_state, base=2, pad=N) |> reverse
    vertex_bits = vertex -> bits[((vertex - 1) * κ + 1):(vertex * κ)]
    colors_by_vertex = Dict([(v, findall(!iszero, vertex_bits(v))) for v ∈ 1:n])

    all(!isnothing, colors_by_vertex) || throw(ArgumentError("The state `basis_state` has at least one vertex without a color."))
    
    return colors_by_vertex
end

# Prototypical log function. Implementations should have the @Zygote.nograd attribute and use 
# a data structure as first argument which can save log data. params is ::Zygote.Params.
function log_qaoa(logger::T, round::Int, ψ_out::Vector{ComplexF64}, objective::Float64, params) where {T}
    throw(ArgumentError("The type `$(T)` does not implement the `log_qaoa`
        function and is not a valid logger."))
end

function plot_circuit_output_distribution(circ::Circuit, ψ)
    data = output_distribution(circ, ψ, as_bitstrings=true, include_zero=false)
    sort!(data, by=d -> d[1])
    bar(data .|> (d -> "|$(join(d[1]))⟩"), 
                            data .|> (d -> d[2]), 
                        ylims=(0, 1), xrotation = 60, color="#ff8080", x_ticks=:all,
                        bottom_margin=5Plots.mm, 
                        xtickfontsize=5, legend=:topleft,
                        label = "outcome probabilities"
                    )   
end
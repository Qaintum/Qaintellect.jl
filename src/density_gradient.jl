
function Base.dot(ρ1::DensityMatrix, ρ2::DensityMatrix)
    ρ1.N == ρ2.N || error("Density matrix must be of same size")
        v_new = 
end
module Qaintellect

using Qaintessent
using Zygote, Flux

include("flux_integration.jl")


# re-export definitions from Qaintessent.jl

# gates.jl
export
    AbstractGate,
    X,
    Y,
    Z,
    XGate,
    YGate,
    ZGate,
    HadamardGate,
    SGate,
    TGate,
    SdagGate,
    TdagGate,
    RxGate,
    RyGate,
    RzGate,
    RotationGate,
    PhaseShiftGate,
    SwapGate,
    EntanglementXXGate,
    EntanglementYYGate,
    EntanglementZZGate,
    ControlledGate,
    controlled_not,
    MatrixGate,
    matrix,
    sparse_matrix,
    num_wires

# circuitgate.jl
export
    AbstractCircuitGate,
    CircuitGate,
    circuit_gate

# circuit.jl
export
    Moment,
    single_qubit_circuit_gate,
    two_qubit_circuit_gate,
    controlled_circuit_gate,
    circuit_gate,
    MeasurementOperator,
    Circuit,
    distribution,
    rdm

# density_matrix.jl
export
    DensityMatrix,
    pauli_group_matrix,
    density_from_statevector,
    density_from_matrix

# commute.jl
export
    iscommuting

# apply.jl
export
    apply

end

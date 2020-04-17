module Qaintellect

using Qaintessent
using Zygote, Flux

include("flux_integration.jl")


# re-export definitions from Qaintessent.jl

# gates
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
    ControlledGate,
    controlled_not

# circuit
export
    AbstractCircuitGate,
    CircuitGate,
    single_qubit_circuit_gate,
    two_qubit_circuit_gate,
    controlled_circuit_gate,
    apply,
    rdm,
    CircuitGateChain,
    MeasurementOps,
    Circuit

# models
export
    qft_circuit

end

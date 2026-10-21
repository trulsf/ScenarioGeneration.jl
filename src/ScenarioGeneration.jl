"""
    ScenarioGeneration

Prototype of scenario generation methods that combine the operational scenarios of the
representative periods of a `TimeStruct` time structure into joint scenario paths. Each
method is a subtype of `TimeStruct.PairingMethod`, and `TimeStruct.generate_pairing`
returns a `TimeStruct.ScenarioPairing` that can be used with `TimeStruct.scenario_paths` or
stored in `TimeStruct.RepresentativePeriods`. A method can also be passed directly wherever
a `ScenarioPairing` is expected.

The scenario probabilities are given either as a time structure, in which case they are
obtained through `TimeStruct.scenario_probabilities`, or directly as a vector indexed by
representative period and scenario, `probs[r][s]`.
"""
module ScenarioGeneration

import Random
using TimeStruct
import TimeStruct: _paths, _default_check_marginals, _marginals

export PairingMethod, generate_pairing
export CartesianPairing, QuantilePairing, RandomPairing, RakedPairing

include("pairing_method.jl")
include("cartesian_pairing.jl")
include("quantile_pairing.jl")
include("random_pairing.jl")
include("raked_pairing.jl")

end # module

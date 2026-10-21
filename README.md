# ScenarioGeneration.jl

Prototype of scenario generation methods for [TimeStruct.jl](https://github.com/sintefore/TimeStruct.jl).
The methods combine the operational scenarios of the representative periods of a time
structure into joint scenario paths, returned as a `TimeStruct.ExplicitPairing`.

## Methods

All methods are subtypes of `PairingMethod` and are used through `generate_pairing`:

| Method                         | Paths                                                         |
| ------------------------------ | ------------------------------------------------------------- |
| `CartesianPairing()`           | All combinations of scenarios, assuming independence          |
| `QuantilePairing(; order)`     | Scenarios coupled by their quantiles (comonotonic coupling)   |
| `RandomPairing(n; stratified)` | `n` sampled paths with equal probability                      |
| `RakedPairing(scenarios)`      | Given paths, with probabilities fitted through raking         |

## Example

```julia
using TimeStruct
using ScenarioGeneration

day = SimpleTimes(4, 1)
oscs = OperationalScenarios([day, day, day], [0.5, 0.3, 0.2])
rps = RepresentativePeriods(2, 8760, oscs)

# Generate the pairing and store it in the time structure
pairing = generate_pairing(QuantilePairing(), rps)
rps = RepresentativePeriods(2, 8760, oscs; pairing)
scenario_paths(rps)

# The scenario probabilities can also be given directly as `probs[r][s]`
generate_pairing(CartesianPairing(), [[0.5, 0.5], [0.3, 0.4, 0.3]])
```

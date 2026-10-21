"""
    QuantilePairing(; order = nothing, atol = 1e-9)

Pairing method that couples the operational scenarios of the representative periods by their
quantiles, that is, the comonotonic coupling. The scenarios of each representative period
are ordered, and their cumulative probabilities define a partition of the interval [0, 1].
The paths correspond to the intervals of the joint refinement of these partitions, with
probabilities given by the lengths of the intervals. Hence, the scenario at quantile `q` in
one representative period is paired with the scenario at quantile `q` in all other
representative periods.

The path probabilities are consistent with the scenario probabilities, and the number of
paths is at most the total number of scenarios minus the number of representative periods
plus one. If all representative periods have the same scenario probabilities, the result is
equivalent to `DiagonalPairing`.

The keyword `order` gives the ordering of the scenarios for each representative period,
*e.g.*, from dry to wet, as a vector of permutations of the scenario indices. The default
`nothing` uses the scenario indices as ordering. Reversing the ordering of a representative
period pairs low quantiles in this period with high quantiles in the other periods.

Breakpoints closer than `atol` are merged to avoid paths with negligible probabilities due to
rounding.

## Example

```julia
# Scenario probabilities [0.5, 0.5] and [0.3, 0.4, 0.3] result in the paths
# [1, 1] (0.3), [1, 2] (0.2), [2, 2] (0.2), and [2, 3] (0.3)
generate_pairing(QuantilePairing(), [[0.5, 0.5], [0.3, 0.4, 0.3]])

# Scenario 2 is the lowest in the second representative period
generate_pairing(QuantilePairing(; order = [[1, 2], [2, 1, 3]]), [[0.5, 0.5], [0.3, 0.4, 0.3]])
```
"""
struct QuantilePairing <: PairingMethod
    order::Union{Nothing,Vector{Vector{Int}}}
    atol::Float64
end
function QuantilePairing(; order = nothing, atol = 1e-9)
    return QuantilePairing(
        isnothing(order) ? nothing : [Vector{Int}(o) for o in order],
        atol,
    )
end

function _paths(method::QuantilePairing, probs)
    order = _order(method, probs)

    # Cumulative probabilities of the ordered scenarios, normalized to end at one
    cums = [cumsum(probs[r][order[r]]) ./ sum(probs[r]) for r in eachindex(probs)]
    breaks = [0.0]
    for b in sort(reduce(vcat, cums))
        b - last(breaks) > method.atol && push!(breaks, b)
    end
    breaks[end] = 1.0

    scens = Vector{Int}[]
    path_probs = Float64[]
    for i in 2:length(breaks)
        mid = (breaks[i-1] + breaks[i]) / 2
        # `min` guards against rounding in the cumulative sums
        push!(
            scens,
            [
                order[r][min(searchsortedfirst(cums[r], mid), length(cums[r]))] for
                r in eachindex(probs)
            ],
        )
        push!(path_probs, breaks[i] - breaks[i-1])
    end
    return scens, path_probs
end

# The validated ordering of the scenarios of each representative period
function _order(method::QuantilePairing, probs)
    isnothing(method.order) && return [collect(eachindex(p)) for p in probs]
    order = method.order
    if length(order) != length(probs)
        throw(
            ArgumentError(
                "`order` is given for $(length(order)) representative periods while the " *
                "scenario probabilities are given for $(length(probs)).",
            ),
        )
    end
    for r in eachindex(probs)
        if sort(order[r]) != collect(eachindex(probs[r]))
            throw(
                ArgumentError(
                    "`order` of representative period $r is not a permutation of its " *
                    "scenario indices.",
                ),
            )
        end
    end
    return order
end

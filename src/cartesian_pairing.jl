"""
    CartesianPairing()

Pairing method with all combinations of the operational scenarios of the representative
periods. The probability of a path is the product of the probabilities of its scenarios.
This corresponds to assuming that the scenarios of the representative periods are
independent. The path probabilities are consistent with the scenario probabilities.

The paths are ordered lexicographically with the scenario of the last representative period
varying fastest.

!!! warning
    The number of paths grows exponentially with the number of representative periods.
"""
struct CartesianPairing <: PairingMethod end

function _paths(::CartesianPairing, probs)
    # `Iterators.product` varies the first index fastest, hence the reversals
    combs = Iterators.product((eachindex(p) for p in reverse(probs))...)
    scens = [reverse(collect(Int, c)) for c in vec(collect(combs))]
    return scens, _product_probs(scens, probs)
end

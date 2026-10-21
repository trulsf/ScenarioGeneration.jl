"""
    RandomPairing(n; rng = Random.default_rng(), stratified = true)

Pairing method with `n` sampled paths, each with the probability `1/n`. This corresponds to
a Monte Carlo approximation of [`CartesianPairing`](@ref) with a fixed number of paths.

If `stratified` is `true` (default), each scenario of a representative period is assigned
to a number of paths proportional to its probability, rounded to an integer through the
largest remainder method. The assignments are then shuffled independently for each
representative period. Hence, the path probabilities are consistent with the scenario
probabilities up to the rounding to multiples of `1/n`, while the combination of scenarios
across representative periods is random.

If `stratified` is `false`, the scenario of each representative period is drawn
independently according to the scenario probabilities. The path probabilities are then only
consistent with the scenario probabilities in expectation.

As the sampled path probabilities are in general not exactly consistent with the scenario
probabilities, the consistency check of the `ScenarioPairing` is disabled by default. The
scenario probabilities can be made consistent through the keyword argument
`scenario_probability = :from_pairing` of `RepresentativePeriods`.

!!! note
    The same combination of scenarios can occur in several paths. The probability of a
    combination is then the sum of the probabilities of these paths.
"""
struct RandomPairing{R<:Random.AbstractRNG} <: PairingMethod
    n::Int
    rng::R
    stratified::Bool
    function RandomPairing(
        n::Integer;
        rng::Random.AbstractRNG = Random.default_rng(),
        stratified::Bool = true,
    )
        n < 1 && throw(ArgumentError("The number of paths must be positive."))
        return new{typeof(rng)}(n, rng, stratified)
    end
end

_default_check_marginals(::RandomPairing) = :none

function _paths(method::RandomPairing, probs)
    (; n, rng) = method
    if method.stratified
        assign = [Random.shuffle(rng, _stratify(p, n)) for p in probs]
        scens = [[assign[r][k] for r in eachindex(probs)] for k in 1:n]
    else
        cum = [cumsum(p) for p in probs]
        # `min` guards against rounding in the cumulative sum
        scens = [
            [min(searchsortedfirst(c, rand(rng) * last(c)), length(c)) for c in cum] for
            _ in 1:n
        ]
    end
    return scens, fill(1.0 / n, n)
end

# Scenario of each of `n` paths, with the number of paths per scenario proportional to the
# probabilities `p`, rounded with the largest remainder method
function _stratify(p::Vector{Float64}, n::Integer)
    target = n .* p ./ sum(p)
    count = floor.(Int, target)
    order = sortperm(target .- count; rev = true)
    count[order[1:(n-sum(count))]] .+= 1
    return reduce(vcat, [fill(s, count[s]) for s in eachindex(count)])
end

"""
    RakedPairing(scenarios; atol = 1e-9, maxiter = 10_000)

Pairing method with the given paths `scenarios` (a matrix with one row per path, or a vector
of paths) and path probabilities derived from the scenario probabilities.

The path probabilities are the consistent probabilities that are closest (in terms of
relative entropy) to the probabilities obtained when assuming independent representative
periods. They are computed with iterative proportional fitting (raking). An error is thrown
if no consistent path probabilities exist for the given paths, *e.g.*, if a scenario with
positive probability is not part of any path.
"""
struct RakedPairing <: PairingMethod
    scenarios::Vector{Vector{Int}}
    atol::Float64
    maxiter::Int
end
function RakedPairing(scenarios::AbstractMatrix{<:Integer}; atol = 1e-9, maxiter = 10_000)
    return RakedPairing([Vector{Int}(row) for row in eachrow(scenarios)], atol, maxiter)
end
function RakedPairing(
    scenarios::Vector{<:AbstractVector{<:Integer}};
    atol = 1e-9,
    maxiter = 10_000,
)
    n_rp = isempty(scenarios) ? 0 : length(first(scenarios))
    if any(length(path) != n_rp for path in scenarios)
        throw(ArgumentError("All paths must have the same length."))
    end
    return RakedPairing([Vector{Int}(path) for path in scenarios], atol, maxiter)
end

function _paths(method::RakedPairing, probs)
    scens = method.scenarios
    n_rp = isempty(scens) ? 0 : length(first(scens))
    if n_rp != length(probs)
        throw(
            ArgumentError(
                "The paths specify $n_rp representative periods while the scenario " *
                "probabilities are given for $(length(probs)).",
            ),
        )
    elseif any(s[r] ∉ eachindex(probs[r]) for s in scens for r in eachindex(probs))
        throw(ArgumentError("The paths use scenarios that do not exist."))
    end
    return scens, _rake(scens, probs, method.atol, method.maxiter)
end

# Iterative proportional fitting of the path probabilities to the scenario probabilities,
# starting from the probabilities assuming independent representative periods
function _rake(scens, probs, atol, maxiter)
    for r in eachindex(probs), s in eachindex(probs[r])
        if probs[r][s] > 0 && !any(sc[r] == s for sc in scens)
            throw(
                ArgumentError(
                    "Scenario $s of representative period $r has a positive probability " *
                    "but is not part of any path. Hence, no path probabilities consistent " *
                    "with the scenario probabilities exist.",
                ),
            )
        end
    end
    path_probs = _product_probs(scens, probs)
    if sum(path_probs) == 0
        throw(ArgumentError("All paths include a scenario with zero probability."))
    end
    path_probs ./= sum(path_probs)
    for _ in 1:maxiter
        _marginal_deviation(scens, path_probs, probs) <= atol && return path_probs
        for r in eachindex(probs)
            marg = _marginals(scens, path_probs, probs)[r]
            for (k, s) in enumerate(scens)
                m = marg[s[r]]
                path_probs[k] = m > 0 ? path_probs[k] * probs[r][s[r]] / m : 0.0
            end
        end
    end
    dev = _marginal_deviation(scens, path_probs, probs)
    return throw(
        ArgumentError(
            "No path probabilities consistent with the scenario probabilities exist for " *
            "the given paths (maximum deviation $dev after $maxiter iterations). Add " *
            "paths or provide the path probabilities directly.",
        ),
    )
end

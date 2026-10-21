# The pairing interface (`PairingMethod`, `generate_pairing`, `_paths`,
# `_default_check_marginals`) and the helpers `_probs` and `_marginals` are defined in
# `TimeStruct`. The methods of this package extend `_paths` and `_default_check_marginals`.

# The probability of each path when assuming independent representative periods
function _product_probs(scens, probs)
    return [prod(probs[r][s[r]] for r in eachindex(s)) for s in scens]
end

# The largest deviation between the summed path probabilities of a scenario and its
# scenario probability
function _marginal_deviation(scens, path_probs, probs)
    marg = _marginals(scens, path_probs, probs)
    return maximum(maximum(abs.(marg[r] .- probs[r])) for r in eachindex(probs))
end

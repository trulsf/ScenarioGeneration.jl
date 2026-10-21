using Test
using Random
using TimeStruct
using ScenarioGeneration

const day = SimpleTimes(4, 1)
const oscs = OperationalScenarios([day, day, day], [0.5, 0.3, 0.2])

@testset "Cartesian pairing" begin
    rps = RepresentativePeriods(2, 8760, oscs)
    pairing = generate_pairing(CartesianPairing(), rps)
    paths = scenario_paths(rps, pairing)
    @test length(paths) == 9
    @test path_scenarios(paths[2]) == [1, 2]
    @test probability(paths[2]) ≈ 0.5 * 0.3
    @test sum(probability, paths) ≈ 1
    @test path_marginals(rps, pairing) ≈ scenario_probabilities(rps)
    @test Set(Iterators.flatten(paths)) == Set(rps)
    # Probabilities given directly
    @test generate_pairing(CartesianPairing(), [[0.5, 0.5], [1.0]]).scenarios == [1 1; 2 1]
    # Stored in the time structure
    rps_c = RepresentativePeriods(2, 8760, oscs; pairing)
    @test length(scenario_paths(rps_c)) == 9
    # Per strategic period only
    @test_throws ArgumentError generate_pairing(CartesianPairing(), TwoLevel(2, 1, rps))
    @test length(
        generate_pairing(CartesianPairing(), first(strategic_periods(TwoLevel(2, 1, rps)))).probability,
    ) == 9
end

@testset "Quantile pairing" begin
    # Different numbers of scenarios
    pairing = generate_pairing(QuantilePairing(), [[0.5, 0.5], [0.3, 0.4, 0.3]])
    @test pairing.scenarios == [1 1; 1 2; 2 2; 2 3]
    @test pairing.probability ≈ [0.3, 0.2, 0.2, 0.3]
    rps = RepresentativePeriods(
        8760,
        [0.5, 0.5],
        [
            OperationalScenarios([day, day], [0.5, 0.5]),
            OperationalScenarios([day, day, day], [0.3, 0.4, 0.3]),
        ],
    )
    @test generate_pairing(QuantilePairing(), rps).scenarios == pairing.scenarios
    @test path_marginals(rps, pairing) ≈ scenario_probabilities(rps)
    @test_logs scenario_paths(rps, pairing)
    # `DiagonalPairing` is not possible for these representative periods
    @test_throws ArgumentError scenario_paths(rps, DiagonalPairing())

    # Equivalent to `DiagonalPairing` for identical scenario probabilities
    rps = RepresentativePeriods(3, 8760, oscs)
    pairing = generate_pairing(QuantilePairing(), rps)
    @test pairing.scenarios == [1 1 1; 2 2 2; 3 3 3]
    @test pairing.probability ≈ probability.(scenario_paths(rps, DiagonalPairing()))

    # Ordering of the scenarios
    pairing = generate_pairing(
        QuantilePairing(; order = [[1, 2], [2, 1, 3]]),
        [[0.5, 0.5], [0.3, 0.4, 0.3]],
    )
    @test pairing.scenarios == [1 2; 1 1; 2 1; 2 3]
    @test pairing.probability ≈ [0.4, 0.1, 0.2, 0.3]
    # Reversed ordering pairs low with high quantiles
    pairing =
        generate_pairing(QuantilePairing(; order = [[1, 2, 3], [3, 2, 1], [1, 2, 3]]), rps)
    @test path_marginals(rps, pairing) ≈ scenario_probabilities(rps)
    @test all(
        pairing.scenarios[k, 2] >= pairing.scenarios[l, 2] for
        k in axes(pairing.scenarios, 1), l in axes(pairing.scenarios, 1) if k < l
    )

    # The number of paths is bounded by the total number of scenarios
    probs = [[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.5], [0.6, 0.4]]
    pairing = generate_pairing(QuantilePairing(), probs)
    @test size(pairing.scenarios, 1) <= sum(length, probs) - length(probs) + 1
    @test sum(pairing.probability) ≈ 1
    # Monotone: each period's scenario index never decreases along the paths
    @test all(issorted(col) for col in eachcol(pairing.scenarios))

    # Zero probabilities and rounding do not result in additional paths
    pairing = generate_pairing(
        QuantilePairing(),
        [[0.5, 0.0, 0.5], [1 / 3, 1 / 3, 1 / 3], [1 / 3, 2 / 3]],
    )
    @test all(!=(2), pairing.scenarios[:, 1])
    @test all(>(1e-6), pairing.probability)
    @test size(pairing.scenarios, 1) == 4

    @test_throws ArgumentError generate_pairing(QuantilePairing(; order = [[1, 2, 3]]), rps)
    @test_throws ArgumentError generate_pairing(
        QuantilePairing(; order = [[1, 2, 2], [1, 2, 3], [1, 2, 3]]),
        rps,
    )
    @test_throws ArgumentError generate_pairing(QuantilePairing(), TwoLevel(2, 1, rps))
end

@testset "Random pairing" begin
    rps = RepresentativePeriods(3, 8760, oscs)
    probs = scenario_probabilities(rps)

    pairing = generate_pairing(RandomPairing(5; rng = Xoshiro(1)), rps)
    @test size(pairing.scenarios) == (5, 3)
    @test pairing.probability ≈ fill(0.2, 5)
    @test all(in(1:3), pairing.scenarios)
    # Not checked by default as the sample is not exactly consistent
    @test_logs scenario_paths(rps, pairing)

    # Reproducible for a given random number generator
    @test generate_pairing(RandomPairing(5; rng = Xoshiro(1)), rps).scenarios ==
          pairing.scenarios
    @test generate_pairing(RandomPairing(50; rng = Xoshiro(3)), rps).scenarios !=
          generate_pairing(RandomPairing(50; rng = Xoshiro(4)), rps).scenarios

    # Stratified: the marginal probabilities are within 1/n of the scenario probabilities
    for n in (1, 2, 5, 7, 10, 33)
        marg = path_marginals(rps, generate_pairing(RandomPairing(n), rps))
        @test all(abs(marg[r][s] - probs[r][s]) < 1 / n for r in 1:3, s in 1:3)
        # Each scenario is used by an integer number of paths
        @test all(isapprox(m * n, round(m * n); atol = 1e-9) for r in 1:3 for m in marg[r])
    end
    @test path_marginals(rps, generate_pairing(RandomPairing(10), rps)) ≈ probs
    # The scenario combinations differ between representative periods
    @test length(
        unique(
            eachrow(generate_pairing(RandomPairing(10; rng = Xoshiro(1)), rps).scenarios),
        ),
    ) > 3

    # Not stratified: the marginal probabilities approach the scenario probabilities
    pairing =
        generate_pairing(RandomPairing(10000; stratified = false, rng = Xoshiro(1)), rps)
    @test all(isapprox.(path_marginals(rps, pairing), probs; atol = 0.02))

    # Scenarios with zero probability are never sampled
    for stratified in (true, false)
        pairing = generate_pairing(
            RandomPairing(101; stratified),
            [[0.5, 0.0, 0.5], [0.5, 0.0, 0.5]],
        )
        @test all(!=(2), pairing.scenarios)
    end

    # Scenario probabilities derived from the sampled paths are consistent
    pairing = generate_pairing(RandomPairing(7; rng = Xoshiro(2)), rps)
    rps_fp = RepresentativePeriods(
        3,
        8760,
        oscs;
        pairing = ScenarioPairing(
            pairing.scenarios,
            pairing.probability;
            check_marginals = :error,
        ),
        scenario_probability = :from_pairing,
    )
    @test path_marginals(rps_fp) ≈ scenario_probabilities(rps_fp)

    @test_throws ArgumentError RandomPairing(0)
end

@testset "Raked pairing" begin
    rps = RepresentativePeriods(2, 8760, oscs)
    probs = scenario_probabilities(rps)

    scens = [1 1; 2 2; 3 3; 1 2; 2 1]
    pairing = generate_pairing(RakedPairing(scens), rps)
    @test pairing.scenarios == scens
    @test all(>=(0), pairing.probability)
    @test sum(pairing.probability) ≈ 1
    @test path_marginals(rps, pairing) ≈ probs
    @test generate_pairing(RakedPairing(collect.(eachrow(scens))), rps).probability ≈
          pairing.probability
    # Unique when each scenario is in exactly one path
    @test generate_pairing(RakedPairing([1 1; 2 2; 3 3]), rps).probability ≈ [0.5, 0.3, 0.2]
    # Different scenario probabilities in the representative periods
    probs_ab = [[0.5, 0.3, 0.2], [0.6, 0.4]]
    pairing = generate_pairing(RakedPairing([1 1; 2 1; 2 2; 3 2]), probs_ab)
    rps_ab = RepresentativePeriods(
        8760,
        [0.5, 0.5],
        [oscs, OperationalScenarios([day, day], [0.6, 0.4])],
    )
    @test path_marginals(rps_ab, pairing) ≈ probs_ab

    # No consistent probabilities exist
    @test_throws ArgumentError generate_pairing(RakedPairing([1 1; 2 3; 3 2]), rps)
    @test_throws ArgumentError generate_pairing(RakedPairing([1 1; 2 2]), rps)
    @test_throws ArgumentError generate_pairing(RakedPairing([1 1 1]), rps)
    @test_throws ArgumentError generate_pairing(RakedPairing([1 4]), rps)
end

@testset "Methods used directly" begin
    rps = RepresentativePeriods(2, 8760, oscs)
    # A method can be passed wherever a pairing is expected
    @test length(scenario_paths(rps, CartesianPairing())) == 9
    @test path_marginals(rps, QuantilePairing()) ≈ scenario_probabilities(rps)
    @test length(scenario_paths(TwoLevel(2, 1, rps), CartesianPairing())) == 18
    # The method is resolved once when stored in the time structure
    rps_r =
        RepresentativePeriods(2, 8760, oscs; pairing = RandomPairing(4; rng = Xoshiro(1)))
    @test rps_r.pairing isa ScenarioPairing
    @test size(rps_r.pairing.scenarios) == (4, 2)
    @test path_scenarios.(scenario_paths(rps_r)) == path_scenarios.(scenario_paths(rps_r))
    rps_q = RepresentativePeriods(
        8760,
        [0.5, 0.5],
        [OperationalScenarios([day, day], [0.5, 0.5]), OperationalScenarios(3, day)];
        pairing = QuantilePairing(),
    )
    @test rps_q.pairing.probability ≈ [1 / 3, 1 / 6, 1 / 6, 1 / 3]
end

@testset "Pairing methods" begin
    probs = [[0.5, 0.3, 0.2], [0.6, 0.4]]
    for method in
        (CartesianPairing(), QuantilePairing(), RakedPairing([1 1; 2 1; 2 2; 3 2]))
        @test method isa PairingMethod
        @test generate_pairing(method, probs).check_marginals == :warn
    end
    @test generate_pairing(RandomPairing(5), probs).check_marginals == :none
    @test generate_pairing(CartesianPairing(), probs; check_marginals = :error).check_marginals ==
          :error
    @test_throws ArgumentError RakedPairing([[1, 1], [2]])
end

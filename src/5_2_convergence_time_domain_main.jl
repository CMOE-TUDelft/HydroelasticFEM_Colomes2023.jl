"""
Core computational body for Section 5.2 periodic-beam convergence (time domain).

This file intentionally contains the simulation/physics logic only.
Plotting and reporting are kept in `scripts/5-2-convergence/`.
"""

module ConvergenceTimeDomain

using Gridap

import HydroElasticFEM as HE
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH

export params
export build_regular_wave_state
export exact_wave_functions
export build_time_problem
export compute_errors
export run_case
export run_warmup_case

const params = (
    L = 2 * pi,
    H = 1.0,
    g = 9.81,
    ρ_w = 1.0e3,
    ρ_b = 1.0e2,
    h_b = 1.0e-2,
    k = 15,
    η0 = 0.01,
    Δt = 1.0e-6,
    tf = 1.0e-4,
    order_phi = 2,
    ns = [4, 8, 16, 32],
    orders = [2, 3, 4],
)

const _exact_wave_cache = Dict{NamedTuple, NamedTuple}()

"""Create a WaveSpec AiryState used to parameterize the exact regular-wave fields."""
function build_regular_wave_state(; p=params)
    ω = sqrt(p.g * p.k * tanh(p.k * p.H))
    T = 2 * pi / ω
    H_wave = 2 * p.η0

    spec = P.WaveSpec.ContinuousSpectrums.RegularWave(H_wave, T)
    ds = P.WaveSpec.SpectralSpreading.DiscreteSpectralSpreading(spec; mess=false)
    spread = P.WaveSpec.AngularSpreading.DiscreteAngularSpreading(0.0)
    θ_vec = [0.0]
    return P.WaveSpec.AiryWaves.AiryState(ds, spread, 1, 1, [ω], [p.k], θ_vec, p.H, 1)
end

"""Return constants and exact field functions used in the benchmark."""
function exact_wave_functions(; p=params)
    cache_key = (
        L = p.L,
        H = p.H,
        g = p.g,
        ρ_w = p.ρ_w,
        ρ_b = p.ρ_b,
        h_b = p.h_b,
        k = p.k,
        η0 = p.η0,
        tf = p.tf,
        ns_max = maximum(p.ns),
    )
    if haskey(_exact_wave_cache, cache_key)
        return _exact_wave_cache[cache_key]
    end

    sea = build_regular_wave_state(p=p)
    ω = sea.ω[1]
    k = sea.k[1]
    ψ = P.WaveSpec.AiryWaves.get_random_phases(sea)[1, 1]

    mρ = p.ρ_b * p.h_b / p.ρ_w
    EIρ = mρ * ω^2 / k^4

    η(x, t) = p.η0 * cos(k * x[1] - ω * t + ψ)
    ϕ(x, t) = p.η0 * ω / k * (cosh(k * x[2]) / sinh(k * p.H)) * sin(k * x[1] - ω * t + ψ)
    ∂tη(x, t) = p.η0 * ω * sin(k * x[1] - ω * t + ψ)
    ∂ttη(x, t) = -ω^2 * η(x, t)
    ∂tϕ(x, t) = -p.η0 * ω^2 / k * (cosh(k * x[2]) / sinh(k * p.H)) * cos(k * x[1] - ω * t + ψ)
    ∂ttϕ(x, t) = -ω^2 * ϕ(x, t)

    out = (; sea, ω, k, mρ, EIρ, η, ϕ, ∂tη, ∂ttη, ∂tϕ, ∂ttϕ)
    _exact_wave_cache[cache_key] = out
    return out
end

"""
Build the time-domain problem and run the simulation.
"""
function build_time_problem(
    n::Int,
    order::Int;
    p=params,
    verbose_steps::Bool=false,
    stage_label::String="time-domain",
)
    verbose_steps && println("[", stage_label, "] Building problem with n=", n, " order=", order)
    verbose_steps && println("[", stage_label, "]   - Exact wave functions")
    wv = exact_wave_functions(p=p)

    verbose_steps && println("[", stage_label, "]   - Building domains and physics")
    tank = HE.TankDomain(
        L=p.L,
        H=p.H,
        nx=2 * n,
        ny=n,
        is_periodic=(true, false),
        structure_domains=[
            HE.StructureDomain(L=p.L, x₀=[0.0, p.H], domain_symbol=:Γs),
        ],
    )

    beam = P.EulerBernoulliBeam(
        L=p.L,
        mᵨ=wv.mρ,
        EIᵨ=wv.EIρ,
        symbol=:w,
        fe=PH.FESpaceConfig(order=order, vector_type=Vector{Float64}),
        space_domain_symbol=:Γs,
    )

    potential = P.PotentialFlow(
        g=p.g,
        sea_state=wv.sea,
        fe=PH.FESpaceConfig(order=p.order_phi, vector_type=Vector{Float64}),
        space_domain_symbol=:Ω,
    )

    verbose_steps && println("[", stage_label, "]   - Building time config")
    cfg = PH.TimeDomainConfig(t₀=0.0, tf=p.tf)
    tcfg = PH.TimeConfig(
        Δt=p.Δt,
        t₀=0.0,
        tf=p.tf,
        ρ∞=1.0,
        u0=[x -> wv.ϕ(x, 0.0), x -> wv.η(x, 0.0)],
        u0t=[x -> wv.∂tϕ(x, 0.0), x -> wv.∂tη(x, 0.0)],
        u0tt=[x -> wv.∂ttϕ(x, 0.0), x -> wv.∂ttη(x, 0.0)],
    )

    verbose_steps && println("[", stage_label, "]   - Building problem")
    problem = S.build_problem(tank, P.PhysicsParameters[potential, beam], cfg; tconfig=tcfg)

    verbose_steps && println("[", stage_label, "]   - Problem built, getting simulation")
    result = S.simulate(problem, tcfg)

    return problem, result, p.tf
end

"""
Compute the L2 errors of the solution at the end of the simulation against the exact wave functions.
"""
function compute_errors(problem, result; p=params, kwargs...)
    t_end = get(kwargs, :t_end, p.tf)
    wv = exact_wave_functions(p=p)
    dom = S.get_integration_domains(problem)
    dΓ = dom[:dΓη]
    dΩ = dom[:dΩ]

    uh_end = nothing
    for (_, uh) in result.solution
        uh_end = uh
    end
    isnothing(uh_end) && error("Time-domain solution history is empty.")

    ϕh = uh_end[result.fmap[:ϕ]]
    w_h = uh_end[result.fmap[:w]]

    ew = w_h - (x -> wv.η(x, t_end))
    eϕ = ϕh - (x -> wv.ϕ(x, t_end))

    l2_w = sqrt(abs(sum(∫(ew * conj(ew))dΓ)))
    l2_ϕ = sqrt(abs(sum(∫(eϕ * conj(eϕ))dΩ)))

    return l2_w, l2_ϕ
end

"""
Run a single time-domain simulation case and compute errors.
"""
function run_case(n::Int, order::Int; p=params, verbose_steps::Bool=false, stage_label::String="run_case")
    problem, result, t_end = build_time_problem(
        n,
        order;
        p=p,
        verbose_steps=verbose_steps,
        stage_label=stage_label,
    )
    l2_w, l2_ϕ = compute_errors(problem, result; p=p, t_end=t_end)

    return Dict(
        :n => n,
        :order => order,
        :L2_error_w => l2_w,
        :L2_error_phi => l2_ϕ,
    )
end

"""
Run a single-step warm-up solve at coarse resolution.

The warm-up always uses `tf = Δt`, i.e., one single time step.
"""
function run_warmup_case(
    ;
    n::Int,
    order::Int,
    k::Real,
    order_phi::Int,
    Δt::Real,
    verbose_steps::Bool=false,
    stage_label::String="warmup",
)
    p_warm = merge(params, (
        k=k,
        order_phi=order_phi,
        Δt=Δt,
        tf=Δt,
    ))

    problem, result, t_end = build_time_problem(
        n,
        order;
        p=p_warm,
        verbose_steps=verbose_steps,
        stage_label=stage_label,
    )
    l2_w, l2_ϕ = compute_errors(problem, result; p=p_warm, t_end=t_end)
    return (; l2_w, l2_ϕ)
end

end # module ConvergenceTimeDomain

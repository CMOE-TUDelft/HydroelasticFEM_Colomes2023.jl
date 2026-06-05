"""
Core computational body for Section 5.2 periodic-beam convergence (time domain).
"""

module ConvergenceTimeDomain

using Gridap
using Parameters

import HydroElasticFEM as HE
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH

export PeriodicBeam_params
export run_periodic_beam

@with_kw struct PeriodicBeam_params
    name::String        = "PeriodicBeam"
    n::Int              = 4
    dt::Real            = 0.001
    tf::Real            = 1.0
    orderϕ::Int         = 2
    orderη::Int         = 2
    k::Int              = 1
    vtk_output::Bool    = false
    verbose_steps::Bool = false
end

function run_periodic_beam(params::PeriodicBeam_params)
    @unpack name, n, dt, tf, orderϕ, orderη, k, vtk_output, verbose_steps = params

    # Fixed parameters
    ## Geometry
    L = 2.0 * π
    H = 1.0

    ## Physics
    g   = 9.81
    ρ_w = 1.0e3
    ρ_b = 1.0e2
    h_b = 1.0e-2
    η₀  = 0.01
    t₀  = 0.0

    ## Derived quantities
    ω  = sqrt(g * k * tanh(k * H))
    d₀ = ρ_b * h_b / ρ_w          # normalised mass per unit length  (mᵨ)
    Dᵨ = d₀ * ω^2 / k^4           # normalised flexural rigidity     (EIᵨ)

    ## Exact wave fields — deterministic (ψ = 0)
    η(x, t)    =  η₀ * cos(k * x[1] - ω * t)
    ϕ(x, t)    =  η₀ * ω / k * cosh(k * x[2]) / sinh(k * H) * sin(k * x[1] - ω * t)
    ∂tη(x, t)  =  η₀ * ω * sin(k * x[1] - ω * t)
    ∂tϕ(x, t)  = -η₀ * ω^2 / k * cosh(k * x[2]) / sinh(k * H) * cos(k * x[1] - ω * t)
    ∂ttη(x, t) = -ω^2 * η(x, t)
    ∂ttϕ(x, t) = -ω^2 * ϕ(x, t)

    # Define fluid domain
    println("Defining fluid domain")
    tank = HE.TankDomain(
        L                 = L,
        H                 = H,
        nx                = 2 * n,
        ny                = n,
        is_periodic       = (true, false),
        structure_domains = [HE.StructureDomain(L=L, x₀=[0.0, H], domain_symbol=:Γs)],
    )

    # Define physics
    # No sea_state → no inlet wave-generation BC (correct for periodic benchmark)
    println("Defining physics")
    potential = P.PotentialFlow(
        g                   = g,
        fe                  = PH.FESpaceConfig(order=orderϕ, vector_type=Vector{Float64}),
        space_domain_symbol = :Ω,
    )
    beam = P.EulerBernoulliBeam(
        L                   = L,
        mᵨ                  = d₀,
        EIᵨ                 = Dᵨ,
        g                   = g,
        symbol              = :w,
        fe                  = PH.FESpaceConfig(order=orderη, vector_type=Vector{Float64},γ=1.0*orderη*(orderη+1)),
        space_domain_symbol = :Γs,
    )

    # Time configuration (ρ∞ = 1.0 → trapezoidal rule, γ = 0.5, β = 0.25)
    println("Defining time configuration")
    cfg  = PH.TimeDomainConfig(t₀=t₀, tf=tf)
    tcfg = PH.TimeConfig(
        Δt   = dt,
        t₀   = t₀,
        tf   = tf,
        ρ∞   = 1.0,
        u0   = [x -> ϕ(x, t₀),    x -> η(x, t₀)],
        u0t  = [x -> ∂tϕ(x, t₀),  x -> ∂tη(x, t₀)],
        u0tt = [x -> ∂ttϕ(x, t₀), x -> ∂ttη(x, t₀)],
    )

    # Assemble and solve
    println("Assembling and solving")
    problem = S.build_problem(tank, P.PhysicsParameters[potential, beam], cfg; tconfig=tcfg)
    result  = S.simulate(problem, tcfg)

    dom = S.get_integration_domains(problem)
    dΓ  = dom[:dΓη]
    dΩ  = dom[:dΩ]

    # L² norm helpers
    l2_Ω(u) = sqrt(abs(sum(∫(u * u)dΩ)))
    l2_Γ(u) = sqrt(abs(sum(∫(u * u)dΓ)))

    # Reference energy amplitudes from the exact solution
    E_kin_f₀ = 0.25 * d₀ * ω^2 * η₀^2 * L
    E_kin_s₀ = 0.25 * g  * η₀^2 * L
    E_pot_f₀ = 0.25 * Dᵨ * k^4 * η₀^2 * L
    E_ela_s₀ = 0.25 * g  * η₀^2 * L

    t_global = Float64[]
    e_ϕ      = Float64[]
    e_η      = Float64[]
    E_kin_f  = Float64[]
    E_pot_f  = Float64[]
    E_kin_s  = Float64[]
    E_ela_s  = Float64[]

    ηₙ = x -> η(x, t₀)
    tₙ = t₀

    if vtk_output == true
        filename = "data/VTKOutput/5-2-1-spatial_convergence/"*name
        pvd_Ω = createpvd(filename * "_O", append=false)
        pvd_Γ = createpvd(filename * "_G", append=false)
    end

    for (t, uh) in result.solution
        verbose_steps && println("t = ", t)

        ϕh = uh[result.fmap[:ϕ]]
        wh = uh[result.fmap[:w]]

        push!(e_ϕ, l2_Ω(ϕh - (x -> ϕ(x, t))))
        push!(e_η, l2_Γ(wh - (x -> η(x, t))))

        dt_local = t - tₙ
        ηₜ = dt_local > 0 ? (wh - ηₙ) / dt_local : (x -> 0.0)

        push!(E_kin_f, 0.5 * sum(∫(∇(ϕh) ⋅ ∇(ϕh))dΩ))
        push!(E_pot_f, 0.5 * g  * sum(∫(wh * wh)dΓ))
        push!(E_kin_s, 0.5 * d₀ * sum(∫(ηₜ * ηₜ)dΓ))
        push!(E_ela_s, 0.5 * Dᵨ * sum(∫(Δ(wh) * Δ(wh))dΓ))
        push!(t_global, t)

        ηₙ = wh
        tₙ = t
    end

    return e_ϕ, e_η, E_kin_f, E_pot_f, E_kin_s, E_ela_s, E_kin_f₀, E_kin_s₀, E_pot_f₀, E_ela_s₀, t_global
end

end # module ConvergenceTimeDomain

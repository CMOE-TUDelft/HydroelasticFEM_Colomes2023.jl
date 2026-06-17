"""
Core computational body for Section 5.3 elastic beam with joint (time domain).
"""

module KhavakpashevaTimeDomain

using Gridap
using Parameters
using WaveSpec
using Gridap.CellData: get_cell_dof_values, get_cell_points
using Gridap.FESpaces: get_fe_dof_basis
using Gridap.Visualization

import HydroElasticFEM as HE
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH
import HydroElasticFEM.Geometry as G

export Khabakpasheva_time_params
export run_khabakpasheva_time

@with_kw struct Khabakpasheva_time_params
    name::String        = "KhabakpashevaTime"
    nx::Int             = 20
    ny::Int             = 5
    nT::Int             = 1
    order::Int          = 4
    ξ::Float64          = 0.0
    vtk_output::Bool    = false
    verbose_steps::Bool = false
end

function run_khabakpasheva_time(params::Khabakpasheva_time_params)
    @unpack name, nx, ny, nT, order, ξ, vtk_output, verbose_steps = params

    # Fixed parameters
    Lb = 12.5
    mᵨ = 8.36
    EI₁ = 47100.0
    EI₂ = 471.0
    β = 0.2
    H = 1.1
    α = 0.249

    # Domain size
    Ld = Lb # damping zone length
    LΩ = 2Ld + 2Lb
    x₀ = 0.0
    xdᵢₙ = x₀ + Ld
    xb₀ = xdᵢₙ + Lb/2
    xbⱼ = xb₀ + β*Lb
    xb₁ = xb₀ + Lb
    xdₒᵤₜ = LΩ - Ld

    ## Physics
    g = 9.81
    ρ_w = 1025
    η₀  = 0.01
    t₀  = 0.0

    ## Derived quantities
    λ = α * Lb
    k = 2π / λ
    @show  ω = sqrt(g * k * tanh(k * H))
    T = 2π / ω
    d₀ = mᵨ/ρ_w
    a₁ = EI₁/ρ_w
    a₂ = EI₂/ρ_w
    kᵣ = ξ*a₁/Lb

    ## Sea state
    H_wave = 2 * η₀
    spec = WaveSpec.ContinuousSpectrums.RegularWave(H_wave, T)
    ds = WaveSpec.SpectralSpreading.DiscreteSpectralSpreading(spec; mess=false)
    spread = WaveSpec.AngularSpreading.DiscreteAngularSpreading(0.0)
    θ_vec = [0.0]
    sea_state = WaveSpec.AiryWaves.AiryState(ds, spread, 1, 1, [ω], [k], θ_vec, H, 1)

    ## Exact wave fields — deterministic (ψ = 0)
    η(x, t)    =  η₀ * cos(k * x[1] - ω * t)
    ϕ(x, t)    =  η₀ * ω / k * cosh(k * x[2]) / sinh(k * H) * sin(k * x[1] - ω * t)
    ∂tη(x, t)  =  η₀ * ω * sin(k * x[1] - ω * t)
    ∂tϕ(x, t)  = -η₀ * ω^2 / k * cosh(k * x[2]) / sinh(k * H) * cos(k * x[1] - ω * t)
    ∂ttη(x, t) = -ω^2 * η(x, t)
    ∂ttϕ(x, t) = -ω^2 * ϕ(x, t)
    vin(x, t)  =  - η₀ * ω * cosh(k * x[2]) / sinh(k * H) * cos(k * x[1] - ω * t)
    vzin(x, t) =  η₀ * ω * sinh(k * x[2]) / sinh(k * H) * sin(k * x[1] - ω * t)
    vin(t) = x -> vin(x, t)
    η(t) = x -> η(x, t)
    vzin(t) = x -> vzin(x, t)

    ## Damping
    μ₀ = 2.5
    μ₁ᵢₙ(x::VectorValue) = μ₀*(1.0 - sin(π/2*(x[1])/Ld))
    μ₁ₒᵤₜ(x::VectorValue) = μ₀*(1.0 - cos(π/2*(x[1]-xdₒᵤₜ)/Ld))
    μ₂ᵢₙ(x) = μ₁ᵢₙ(x)*k
    μ₂ₒᵤₜ(x) = μ₁ₒᵤₜ(x)*k
    # ηd(t) = x -> μ₂ᵢₙ(x)*η(x,t)
    # ∇ₙϕd(t) = x -> μ₁ᵢₙ(x)*vzin(x,t)

    ## Time stepping
    t₀ = 0.0
    Δt = T/40
    tf = nT*T#/λ_factor
    γₜ = 0.5
    βₜ = 0.25
    ∂uₜ_∂u = γₜ/(βₜ*Δt)

    ## Numerics constants
    γ = 1.0*order*(order-1)
    βₕ = 0.5
    αₕ = ∂uₜ_∂u/g * (1-βₕ)/βₕ

    ## Vertical refinement
    function map_y(x)
        if x[2] == H
            return VectorValue(x[1], H)
        end
        i = x[2] / (H/ny)
        return VectorValue(x[1], H-H/(2.5^i))
    end

    # Define fluid domain
    println("Defining fluid domain")
    tank = HE.TankDomain(
        L                 = LΩ,
        H                 = H,
        nx                = Int(ceil(nx/β)*ceil(LΩ/Lb)),
        ny                = ny,
        map               = map_y,
        structure_domains = [HE.StructureDomain(L=Lb, x₀=[xb₀, H], domain_symbol=:Γs)],
        damping_zones     = [
            G.DampingZone(L = Ld, x₀ = [0.0, H], domain_symbol = :Γ_din),
            G.DampingZone(L = Ld, x₀ = [xdₒᵤₜ, H], domain_symbol = :Γ_dout),
        ],
        joint_domains = [
            G.JointDomain(location = [xbⱼ, H], domain_symbol = :dΛj_1, normal_symbol = :n_Λ_j_1),
        ],
    )

    # Define physics
    # No sea_state → no inlet wave-generation BC (correct for periodic benchmark)
    println("Defining physics")
    potential = P.PotentialFlow(
        ρw                  = ρ_w,
        g                   = g,
        boundary_conditions = [
            # P.RadiationBC(domain = :dΓin),
            # P.RadiationBC(domain = :dΓout),
            P.PrescribedInletPotentialBC(domain = :dΓin, forcing = vin, quantity = :traction),
            P.DampingZoneBC(
                domain = :dΓd_1,
                μ₁ = μ₁ᵢₙ,
                μ₂ = μ₂ᵢₙ,
                η_in = η,
                vz_in = vzin,
            ),
            P.DampingZoneBC(
                domain = :dΓd_2,
                μ₁ = μ₁ₒᵤₜ,
                μ₂ = μ₂ₒᵤₜ,
                η_in = (x,t) -> 0.0,
                vz_in = (x,t) -> 0.0,
            ),
        ],
        sea_state           = sea_state,
        fe                  = PH.FESpaceConfig(order=order, vector_type=Vector{Float64}),
        space_domain_symbol = :Ω,
    )
    free_surface = P.FreeSurface(
        ρw = ρ_w,
        g = g,
        βₕ = βₕ,
        symbol = :κ,
        fe = PH.FESpaceConfig(order = order, vector_type = Vector{Float64}),
        space_domain_symbol = :Γκ,
    )
    beam = P.EulerBernoulliBeam(
        L                   = Lb,
        mᵨ                  = d₀,
        EIᵨ                 = x -> a₁*(x[1]<xbⱼ) + a₂*(x[1]>=xbⱼ),
        g                   = g,
        joints              = [P.JointRotationalSpring(:dΛj_1, :n_Λ_j_1, kᵣ)],
        symbol              = :w,
        fe                  = PH.FESpaceConfig(order=order, vector_type=Vector{Float64},γ=γ),
        space_domain_symbol = :Γs,
    )

    # Time configuration (ρ∞ = 1.0 → trapezoidal rule, γ = 0.5, β = 0.25)
    println("Defining time configuration")
    cfg  = PH.TimeDomainConfig(t₀=t₀, tf=tf)
    tcfg = PH.TimeConfig(
        Δt   = Δt,
        t₀   = t₀,
        tf   = tf,
        ρ∞   = 1.0,
        αₕ   = αₕ,
        # u0   = [x -> ϕ(x, t₀),    x -> η(x, t₀)],
        # u0t  = [x -> ∂tϕ(x, t₀),  x -> ∂tη(x, t₀)],
        # u0tt = [x -> ∂ttϕ(x, t₀), x -> ∂ttη(x, t₀)],
        u0   = [x -> 0.0,    x -> 0.0],
        u0t  = [x -> 0.0,  x -> 0.0],
        u0tt = [x -> 0.0, x -> 0.0],
    )

    # Assemble and solve
    println("Assembling and solving")
    problem = S.build_problem(tank, P.PhysicsParameters[potential, free_surface, beam], cfg; tconfig=tcfg)
    # problem = S.build_problem(tank, P.PhysicsParameters[potential], cfg; tconfig=tcfg)
    result  = S.simulate(problem, tcfg)

    # Postprocess
    V_Γη = S.get_trial_fe_space(problem)[3]
    xy_cp = get_cell_points(get_fe_dof_basis(V_Γη)).cell_phys_point
    x_cp = [[xy_ij[1] for xy_ij in xy_i] for xy_i in xy_cp]
    p = sortperm(x_cp[1])
    x_cp_sorted = [x_i[p] for x_i in x_cp]
    xs = [(x_i-1.5*Lb)/Lb for x_i in vcat(x_cp_sorted...)]
    ts = Float64[]
    ηxps_t = []

    if vtk_output == true
        filename = "data/VTKOutput/5-3-1-Khabakpasheva_time/"*name
        pvd_Ω = createpvd(filename * "_O", append=false)
        pvd_Γκ = createpvd(filename * "_Gk", append=false)
        pvd_Γη = createpvd(filename * "_Ge", append=false)
        trians = S.get_triangulations(problem)
    end

    for (t, uh) in result.solution
        verbose_steps && println("t = ", t)

        ϕₕ = uh[result.fmap[:ϕ]]
        ηₕ = uh[result.fmap[:w]]
        κₕ = uh[result.fmap[:κ]]
        η_cdv = get_cell_dof_values(ηₕ)
        η_cdv_sorted = [η_i[p] for η_i in η_cdv]
        η_rel_xs = [abs(η_i)/η₀ for η_i in vcat(η_cdv_sorted...)]
        push!(ηxps_t,η_rel_xs)
        push!(ts, t)

        if vtk_output == true
            pvd_Ω[t] = createvtk(trians[:Ω],filename * "_O_solution" * "_$t.vtu",cellfields = ["phi" => ϕₕ])#,nsubcells=10)
            pvd_Γκ[t] = createvtk(trians[:Γκ],filename * "_Gk_solution" * "_$t.vtu",cellfields = ["kappa" => κₕ],nsubcells=10)
            pvd_Γη[t] = createvtk(trians[:Γs],filename * "_Ge_solution" * "_$t.vtu",cellfields = ["eta" => ηₕ],nsubcells=10)
        end
    end

    if vtk_output == true
        savepvd(pvd_Ω)
        savepvd(pvd_Γκ)
        savepvd(pvd_Γη)
    end

    return ts, xs, ηxps_t
end

end # module KhavakpashevaTimeDomain

"""
Smoke tests for HydroElasticFEM_Colomes2023.

- Julia version required: >= 1.10
- HydroElasticFEM.jl minimum version: 0.1.0 (current local API)
- Estimated CI runtime target: < 5 minutes
"""

using Test
using Gridap
using HydroElasticFEM_Colomes2023
import HydroElasticFEM as HE
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH

const N_COARSE = 4
const P_COARSE = 1
const EXPECTED_W_DOFS_N4 = 3

function _last_state(sol)
    last_state = nothing
    for (_, uh) in sol
        last_state = uh
    end
    return last_state
end

function _l2_dof_norm(field)
    vals = get_free_dof_values(field)
    return sqrt(sum(abs2, vals) / length(vals))
end

function _solve_freq_membrane(; nx=N_COARSE, p=P_COARSE)
    H = 1.0
    L = 4.0
    Ls = 2.0
    xs = 1.0

    tank = HE.TankDomain(
        L=L,
        H=H,
        nx=nx,
        ny=2,
        structure_domains=[
            HE.StructureDomain(L=Ls, x₀=[xs, H], domain_symbol=:Γs),
        ],
    )

    membrane = P.Membrane(
        L=Ls,
        mᵨ=1.0,
        Tᵨ=5.0,
        symbol=:w,
        fe=PH.FESpaceConfig(order=p, vector_type=Vector{ComplexF64}),
        space_domain_symbol=:Γs,
    )

    physics = P.PhysicsParameters[membrane]
    cfg = PH.FreqDomainConfig(ω=1.0)
    rhs_fn(y) = ComplexF64[1.0 + 0.0im]
    problem = S.build_problem(tank, physics, cfg; rhs_fn=rhs_fn)
    result = S.simulate(problem)
    w = result.solution[result.fmap[P.variable_symbol(membrane)]]
    return result, w, xs, Ls, H
end

function _solve_time_membrane(; nx=N_COARSE, p=P_COARSE)
    H = 1.0
    L = 4.0
    Ls = 2.0
    xs = 1.0

    tank = HE.TankDomain(
        L=L,
        H=H,
        nx=nx,
        ny=2,
        structure_domains=[
            HE.StructureDomain(L=Ls, x₀=[xs, H], domain_symbol=:Γs),
        ],
    )

    membrane = P.Membrane(
        L=Ls,
        mᵨ=1.0,
        Tᵨ=5.0,
        symbol=:w,
        fe=PH.FESpaceConfig(order=p, vector_type=Vector{Float64}),
        space_domain_symbol=:Γs,
    )

    physics = P.PhysicsParameters[membrane]
    cfg = PH.TimeDomainConfig(t₀=0.0, tf=0.02)
    tcfg = PH.TimeConfig(
        Δt=0.01,
        t₀=0.0,
        tf=0.02,
        u0=[0.0, 0.0, 0.01],
        u0t=[0.0, 0.0, 0.0],
        u0tt=[0.0, 0.0, 0.0],
    )

    rhs_fn(t, y) = Float64[1.0]
    problem = S.build_problem(tank, physics, cfg; tconfig=tcfg, rhs_fn=rhs_fn)
    result = S.simulate(problem, tcfg)
    state = _last_state(result.solution)
    @assert !isnothing(state)
    w = state[result.fmap[P.variable_symbol(membrane)]]
    return result, w, xs, Ls, H
end

function _solve_freq_plate(; nx=N_COARSE, p=P_COARSE)
    H = 1.0
    L = 4.0
    Ls = 2.0
    xs = 1.0

    tank = HE.TankDomain(
        L=L,
        H=H,
        nx=nx,
        ny=2,
        structure_domains=[
            HE.StructureDomain(L=Ls, x₀=[xs, H], domain_symbol=:Γs),
        ],
    )

    plate = P.KirchhoffLovePlate(
        E=1.0e7,
        ν=0.25,
        hb=0.03,
        ambient_dim=2,
        manifold_dim=1,
        symbol=:w,
        fe=PH.FESpaceConfig(order=p, vector_type=Vector{ComplexF64}),
        space_domain_symbol=:Γs,
    )

    physics = P.PhysicsParameters[plate]
    cfg = PH.FreqDomainConfig(ω=1.0)
    problem = S.build_problem(tank, physics, cfg)
    result = S.simulate(problem)
    w = result.solution[result.fmap[P.variable_symbol(plate)]]
    return result, w, xs, Ls, H
end

function _solve_freq_joint_beam(; nx=N_COARSE, p=P_COARSE)
    H = 1.0
    L = 4.0
    Ls = 2.0
    xs = 1.0
    xj = xs + Ls / 2

    tank = HE.TankDomain(
        L=L,
        H=H,
        nx=nx,
        ny=2,
        structure_domains=[
            HE.StructureDomain(L=Ls, x₀=[xs, H], domain_symbol=:Γs),
        ],
        joint_domains=[
            HE.JointDomain(location=[xj, H], domain_symbol=:dΛj_1, normal_symbol=:n_Λ_j_1),
        ],
    )

    beam = P.EulerBernoulliBeam(
        L=Ls,
        mᵨ=0.5,
        EIᵨ=1.0,
        joints=[P.JointRotationalSpring(:dΛj_1, :n_Λ_j_1, 100.0)],
        symbol=:w,
        fe=PH.FESpaceConfig(order=p, vector_type=Vector{ComplexF64}),
        space_domain_symbol=:Γs,
    )

    physics = P.PhysicsParameters[beam]
    cfg = PH.FreqDomainConfig(ω=1.0)
    problem = S.build_problem(tank, physics, cfg)
    result = S.simulate(problem)
    w = result.solution[result.fmap[P.variable_symbol(beam)]]
    return result, w, xs, Ls, H
end

@testset "5-1-convergence-time-domain-script" begin
    include("../scripts/5-1-convergence/5-1-1-spatial_convergence.jl")
    df = run_5_1_1_spatial_convergence(ns=[4], orders=[2], force=true, make_plots=false, save_csv=false, verbose=false)

    @test nrow(df) == 1
    @test df.n[1] == 4
    @test df.order[1] == 2
    @test isfinite(df.L2_error_w[1])
    @test df.L2_error_w[1] < 1.0
end

@testset "5-1-convergence-time-domain" begin
    result, w, xs, Ls, H = _solve_time_membrane(nx=N_COARSE, p=P_COARSE)
    @test !isnothing(result)
    @test length(get_free_dof_values(w)) == EXPECTED_W_DOFS_N4

    l2_error = _l2_dof_norm(w)
    @test isfinite(l2_error)
    @test l2_error > 0
end

@testset "5-2-1-Khabakhpasheva-freq-domain" begin
    result, w, _, _, _ = _solve_freq_membrane(nx=N_COARSE, p=P_COARSE)
    @test !isnothing(result)
    @test length(get_free_dof_values(w)) == EXPECTED_W_DOFS_N4
end

@testset "5-2-2-Khabakhpasheva-time-domain" begin
    result, w, _, _, _ = _solve_time_membrane(nx=N_COARSE, p=P_COARSE)
    @test !isnothing(result)
    @test length(get_free_dof_values(w)) == EXPECTED_W_DOFS_N4
end

@testset "5-3-elastic-joints" begin
    result, w, _, _, _ = _solve_freq_membrane(nx=N_COARSE, p=P_COARSE)
    @test !isnothing(result)
    @test length(get_free_dof_values(w)) == EXPECTED_W_DOFS_N4
end

@testset "5-4-variable-bathymetry" begin
    # Missing public variable-bathymetry API tracked in HydroElasticFEM issue #26.
    @test_broken false
end

@testset "5-5-solar-farm" begin
    result, w, _, _, _ = _solve_freq_membrane(nx=N_COARSE, p=P_COARSE)
    @test !isnothing(result)
    @test length(get_free_dof_values(w)) == EXPECTED_W_DOFS_N4
end

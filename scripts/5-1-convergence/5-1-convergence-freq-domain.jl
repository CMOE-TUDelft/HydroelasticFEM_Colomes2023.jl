using DrWatson
using Gridap
using CairoMakie
using DataFrames
using DelimitedFiles
using Printf

using HydroElasticFEM_Colomes2023
import HydroElasticFEM as HE
import HydroElasticFEM.Physics as P
import HydroElasticFEM.Simulation as S
import HydroElasticFEM.ParameterHandler as PH

include("../../src/plot_theme.jl")

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
    ns = [4, 8, 16, 32, 64],
    orders = [2, 3, 4],
)

function wave_parameters(p=params)
    ω = sqrt(p.g * p.k * tanh(p.k * p.H))
    mρ = p.ρ_b * p.h_b / p.ρ_w
    EIρ = mρ * ω^2 / p.k^4
    return (; ω, mρ, EIρ)
end

function eta_exact(x, t; p=params)
    p.η0 * cos(p.k * x[1] - wave_parameters(p).ω * t)
end

function phi_exact(x, t; p=params)
    w = wave_parameters(p).ω
    p.η0 * w / p.k * (cosh(p.k * x[2]) / sinh(p.k * p.H)) * sin(p.k * x[1] - w * t)
end

function ddt_eta_exact(x, t; p=params)
    w = wave_parameters(p).ω
    p.η0 * w * sin(p.k * x[1] - w * t)
end

function ddtt_eta_exact(x, t; p=params)
    w = wave_parameters(p).ω
    -w^2 * eta_exact(x, t; p=p)
end

function ddt_phi_exact(x, t; p=params)
    w = wave_parameters(p).ω
    -p.η0 * w^2 / p.k * (cosh(p.k * x[2]) / sinh(p.k * p.H)) * cos(p.k * x[1] - w * t)
end

function ddtt_phi_exact(x, t; p=params)
    w = wave_parameters(p).ω
    -w^2 * phi_exact(x, t; p=p)
end

function build_time_problem(n::Int, order::Int; p=params)
    wp = wave_parameters(p)
    tank = HE.TankDomain(
        L=p.L,
        H=p.H,
        nx=2 * n,
        ny=2,
        is_periodic=(true, false),
        structure_domains=[
            HE.StructureDomain(L=p.L, x₀=[0.0, p.H], domain_symbol=:Γs),
        ],
    )

    beam = P.EulerBernoulliBeam(
        L=p.L,
        mᵨ=wp.mρ,
        EIᵨ=wp.EIρ,
        symbol=:w,
        fe=PH.FESpaceConfig(order=order, vector_type=Vector{Float64}),
        space_domain_symbol=:Γs,
    )

    potential = P.PotentialFlow(
        g=p.g,
        fe=PH.FESpaceConfig(order=p.order_phi, vector_type=Vector{Float64}),
        space_domain_symbol=:Ω,
    )

    cfg = PH.TimeDomainConfig(t₀=0.0, tf=p.tf)
    tcfg = PH.TimeConfig(
        Δt=p.Δt,
        t₀=0.0,
        tf=p.tf,
        ρ∞=1.0,
        u0=[x -> phi_exact(x, 0.0; p=p), x -> eta_exact(x, 0.0; p=p)],
        u0t=[x -> ddt_phi_exact(x, 0.0; p=p), x -> ddt_eta_exact(x, 0.0; p=p)],
        u0tt=[x -> ddtt_phi_exact(x, 0.0; p=p), x -> ddtt_eta_exact(x, 0.0; p=p)],
    )

    problem = S.build_problem(tank, P.PhysicsParameters[potential, beam], cfg; tconfig=tcfg)
    result = S.simulate(problem, tcfg)
    return problem, result, p.tf
end

function compute_errors(problem, result; p=params, kwargs...)
    t_end = get(kwargs, :t_end, p.tf)
    dom = S.get_integration_domains(problem)
    dΓ = dom[:dΓη]
    dΩ = dom[:dΩ]

    uh_end = nothing
    for (_, uh) in result.solution
        uh_end = uh
    end
    isnothing(uh_end) && error("Time-domain solution history is empty.")

    phi_h = uh_end[result.fmap[:ϕ]]
    w_h = uh_end[result.fmap[:w]]

    ew = w_h - (x -> eta_exact(x, t_end; p=p))
    ephi = phi_h - (x -> phi_exact(x, t_end; p=p))

    l2_w = sqrt(abs(sum(∫(ew * conj(ew))dΓ)))
    l2_phi = sqrt(abs(sum(∫(ephi * conj(ephi))dΩ)))

    return l2_w, l2_phi
end

function run_case(n::Int, order::Int; p=params)
    problem, result, t_end = build_time_problem(n, order; p=p)
    l2_w, l2_phi = compute_errors(problem, result; p=p, t_end=t_end)

    return Dict(
        :n => n,
        :order => order,
        :L2_error_w => l2_w,
        :L2_error_phi => l2_phi,
    )
end

function save_convergence_csv(df::DataFrame, out_csv::String)
    mkpath(dirname(out_csv))
    open(out_csv, "w") do io
        writedlm(io, ["n" "order" "L2_error_w" "L2_error_phi"], ',')
        writedlm(io, Matrix(df[:, [:n, :order, :L2_error_w, :L2_error_phi]]), ',')
    end
    return out_csv
end

function add_slope_triangle!(ax, x0, y0, pplus1; color=:black)
    dx = 0.08 * x0
    dy = y0 * ((x0 / (x0 - dx))^pplus1 - 1)
    lines!(ax, [x0 - dx, x0], [y0, y0], color=color, linewidth=1.2)
    lines!(ax, [x0, x0], [y0, y0 + dy], color=color, linewidth=1.2)
    lines!(ax, [x0 - dx, x0], [y0, y0 + dy], color=color, linewidth=1.2)
    text!(ax, x0 - 0.9dx, y0 + 0.45dy, text="$(pplus1)", color=color, fontsize=10)
end

function plot_convergence(df::DataFrame; p=params)
    mkpath("plots/5-1-convergence")

    fig = Figure(size=(850, 550))
    ax = Axis(
        fig[1, 1],
        xlabel="h = L / n",
        ylabel="L² error in w",
        xscale=log10,
        yscale=log10,
        title="Figure 5.2: Time-domain periodic beam convergence",
    )

    hs = p.L ./ p.ns
    sorted_ns = sort(p.ns)
    sorted_h = p.L ./ sorted_ns

    for order in p.orders
        sub = filter(row -> row.order == order, df)
        sort!(sub, :n)
        scatterlines!(ax, p.L ./ sub.n, sub.L2_error_w, marker=:circle, label="p = $(order)")
    end

    # Reference slope triangles for rates p+1 matching each structural order.
    y_top = maximum(df.L2_error_w)
    y_scales = (0.45, 0.25, 0.14)
    for (i, order) in enumerate(sort(p.orders))
        add_slope_triangle!(ax, sorted_h[i + 1], y_scales[i] * y_top, order + 1)
    end

    axislegend(ax, position=:rb)

    pdf_path = "plots/5-1-convergence/fig5_convergence_time.pdf"
    png_path = "plots/5-1-convergence/fig5_convergence_time.png"
    save(pdf_path, fig)
    save(png_path, fig, px_per_unit=300 / 96)

    return pdf_path, png_path
end

function print_summary(df::DataFrame)
    println("\nConvergence summary (time domain):")
    println(df)

    println("\nEstimated orders from adjacent refinements (w error):")
    for order in unique(df.order)
        sub = sort(filter(row -> row.order == order, df), :n)
        hs = params.L ./ sub.n
        errs = sub.L2_error_w
        rates = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(errs) - 1)]
        @printf("  p = %d -> rates = %s\n", order, string(round.(rates, digits=3)))
    end
end

function run_convergence_time(; ns=params.ns, orders=params.orders, force=false, make_plots=true, save_csv=true, verbose=true)
    mkpath("data/5-1-convergence")

    rows = Dict[]
    for order in orders
        for n in ns
            cfg = Dict(
                :n => n,
                :order => order,
                :L => params.L,
                :H => params.H,
                :k => params.k,
                :Δt => params.Δt,
                :tf => params.tf,
                :order_phi => params.order_phi,
            )

            out, _ = produce_or_load("data/5-1-convergence", cfg; force=force, filename="time_n$(n)_p$(order)") do c
                run_case(c[:n], c[:order])
            end

            push!(rows, out)
        end
    end

    df = DataFrame(rows)
    sort!(df, [:order, :n])

    if save_csv
        save_convergence_csv(df, "data/5-1-convergence/convergence_time.csv")
    end

    if make_plots
        with_theme(VLFS_THEME) do
            plot_convergence(df)
        end
    end

    if verbose
        print_summary(df)
    end

    return df
end

run_convergence_frequency(; kwargs...) = run_convergence_time(; kwargs...)

if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence_time()
end

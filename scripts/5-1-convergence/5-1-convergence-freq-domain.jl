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
    L = 1.0,
    H = 0.5,
    EI = 1.0,
    ρ_s = 1.0,
    ω = 1.0,
    ns = [4, 8, 16, 32, 64],
    orders = [1, 2, 3],
)

const G_ACCEL = 9.81

"""Analytical periodic beam mode used as manufactured reference."""
w_exact(x; L=params.L) = sin(2 * pi * x[1] / L)

"""Potential proxy paired to the periodic beam mode."""
phi_exact(x; L=params.L, ω=params.ω) = (im * ω / G_ACCEL) * w_exact(x; L=L)

function forcing_amplitude(; L=params.L, EI=params.EI, ρ_s=params.ρ_s, ω=params.ω, g=G_ACCEL)
    k = 2 * pi / L
    return g + EI * k^4 - ρ_s * ω^2
end

function build_frequency_problem(n::Int, order::Int; p=params)
    tank = HE.TankDomain(
        L=p.L,
        H=p.H,
        nx=n,
        ny=2,
        structure_domains=[
            HE.StructureDomain(L=p.L, x₀=[0.0, p.H], domain_symbol=:Γs),
        ],
    )

    beam = P.EulerBernoulliBeam(
        L=p.L,
        mᵨ=p.ρ_s,
        EIᵨ=p.EI,
        symbol=:w,
        fe=PH.FESpaceConfig(order=order, vector_type=Vector{ComplexF64}),
        space_domain_symbol=:Γs,
    )

    k = 2 * pi / p.L
    eta0 = 0.05
    phi_in(x) = -im * (eta0 * p.ω / k) * (cosh(k * x[2]) / sinh(k * p.H)) * exp(im * k * x[1])
    vin(x) = (eta0 * p.ω) * (cosh(k * x[2]) / sinh(k * p.H)) * exp(im * k * x[1])
    f_in(x) = -vin(x) - im * k * phi_in(x)

    potential = P.PotentialFlow(
        g=G_ACCEL,
        boundary_conditions=[
            P.PrescribedInletPotentialBC(domain=:dΓin, forcing=f_in, quantity=:traction),
        ],
        fe=PH.FESpaceConfig(order=order, vector_type=Vector{ComplexF64}),
        space_domain_symbol=:Ω,
    )

    free_surface = P.FreeSurface(
        g=G_ACCEL,
        βₕ=0.5,
        fe=PH.FESpaceConfig(order=order, vector_type=Vector{ComplexF64}),
        space_domain_symbol=:Γκ,
    )

    cfg = PH.FreqDomainConfig(ω=p.ω)
    amp = forcing_amplitude(L=p.L, EI=p.EI, ρ_s=p.ρ_s, ω=p.ω)
    rhs_w(x) = ComplexF64(amp * sin(k * x[1]))
    rhs_fn(y) = Any[0.0 + 0.0im, 0.0 + 0.0im, rhs_w]

    problem = S.build_problem(tank, P.PhysicsParameters[potential, free_surface, beam], cfg; rhs_fn=rhs_fn)
    result = S.simulate(problem)
    return problem, result
end

function compute_errors(problem, result; p=params, kwargs...)
    dom = S.get_integration_domains(problem)
    dΓ = dom[:dΓη]
    dΩ = dom[:dΩ]

    w_h = result.solution[result.fmap[:w]]
    phi_h = result.solution[result.fmap[:ϕ]]

    ew = w_h - w_exact
    ephi = phi_h - phi_exact

    l2_w = sqrt(abs(sum(∫(ew * conj(ew))dΓ)))
    l2_phi = sqrt(abs(sum(∫(ephi * conj(ephi))dΩ)))

    return l2_w, l2_phi
end

function run_case(n::Int, order::Int; p=params)
    problem, result = build_frequency_problem(n, order; p=p)
    l2_w, l2_phi = compute_errors(problem, result; p=p)

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
        title="Figure 5: Frequency-domain convergence",
    )

    hs = p.L ./ p.ns
    sorted_ns = sort(p.ns)
    sorted_h = p.L ./ sorted_ns

    for order in p.orders
        sub = filter(row -> row.order == order, df)
        sort!(sub, :n)
        scatterlines!(ax, p.L ./ sub.n, sub.L2_error_w, marker=:circle, label="p = $(order)")
    end

    # Reference slope triangles for rates p+1 = 2, 3, 4.
    y_top = maximum(df.L2_error_w)
    add_slope_triangle!(ax, sorted_h[2], 0.45 * y_top, 2)
    add_slope_triangle!(ax, sorted_h[3], 0.25 * y_top, 3)
    add_slope_triangle!(ax, sorted_h[4], 0.14 * y_top, 4)

    axislegend(ax, position=:rb)

    pdf_path = "plots/5-1-convergence/fig5_convergence_freq.pdf"
    png_path = "plots/5-1-convergence/fig5_convergence_freq.png"
    save(pdf_path, fig)
    save(png_path, fig, px_per_unit=300 / 96)

    return pdf_path, png_path
end

function print_summary(df::DataFrame)
    println("\nConvergence summary (frequency domain):")
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

function run_convergence_frequency(; ns=params.ns, orders=params.orders, force=false, make_plots=true, save_csv=true, verbose=true)
    mkpath("data/5-1-convergence")

    rows = Dict[]
    for order in orders
        for n in ns
            cfg = Dict(
                :n => n,
                :order => order,
                :L => params.L,
                :H => params.H,
                :EI => params.EI,
                :rho_s => params.ρ_s,
                :ω => params.ω,
            )

            out, _ = produce_or_load("data/5-1-convergence", cfg; force=force, filename="freq_n$(n)_p$(order)") do c
                run_case(c[:n], c[:order])
            end

            push!(rows, out)
        end
    end

    df = DataFrame(rows)
    sort!(df, [:order, :n])

    if save_csv
        save_convergence_csv(df, "data/5-1-convergence/convergence_freq.csv")
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

if abspath(PROGRAM_FILE) == @__FILE__
    run_convergence_frequency()
end

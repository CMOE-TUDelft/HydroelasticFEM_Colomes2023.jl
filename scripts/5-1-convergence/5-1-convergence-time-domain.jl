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
    ns = [4, 8, 16, 32],
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
        ny=n,
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

function write_case_pvd(problem, result; n::Int, order::Int, p=params, step_stride::Int=20, write_all_steps::Bool=false)
    outdir = joinpath("data", "VTK", "5-1-convergence", @sprintf("n%d_p%d", n, order))
    mkpath(outdir)

    trians = S.get_triangulations(problem)
    Ω = trians[:Ω]
    Γs = try
        trians[:Γs]
    catch
        trians[:Γη]
    end

    fluid_series = joinpath(outdir, "fluid_series")
    beam_series = joinpath(outdir, "beam_series")

    createpvd(joinpath(outdir, "fluid_solution")) do pvd_fluid
        createpvd(joinpath(outdir, "beam_solution")) do pvd_beam
            kstep = 0
            for (t, uh) in result.solution
                kstep += 1
                write_step = write_all_steps ? ((kstep - 1) % step_stride == 0) : false
                write_step = write_step || (kstep == 1)
                if !write_step
                    continue
                end

                ϕh = uh[result.fmap[:ϕ]]
                wh = uh[result.fmap[:w]]

                ϕex = x -> phi_exact(x, t; p=p)
                wex = x -> eta_exact(x, t; p=p)

                fluid_file = joinpath(fluid_series, @sprintf("fluid_t%05d", kstep))
                beam_file = joinpath(beam_series, @sprintf("beam_t%05d", kstep))
                mkpath(dirname(fluid_file))
                mkpath(dirname(beam_file))

                pvd_fluid[t] = createvtk(
                    Ω,
                    fluid_file,
                    cellfields=[
                        "phi" => ϕh,
                        "phi_exact" => ϕex,
                        "phi_error" => (ϕh - ϕex),
                    ],
                )

                pvd_beam[t] = createvtk(
                    Γs,
                    beam_file,
                    cellfields=[
                        "w" => wh,
                        "w_exact" => wex,
                        "w_error" => (wh - wex),
                    ],
                )
            end
        end
    end

    return outdir
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

function add_rate_triangle_on_curve!(ax, hs, errs, rate; color=:black)
    length(hs) < 2 && return
    idx = max(2, length(hs) - 1)
    x0 = hs[idx]
    y0 = errs[idx]
    x1 = idx < length(hs) ? sqrt(hs[idx] * hs[idx + 1]) : 0.7 * x0
    y1 = y0 * (x1 / x0)^rate
    lines!(ax, [x1, x0], [y0, y0], color=color, linewidth=1.6)
    lines!(ax, [x1, x1], [y1, y0], color=color, linewidth=1.6)
    lines!(ax, [x1, x0], [y1, y0], color=color, linewidth=1.6)
    text!(ax, sqrt(x0 * x1), sqrt(y0 * y1), text="$(rate)", color=color, fontsize=11, align=(:center, :center))
end

function plot_convergence(df::DataFrame; p=params)
    mkpath("plots/5-1-convergence")

    fig = Figure(size=(1120, 520), fontsize=16)
    ax_w = Axis(
        fig[1, 1],
        xlabel="h = L / n",
        ylabel="L² error in deflection",
        xscale=log10,
        yscale=log10,
        title="(a) Deflection convergence",
    )
    ax_phi = Axis(
        fig[1, 2],
        xlabel="h = L / n",
        ylabel="L² error in potential",
        xscale=log10,
        yscale=log10,
        title="(b) Potential-flow convergence",
    )
    hidexdecorations!(ax_w, grid=false)

    markers = [:circle, :rect, :utriangle, :diamond, :cross]
    colors = [:navy, :firebrick, :darkgreen, :goldenrod, :black]

    sorted_ns = sort(p.ns)
    sorted_h = p.L ./ sorted_ns
    series_w = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
    series_phi = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

    for (i, order) in enumerate(p.orders)
        sub = filter(row -> row.order == order, df)
        sort!(sub, :n)
        hvals = collect(Float64.(p.L ./ sub.n))
        e_w = collect(Float64.(sub.L2_error_w))
        e_phi = collect(Float64.(sub.L2_error_phi))
        series_w[order] = (hvals, e_w)
        series_phi[order] = (hvals, e_phi)
        scatterlines!(
            ax_w,
            hvals,
            e_w,
            marker=markers[i],
            markersize=10,
            color=colors[i],
            linewidth=2.6,
            label="p = $(order)",
        )
        scatterlines!(
            ax_phi,
            hvals,
            e_phi,
            marker=markers[i],
            markersize=10,
            color=colors[i],
            linewidth=2.6,
            label="p = $(order)",
        )
    end

    # Reference slope triangles (aligned with each plotted curve) for both fields.
    for (i, order) in enumerate(sort(p.orders))
        local_color = colors[i]
        hs_w, errs_w = series_w[order]
        hs_phi, errs_phi = series_phi[order]
        add_rate_triangle_on_curve!(ax_w, hs_w, errs_w, order + 1; color=local_color)
        add_rate_triangle_on_curve!(ax_phi, hs_phi, errs_phi, order + 1; color=local_color)
    end

    xlims!(ax_w, minimum(sorted_h) * 0.9, maximum(sorted_h) * 1.1)
    xlims!(ax_phi, minimum(sorted_h) * 0.9, maximum(sorted_h) * 1.1)
    ylims!(ax_w, 0.5 * minimum(df.L2_error_w), 2.0 * maximum(df.L2_error_w))
    ylims!(ax_phi, 0.5 * minimum(df.L2_error_phi), 2.0 * maximum(df.L2_error_phi))

    axislegend(ax_w, position=:rb, framevisible=true, title="Polynomial order")
    axislegend(ax_phi, position=:rb, framevisible=true, title="Polynomial order")

    Label(
        fig[0, :],
        "Figure 5.2: Periodic-beam spatial convergence in time domain",
        fontsize=20,
        font=:bold,
        tellwidth=false,
    )

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

function run_convergence_time(
    ;
    ns=params.ns,
    orders=params.orders,
    force=false,
    make_plots=true,
    save_csv=true,
    verbose=true,
    save_pvd=false,
    pvd_step_stride=20,
    pvd_all_steps=false,
)
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

            if save_pvd
                problem, result, t_end = build_time_problem(n, order; p=params)
                l2_w, l2_phi = compute_errors(problem, result; p=params, t_end=t_end)
                write_case_pvd(
                    problem,
                    result;
                    n=n,
                    order=order,
                    p=params,
                    step_stride=pvd_step_stride,
                    write_all_steps=pvd_all_steps,
                )
                out = Dict(
                    :n => n,
                    :order => order,
                    :L2_error_w => l2_w,
                    :L2_error_phi => l2_phi,
                )
            else
                out, _ = produce_or_load("data/5-1-convergence", cfg; force=force, filename="time_n$(n)_p$(order)") do c
                    run_case(c[:n], c[:order])
                end
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

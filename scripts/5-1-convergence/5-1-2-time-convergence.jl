"""
Section 5.1.2: periodic-beam time-step convergence
"""

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

if !isdefined(Main, :params)
    include("../../src/convergence_5_1_time_domain_main.jl")
end
if !isdefined(Main, :VLFS_THEME)
    include("../../src/plot_theme.jl")
end

function add_rate_triangle_time!(ax, hs, errs, rate; color=:black)
    length(hs) < 2 && return
    idx = max(2, length(hs) - 1)
    x0 = hs[idx]
    x1 = idx < length(hs) ? sqrt(hs[idx] * hs[idx + 1]) : 0.7 * x0

    y_base = 1.20 * errs[idx]
    y_rate = y_base * (x1 / x0)^rate

    lines!(ax, [x1, x0], [y_base, y_base], color=color, linewidth=1.6)
    lines!(ax, [x1, x1], [y_rate, y_base], color=color, linewidth=1.6)
    lines!(ax, [x1, x0], [y_rate, y_base], color=color, linewidth=1.6)

    text!(
        ax,
        sqrt(x0 * x1),
        1.03 * y_base,
        text="1",
        color=color,
        fontsize=11,
        align=(:center, :bottom),
    )

    text!(
        ax,
        x1 / 1.06,
        sqrt(y_base * y_rate),
        text="$(rate)",
        color=color,
        fontsize=11,
        align=(:right, :center),
    )
end

function plot_time_convergence(df::DataFrame)
    mkpath("plots/5-1-convergence")

    sort!(df, :Δt, rev=true)
    Δt_vals = collect(Float64.(df.Δt))
    e_w = collect(Float64.(df.L2_error_w))
    e_ϕ = collect(Float64.(df.L2_error_phi))

    fig = Figure(size=(1120, 520), fontsize=16)
    ax_w = Axis(
        fig[1, 1],
        xlabel="Δt",
        ylabel="L² error in deflection",
        xscale=log10,
        yscale=log10,
        title="(a) Time convergence in deflection",
    )
    ax_ϕ = Axis(
        fig[1, 2],
        xlabel="Δt",
        ylabel="L² error in potential",
        xscale=log10,
        yscale=log10,
        title="(b) Time convergence in potential",
    )

    scatterlines!(ax_w, Δt_vals, e_w, marker=:circle, markersize=10, color=:navy, linewidth=2.6)
    scatterlines!(ax_ϕ, Δt_vals, e_ϕ, marker=:rect, markersize=10, color=:firebrick, linewidth=2.6)

    add_rate_triangle_time!(ax_w, Δt_vals, e_w, 2; color=:navy)
    add_rate_triangle_time!(ax_ϕ, Δt_vals, e_ϕ, 2; color=:firebrick)

    Label(
        fig[0, :],
        "Section 5.1: Time-step convergence test",
        fontsize=20,
        font=:bold,
        tellwidth=false,
    )

    pdf_path = "plots/5-1-convergence/fig5_time_convergence.pdf"
    png_path = "plots/5-1-convergence/fig5_time_convergence.png"
    save(pdf_path, fig)
    save(png_path, fig, px_per_unit=300 / 96)
    return pdf_path, png_path
end

function run_5_1_2_time_convergence(
    ;
    Δts=[1.0 * 2.0^(-i) for i in 0:4],
    n=64,
    order=4,
    order_phi=order,
    k=1,
    tf=1.0,
    force=false,
    make_plots=true,
    save_csv=true,
    verbose=true,
    verbose_steps=false,
)
    mkpath("data/5-1-convergence")

    warmup_n = minimum(params.ns)
    warmup_order = minimum(params.orders)
    warmup_k = k
    warmup_order_phi = min(order_phi, minimum(params.orders))
    warmup_Δt = minimum(Δts)

    if verbose
        println("[5-1-2] Stage 1/3: warm-up solve (n=$(warmup_n), order=$(warmup_order), Δt=$(warmup_Δt), tf=$(warmup_Δt))")
    end
    warm = run_warmup_case(
        n=warmup_n,
        order=warmup_order,
        k=warmup_k,
        order_phi=warmup_order_phi,
        Δt=warmup_Δt,
        verbose_steps=verbose_steps,
        stage_label="5-1-2 warmup",
    )
    if verbose
        println("[5-1-2] Warm-up complete: L2_w=$(warm.l2_w), L2_phi=$(warm.l2_ϕ)")
        println("[5-1-2] Stage 2/3: time-step convergence sweep")
    end
    rows = Dict[]

    for Δt in Δts
        if verbose
            println("[5-1-2] Solving case Δt=$(Δt), n=$(n), order=$(order)")
        end
        cfg = Dict(
            :n => n,
            :order => order,
            :order_phi => order_phi,
            :k => k,
            :Δt => Δt,
            :tf => tf,
        )

        out, _ = produce_or_load(
            "data/5-1-convergence",
            cfg;
            force=force,
            filename="time_step_n$(n)_p$(order)_dt$(Δt)",
        ) do c
            p_time = merge(params, (
                Δt=c[:Δt],
                tf=c[:tf],
                k=c[:k],
                order_phi=c[:order_phi],
            ))
            problem, result, t_end = build_time_problem(
                c[:n],
                c[:order];
                p=p_time,
                verbose_steps=verbose_steps,
                stage_label="5-1-2 Δt=$(c[:Δt])",
            )
            l2_w, l2_ϕ = compute_errors(problem, result; p=p_time, t_end=t_end)

            Dict(
                :Δt => c[:Δt],
                :n_steps => Int(round(c[:tf] / c[:Δt])),
                :n => c[:n],
                :order => c[:order],
                :L2_error_w => l2_w,
                :L2_error_phi => l2_ϕ,
            )
        end

        push!(rows, out)
    end

    df = DataFrame(rows)
    sort!(df, :Δt, rev=true)

    if save_csv
        out_csv = "data/5-1-convergence/convergence_time_step.csv"
        mkpath(dirname(out_csv))
        open(out_csv, "w") do io
            writedlm(io, ["Δt" "n_steps" "n" "order" "L2_error_w" "L2_error_phi"], ',')
            writedlm(io, Matrix(df[:, [:Δt, :n_steps, :n, :order, :L2_error_w, :L2_error_phi]]), ',')
        end
    end

    if make_plots
        with_theme(VLFS_THEME) do
            plot_time_convergence(df)
        end
    end

    if verbose
        println("[5-1-2] Stage 3/3: summary and outputs complete")
    end

    if verbose
        println("\nTime convergence summary:")
        println(df)
        if nrow(df) > 1
            rates_w = [
                log(df.L2_error_w[i] / df.L2_error_w[i + 1]) / log(df.Δt[i] / df.Δt[i + 1])
                for i in 1:(nrow(df) - 1)
            ]
            rates_ϕ = [
                log(df.L2_error_phi[i] / df.L2_error_phi[i + 1]) / log(df.Δt[i] / df.Δt[i + 1])
                for i in 1:(nrow(df) - 1)
            ]
            @printf("Estimated temporal rates (w): %s\n", string(round.(rates_w, digits=3)))
            @printf("Estimated temporal rates (ϕ): %s\n", string(round.(rates_ϕ, digits=3)))
        end
    end

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_5_1_2_time_convergence()
end

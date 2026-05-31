"""
Section 5.1.3: periodic-beam energy conservation
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

function compute_energy_series(problem, result; p=params, verbose_steps::Bool=false, stage_label::String="5-1-3")
    
    verbose_steps && println("[", stage_label, "] Computing energy series from solution steps...")
    verbose_steps && println("[", stage_label, "]   - Getting integration domains")
    dom = S.get_integration_domains(problem)
    dΓ = dom[:dΓη]
    dΩ = dom[:dΩ]

    wv = exact_wave_functions(p=p)
    d₀ = wv.mρ
    Dᵨ = wv.EIρ
    g = p.g

    verbose_steps && println("[", stage_label, "]   - Iterating over solution steps")
    rows = Dict[]
    ηₙ = x -> wv.η(x, 0.0)
    tₙ = 0.0
    for (istep, (t, uh)) in enumerate(result.solution)
        if verbose_steps
            println("[", stage_label, "] step=", istep, " t=", t)
        end

        ϕh = uh[result.fmap[:ϕ]]
        wh = uh[result.fmap[:w]]

        dt_local = t - tₙ
        ηₜ = dt_local > 0 ? (wh - ηₙ) / dt_local : (x -> 0.0)

        E_kin_f = 0.5 * real(sum(∫(∇(ϕh) ⋅ conj(∇(ϕh)))dΩ))
        E_pot_f = 0.5 * g * real(sum(∫(wh * conj(wh))dΓ))
        E_kin_s = 0.5 * d₀ * real(sum(∫(ηₜ * conj(ηₜ))dΓ))
        E_ela_s = 0.5 * Dᵨ * real(sum(∫(Δ(wh) * conj(Δ(wh)))dΓ))

        E_ϕ = E_kin_f + E_pot_f
        E_w = E_kin_s + E_ela_s
        E_total = E_ϕ + E_w

        push!(rows, Dict(
            :t => t,
            :E_kin_f => E_kin_f,
            :E_pot_f => E_pot_f,
            :E_kin_s => E_kin_s,
            :E_ela_s => E_ela_s,
            :E_phi => E_ϕ,
            :E_w => E_w,
            :E_total => E_total,
        ))

        ηₙ = wh
        tₙ = t
    end

    verbose_steps && println("[", stage_label, "] Energy series computed for ", length(rows), " steps.")
    df = DataFrame(rows)
    sort!(df, :t)

    # Exclude the first time-history entry for postprocessing consistency
    # with the reference workflow.
    if nrow(df) > 1
        df = df[2:end, :]
    end

    E0 = df.E_total[1]
    if E0 == 0
        df[!, :rel_drift] = zeros(nrow(df))
    else
        df[!, :rel_drift] = (df.E_total .- E0) ./ E0
    end
    return df
end

function plot_energy_conservation(df::DataFrame)
    mkpath("plots/5-1-convergence")

    t = collect(Float64.(df.t))
    E0 = df.E_total[1] == 0 ? 1.0 : df.E_total[1]

    fig = Figure(size=(980, 620), fontsize=16)
    ax_E = Axis(
        fig[1, 1],
        xlabel="t",
        ylabel="Normalized energy",
        title="(a) Energy components",
    )
    ax_drift = Axis(
        fig[2, 1],
        xlabel="t",
        ylabel="(E(t)-E(0))/E(0)",
        title="(b) Relative total-energy drift",
    )

    lines!(ax_E, t, df.E_phi ./ E0, color=:firebrick, linewidth=2.2, label="Eϕ / E0")
    lines!(ax_E, t, df.E_w ./ E0, color=:navy, linewidth=2.2, label="Ew / E0")
    lines!(ax_E, t, df.E_total ./ E0, color=:black, linewidth=2.4, label="Etotal / E0")
    axislegend(ax_E, position=:rb, framevisible=true)

    lines!(ax_drift, t, df.rel_drift, color=:darkgreen, linewidth=2.4)

    Label(
        fig[0, :],
        "Section 5.1: Energy conservation test",
        fontsize=20,
        font=:bold,
        tellwidth=false,
    )

    pdf_path = "plots/5-1-convergence/fig5_energy_conservation.pdf"
    png_path = "plots/5-1-convergence/fig5_energy_conservation.png"
    save(pdf_path, fig)
    save(png_path, fig, px_per_unit=300 / 96)
    return pdf_path, png_path
end

function run_5_1_3_energy_conservation(
    ;
    n=16,
    order=2,
    order_phi=order,
    k=15,
    Δt=1.0e-3,
    tf=(2 * pi / sqrt(9.81 * k * tanh(k * 1.0))) * 2,
    force=false,
    make_plots=true,
    save_csv=true,
    verbose=true,
    verbose_steps=false,
)
    warmup_n = minimum(params.ns)
    warmup_order = minimum(params.orders)
    warmup_k = k
    warmup_order_phi = min(order_phi, minimum(params.orders))
    warmup_Δt = Δt

    if verbose
        println("[5-1-3] Stage 1/3: warm-up solve (n=$(warmup_n), order=$(warmup_order), Δt=$(warmup_Δt), tf=$(warmup_Δt))")
    end
    warm = run_warmup_case(
        n=warmup_n,
        order=warmup_order,
        k=warmup_k,
        order_phi=warmup_order_phi,
        Δt=warmup_Δt,
        verbose_steps=verbose_steps,
        stage_label="5-1-3 warmup",
    )
    if verbose
        println("[5-1-3] Warm-up complete: L2_w=$(warm.l2_w), L2_phi=$(warm.l2_ϕ)")
        println("[5-1-3] Stage 2/3: energy-conservation solve")
        println("[5-1-3] Energy conservation solve parameters: n=$(n), order=$(order), order_phi=$(order_phi), k=$(k), Δt=$(Δt), tf=$(tf)")
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
        filename="energy_n$(n)_p$(order)_dt$(Δt)_tf$(tf)",
    ) do c
        p_energy = merge(params, (
            Δt=c[:Δt],
            tf=c[:tf],
            k=c[:k],
            order_phi=c[:order_phi],
        ))
        problem, result, _ = build_time_problem(
            c[:n],
            c[:order];
            p=p_energy,
            verbose_steps=verbose_steps,
            stage_label="5-1-3 main",
        )
        df_local = compute_energy_series(
            problem,
            result;
            p=p_energy,
            verbose_steps=verbose_steps,
            stage_label="5-1-3 main",
        )

        Dict(
            :t => collect(Float64.(df_local.t)),
            :E_kin_f => collect(Float64.(df_local.E_kin_f)),
            :E_pot_f => collect(Float64.(df_local.E_pot_f)),
            :E_kin_s => collect(Float64.(df_local.E_kin_s)),
            :E_ela_s => collect(Float64.(df_local.E_ela_s)),
            :E_phi => collect(Float64.(df_local.E_phi)),
            :E_w => collect(Float64.(df_local.E_w)),
            :E_total => collect(Float64.(df_local.E_total)),
            :rel_drift => collect(Float64.(df_local.rel_drift)),
        )
    end

    df = DataFrame(
        t=Float64.(out[:t]),
        E_kin_f=Float64.(out[:E_kin_f]),
        E_pot_f=Float64.(out[:E_pot_f]),
        E_kin_s=Float64.(out[:E_kin_s]),
        E_ela_s=Float64.(out[:E_ela_s]),
        E_phi=Float64.(out[:E_phi]),
        E_w=Float64.(out[:E_w]),
        E_total=Float64.(out[:E_total]),
        rel_drift=Float64.(out[:rel_drift]),
    )

    if save_csv
        out_csv = "data/5-1-convergence/energy_conservation.csv"
        mkpath(dirname(out_csv))
        open(out_csv, "w") do io
            writedlm(io, ["t" "E_kin_f" "E_pot_f" "E_kin_s" "E_ela_s" "E_phi" "E_w" "E_total" "rel_drift"], ',')
            writedlm(io, Matrix(df[:, [:t, :E_kin_f, :E_pot_f, :E_kin_s, :E_ela_s, :E_phi, :E_w, :E_total, :rel_drift]]), ',')
        end
    end

    if make_plots
        with_theme(VLFS_THEME) do
            plot_energy_conservation(df)
        end
    end

    if verbose
        println("[5-1-3] Stage 3/3: summary and outputs complete")
    end

    if verbose
        println("\nEnergy conservation summary:")
        println(first(df, min(10, nrow(df))))
        @printf("max |relative drift| = %.3e\n", maximum(abs.(df.rel_drift)))
    end

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_5_1_3_energy_conservation()
end

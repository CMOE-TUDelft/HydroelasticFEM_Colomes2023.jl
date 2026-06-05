"""
Section 5.2.2: periodic-beam time-step convergence
"""

if !isdefined(@__MODULE__, :VLFS_THEME)
    include("../../src/plot_theme.jl")
end

# ─── Plotting helpers ─────────────────────────────────────────────────────────

function _add_rate_tri_time!(ax, dts, errs, rate; color=:black)
    length(dts) < 2 && return
    idx = 2#max(2, length(dts) - 1)
    x0 = dts[idx]
    x1 = idx < length(dts) ? sqrt(dts[idx] * dts[idx+1]) : 1.2 * x0
    y_base = 0.8 * errs[idx]
    y_rate = y_base * (x1 / x0)^rate
    lines!(ax, [x1, x0], [y_base, y_base], color=color, linewidth=1.6)
    lines!(ax, [x1, x1], [y_rate, y_base], color=color, linewidth=1.6)
    lines!(ax, [x1, x0], [y_rate, y_base], color=color, linewidth=1.6)
    text!(ax, sqrt(x0*x1), y_base*0.85,      text="1",       color=color, fontsize=11, align=(:center, :bottom))
    text!(ax, x1*1.06,     sqrt(y_base*y_rate), text="$(rate)", color=color, fontsize=11, align=(:right,  :center))
end

function plot_time_convergence(df::DataFrame; L=2π)
    mkpath("plots/5-2-convergence")
    markers = [:circle, :rect, :utriangle]
    colors  = [:navy, :firebrick, :darkgreen]
    
    sub = sort(df, :dt)
    dt  = collect(Float64.(sub.dt))

    fig  = Figure(size=(1120, 520), fontsize=16)
    ax_η = Axis(fig[1,1], xlabel="Δt", ylabel="L² error (deflection η)",
                xticks=(dt, string.(dt)),
                xscale=log10, yscale=log10, title="(a) Beam deflection convergence")
    ax_ϕ = Axis(fig[1,2], xlabel="Δt", ylabel="L² error (potential ϕ)",
                xticks=(dt, string.(dt)),
                xscale=log10, yscale=log10, title="(b) Potential-flow convergence")

    e_η = collect(Float64.(sub.e_η_i))
    e_ϕ = collect(Float64.(sub.e_ϕ_i))
    scatterlines!(ax_η, dt, e_η, marker=markers[1], markersize=10, color=colors[1], linewidth=2.6, label="p = 2")
    scatterlines!(ax_ϕ, dt, e_ϕ, marker=markers[1], markersize=10, color=colors[1], linewidth=2.6, label="p = 2")
    _add_rate_tri_time!(ax_η, dt, e_η, 2; color=colors[1])
    _add_rate_tri_time!(ax_ϕ, dt, e_ϕ, 2; color=colors[1])

    axislegend(ax_η, position=:rb)
    axislegend(ax_ϕ, position=:rb)
    save("plots/5-2-convergence/fig5_time_convergence.pdf", fig)
    save("plots/5-2-convergence/fig5_time_convergence.png", fig, px_per_unit=300/96)
end

function _print_time_summary(df::DataFrame; L=2π)
    println("\nTime convergence summary:")
    sub  = sort(df, :dt)
    dts  = collect(Float64.(sub.dt))
    errs = collect(Float64.(sub.e_η_i))
    rates = [log(errs[i]/errs[i+1]) / log(dts[i]/dts[i+1]) for i in 1:length(errs)-1]
    @printf("  p = %d -> rates(η) = %s\n", 2, string(round.(rates, digits=3)))
end

# ─── Main runner ──────────────────────────────────────────────────────────────

function run_5_2_2_time_convergence(
    ;
    force=false,
    make_plots=true,
    save_csv=true,
    verbose=true,
    verbose_steps=false,      
)
    # ── Define per-case execution function (mirrors MonolithicFEMVLFS) ───────
    function run_5_2_2(case::PeriodicBeam_params)
        case_name = savename(case)
        println("-------------")
        println("Case: ", case_name)
        e_ϕ, e_η, = run_periodic_beam(case)
        e_ϕ_i = last(e_ϕ)
        e_η_i = last(e_η)
        return Dict(
            "name"   => case.name,
            "n"      => case.n,
            "dt"     => Float64(case.dt),
            "tf"     => Float64(case.tf),
            "orderϕ" => case.orderϕ,
            "orderη" => case.orderη,
            "k"      => case.k,
            "e_ϕ_i"  => e_ϕ_i,
            "e_η_i"  => e_η_i,
        )
    end

    # ── Warm-up case ─────────────────────────────────────────────────────────
    k  = 1
    H  = 1.0
    g  = 9.81
    ω  = sqrt(g * k * tanh(k * H))
    T  = 2π / ω
    Δt = T / 1
    n  = 3
    order = 2
    path = datadir("5-2-2-periodic-beam-time-convergence")
    case = PeriodicBeam_params(name="Warm-up", n=n, dt=Δt, tf=T, k=k, orderϕ=order, orderη=order)
    verbose && println("[5-2-2] Warm-up: n=$(n), order=$(order), k=$(k), Δt=$(Δt)")
    produce_or_load(path, case, run_5_2_2; force=force, digits=8)

    # ── Time convergence: k=1, small h high order, varying Δt ────────
    Δts=[1.0 * 2.0^(-i) for i in 0:4]
    n=64
    order=4
    order_phi=order
    k=1
    tf=1.0
    verbose && println("[5-2-2] Time convergence: k=$(k), n=$(n), order=$(order), order_phi=$(order_phi), tf=$(tf)")

    rows = Dict[]
    for Δt in Δts
      case  = PeriodicBeam_params(
          name="timeConvergence",
          n=n, dt=Δt, tf=tf, k=k,
          orderϕ=order, orderη=order,
      )
      verbose && println("[5-2-2] Solving: n=$(n), order=$(order), Δt=$(Δt)")
      data, _ = produce_or_load(path, case, run_5_2_2; force=force, digits=8)
      push!(rows, data)
    end

    df = DataFrame(rows)
    sort!(df, [:dt])

    if save_csv
        mkpath("data/5-2-convergence")
        open("data/5-2-convergence/convergence_time.csv", "w") do io
            writedlm(io, ["Δt" "L2_error_w" "L2_error_phi"], ',')
            writedlm(io, [df.dt df.e_η_i df.e_ϕ_i], ',')
        end
    end

    if make_plots
        with_theme(VLFS_THEME) do
            plot_time_convergence(df)
        end
    end

    if verbose
        _print_time_summary(df)
    end

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_5_2_2_time_convergence()
end
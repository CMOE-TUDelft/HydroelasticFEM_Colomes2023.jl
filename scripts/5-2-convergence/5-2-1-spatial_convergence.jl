"""
Section 5.2.1: periodic-beam spatial convergence (time domain)
"""

if !isdefined(@__MODULE__, :VLFS_THEME)
    include("../../src/plot_theme.jl")
end

# ─── Plotting helpers ─────────────────────────────────────────────────────────

function _add_rate_tri_spatial!(ax, hs, errs, rate; color=:black)
    length(hs) < 2 && return
    idx = max(2, length(hs) - 1)
    x0 = hs[idx]
    x1 = idx < length(hs) ? sqrt(hs[idx] * hs[idx+1]) : 0.7 * x0
    y_base = 1.20 * errs[idx]
    y_rate = y_base * (x1 / x0)^rate
    lines!(ax, [x1, x0], [y_base, y_base], color=color, linewidth=1.6)
    lines!(ax, [x1, x1], [y_rate, y_base], color=color, linewidth=1.6)
    lines!(ax, [x1, x0], [y_rate, y_base], color=color, linewidth=1.6)
    text!(ax, sqrt(x0*x1), 1.03*y_base,      text="1",       color=color, fontsize=11, align=(:center, :bottom))
    text!(ax, x1/1.02,     sqrt(y_base*y_rate), text="$(rate)", color=color, fontsize=11, align=(:right,  :center))
end

function plot_spatial_convergence(df::DataFrame; L=2π)
    mkpath("plots/5-2-convergence")
    markers = [:circle, :rect, :utriangle]
    colors  = [:navy, :firebrick, :darkgreen]

    n = unique(collect(Int.(df.n)))

    fig  = Figure(size=(1120, 520), fontsize=16)
    ax_η = Axis(fig[1,1], xlabel="h = L/n", ylabel="L² error (deflection η)",
                xticks=(L ./ n, ["L/$(ni)" for ni in n]),
                xscale=log10, yscale=log10, title="(a) Beam deflection convergence")
    ax_ϕ = Axis(fig[1,2], xlabel="h = L/n", ylabel="L² error (potential ϕ)",
                xticks=(L ./ n, ["L/$(ni)" for ni in n]),
                xscale=log10, yscale=log10, title="(b) Potential-flow convergence")

    for (i, order) in enumerate(sort(unique(Int.(df.orderη))))
        sub = sort(filter(r -> r.orderη == order && r.orderϕ == order, df), :n)
        h   = L ./ Float64.(sub.n)
        e_η = collect(Float64.(sub.e_η_i))
        e_ϕ = collect(Float64.(sub.e_ϕ_i))
        scatterlines!(ax_η, h, e_η, marker=markers[i], markersize=10, color=colors[i], linewidth=2.6, label="p = $order")
        scatterlines!(ax_ϕ, h, e_ϕ, marker=markers[i], markersize=10, color=colors[i], linewidth=2.6, label="p = $order")
        _add_rate_tri_spatial!(ax_η, h, e_η, order+1; color=colors[i])
        _add_rate_tri_spatial!(ax_ϕ, h, e_ϕ, order+1; color=colors[i])
    end

    axislegend(ax_η, position=:rb)
    axislegend(ax_ϕ, position=:rb)
    save("plots/5-2-convergence/fig5_spatial_convergence.pdf", fig)
    save("plots/5-2-convergence/fig5_spatial_convergence.png", fig, px_per_unit=300/96)
end

function _print_spatial_summary(df::DataFrame; L=2π)
    println("\nSpatial convergence summary:")
    for order in sort(unique(Int.(df.orderη)))
        sub  = sort(filter(r -> r.orderη == order && r.orderϕ == order, df), :n)
        hs   = L ./ Float64.(sub.n)
        errs = collect(Float64.(sub.e_η_i))
        rates = [log(errs[i]/errs[i+1]) / log(hs[i]/hs[i+1]) for i in 1:length(errs)-1]
        @printf("  p = %d -> rates(η) = %s\n", order, string(round.(rates, digits=3)))
    end
end

# ─── Main runner ──────────────────────────────────────────────────────────────

function run_5_2_1_spatial_convergence(
    ;
    force=false,
    make_plots=true,
    save_csv=true,
    verbose=true,
    verbose_steps=false,    
    vtk_output=false,  
)
    
    function run_5_2_1(case::PeriodicBeam_params)
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
    path = datadir("5-2-1-periodic-beam-spatial-convergence")
    case = PeriodicBeam_params(name="Warm-up", n=n, dt=Δt, tf=T, k=k, orderϕ=order, orderη=order)
    verbose && println("[5-2-1] Warm-up: n=$(n), order=$(order), k=$(k), Δt=$(Δt)")
    produce_or_load(path, case, run_5_2_1; force=force, digits=8)

    # ── Spatial convergence: k=15, very small Δt, varying n and order ────────
    Δt_sw = 1.0e-6
    tf_sw = 1.0e-4
    k_sw  = 15
    verbose && println("[5-2-1] Spatial convergence: k=$(k_sw), Δt=$(Δt_sw), tf=$(tf_sw)")

    rows = Dict[]
    for order in 2:4
        for i in 1:4
            nelem = 2^(i + 1)
            case  = PeriodicBeam_params(
                name="spatialConvergence",
                n=nelem, dt=Δt_sw, tf=tf_sw, k=k_sw,
                orderϕ=order, orderη=order,
                verbose_steps=verbose_steps,
                vtk_output=vtk_output,
            )
            verbose && println("[5-2-1] Solving: n=$(nelem), order=$(order)")
            data, _ = produce_or_load(path, case, run_5_2_1; force=force, digits=8)
            push!(rows, data)
        end
    end

    df = DataFrame(rows)
    sort!(df, [:orderη, :n])

    if save_csv
        mkpath("data/5-2-convergence")
        open("data/5-2-convergence/convergence_time.csv", "w") do io
            writedlm(io, ["n" "order" "L2_error_w" "L2_error_phi"], ',')
            writedlm(io, [Int.(df.n) Int.(df.orderη) df.e_η_i df.e_ϕ_i], ',')
        end
    end

    if make_plots
        with_theme(VLFS_THEME) do
            plot_spatial_convergence(df)
        end
    end

    if verbose
        _print_spatial_summary(df)
    end

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_5_2_1_spatial_convergence()
end
"""
Section 5.3.1: elastic beam with joint in time domain
"""

if !isdefined(@__MODULE__, :VLFS_THEME)
    include("../../src/plot_theme.jl")
end

# ─── Plotting helpers ─────────────────────────────────────────────────────────

function plot_khabakpasheva_time(df::DataFrame)
    mkpath("plots/5-3-Khabakhpasheva")
    fig = Figure(size=(1120, 520), fontsize=16)
    ax1 = Axis(fig[1,1], xlabel="x", ylabel="η(x,t)", title="Time domain solution with hinge ξ=0")
    ax2 = Axis(fig[1,2], xlabel="x", ylabel="η(x,t)", title="Time domain solution without hinge ξ=625")

    ts = df.t[df.ξ .== 0][1]
    x1 = df.x[df.ξ .== 0][1]
    η1 = df.η[df.ξ .== 0][1]
    η2 = df.η[df.ξ .== 625][1]

    η1xps = permutedims(reshape(hcat(η1...),length(x1),length(ts)))
    η2xps = permutedims(reshape(hcat(η2...),length(x1),length(ts)))
    η1_max = [maximum(abs.(η1xps[1000:2000,it])) for it in 1:length(x1)]
    η2_max = [maximum(abs.(η2xps[1000:2000,it])) for it in 1:length(x1)]

    lines!(ax1, x1, η1_max, color=:navy, linewidth=2.0, label="Envelope of η(x,t) for ξ=0")
    lines!(ax2, x1, η2_max, color=:firebrick, linewidth=2.0, label="Envelope of η(x,t) for ξ=625")

    println("Selected time steps for plotting: ")
    its = 1000:5:1015
    for it in its
        println("Plotting t=$(round(ts[it], digits=2))")
        lines!(ax1, x1, η1xps[it, :], color=:navy, linewidth=1.0, label="t=$(round(ts[it], digits=2))s") 
        lines!(ax2, x1, η2xps[it, :], color=:firebrick, linewidth=1.0, label="t=$(round(ts[it], digits=2))s")
    end

    save("plots/5-3-Khabakhpasheva/fig5_khabakpasheva_time_xi_0.pdf", fig)
    save("plots/5-3-Khabakhpasheva/fig5_khabakpasheva_time_xi_0.png", fig, px_per_unit=300/96)
    save("plots/5-3-Khabakhpasheva/fig5_khabakpasheva_time_xi_625.pdf", fig)
    save("plots/5-3-Khabakhpasheva/fig5_khabakpasheva_time_xi_625.png", fig, px_per_unit=300/96)
end

function _print_khabakpasheva_time_summary(df::DataFrame)
    println("\nKhabakpasheva time domain summary:")
    ts = df.t[df.ξ .== 0][1]
    x1 = df.x[df.ξ .== 0][1]
    η1 = df.η[df.ξ .== 0][1]
    η2 = df.η[df.ξ .== 625][1]

    η1xps = permutedims(reshape(hcat(η1...),length(x1),length(ts)))
    η2xps = permutedims(reshape(hcat(η2...),length(x1),length(ts)))
    η1_max = [maximum(abs.(η1xps[1000:2000,it])) for it in 1:length(x1)]
    η2_max = [maximum(abs.(η2xps[1000:2000,it])) for it in 1:length(x1)]

    println("Case ξ=0: t range = [$(minimum(ts)), $(maximum(ts))], x range = [$(minimum(x1)), $(maximum(x1))], max |η(x,t)| across x: ", maximum(η1_max))
    println("Case ξ=625: t range = [$(minimum(ts)), $(maximum(ts))], x range = [$(minimum(x1)), $(maximum(x1))], max |η(x,t)| across x: ", maximum(η2_max))
end

# ─── Main runner ──────────────────────────────────────────────────────────────

function run_5_3_1_khabakpasheva_time(
    ;
    force=false,
    make_plots=true,
    save_csv=true,
    verbose=true,
    verbose_steps=false, 
    vtk_output=false,     
    test_suit=false, # If true, only run the warm-up case for testing purposes
)
    # ── Define per-case execution function (mirrors MonolithicFEMVLFS) ───────
    function run_5_3_1(case::Khabakpasheva_time_params)
        case_name = savename(case)
        println("-------------")
        println("Case: ", case_name)
        ts, xs, ηxps_t = run_khabakpasheva_time(case)
        return Dict(
            "name"   => case.name,
            "nx"     => case.nx,
            "ny"     => case.ny,
            "order"  => case.order,
            "ξ"      => case.ξ,
            "t"      => ts,
            "x"      => xs,
            "η"      => ηxps_t,
        )
    end

    # ── Warm-up case ─────────────────────────────────────────────────────────
    nx = 10
    ny = 1
    nT = 1
    order = 2
    path = datadir("5-3-1-Khabakhpasheva-time-domain")
    case = Khabakpasheva_time_params(
        name="Warm-up",
        nx=nx,
        ny=ny,
        nT=nT,
        order=order,
    )
    verbose && println("[5-3-1] Warm-up: nx=$(nx), ny=$(ny), order=$(order)")
    produce_or_load(path, case, run_5_3_1; force=true, digits=8)

    if test_suit
        return nothing
    end

    # ── Time domain solution with hinge ξ=0 ────────
    case = Khabakpasheva_time_params(
        name="xi-0",
        nT=50,
        order=4,
        verbose_steps=verbose_steps,
        vtk_output=vtk_output,
    )
    verbose && println("[5-3-1] Time domain solution with hinge ξ=0: ")
    data1, _ = produce_or_load(path, case, run_5_3_1, force=force, digits=8)

    # ── Time domain solution without hinge ξ=625 ────────
    case = Khabakpasheva_time_params(
        ξ=625,
        name="xi-625",
        nT=50,
        order=4,
        verbose_steps=verbose_steps,
        vtk_output=vtk_output,
    )
    verbose && println("[5-3-1] Time domain solution without hinge ξ=625: ")
    data2, _ = produce_or_load(path, case, run_5_3_1, force=force, digits=8)

    # ── Gather data ─────────────────────────────────────────────────────────
    df = @linq collect_results(path) |> where(:order .== 4) # Filter for order 4 cases only
    println("DataFrame summary:")
    println(describe(df))
    # df2 = DataFrame(data2)

    if make_plots
        with_theme(VLFS_THEME) do
            plot_khabakpasheva_time(df)
        end
    end

    if save_csv
        mkpath("data/5-3-Khabakhpasheva")
        open("data/5-3-Khabakhpasheva/khabakpasheva_time_xi_0.csv", "w") do io
            writedlm(io, ["t" "x" "η"], ',')
            writedlm(io, [df.t[df.ξ .== 0] df.x[df.ξ .== 0] df.η[df.ξ .== 0]], ',')
        end
        open("data/5-3-Khabakhpasheva/khabakpasheva_time_xi_625.csv", "w") do io
            writedlm(io, ["t" "x" "η"], ',')
            writedlm(io, [df.t[df.ξ .== 625] df.x[df.ξ .== 625] df.η[df.ξ .== 625]], ',')
        end
    end

    if verbose
        _print_khabakpasheva_time_summary(df)
    end

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_5_3_1_khabakpasheva_time()
end
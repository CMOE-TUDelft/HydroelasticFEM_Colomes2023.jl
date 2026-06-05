"""
Section 5.2.1: periodic-beam spatial convergence (time domain)
"""

using DrWatson
using CairoMakie
using DataFrames
using DelimitedFiles
using Printf

const var"5_2"          = ConvergenceTimeDomain
const PeriodicBeam_params = var"5_2".PeriodicBeam_params
const run_periodic_beam   = var"5_2".run_periodic_beam

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
    text!(ax, x1/1.06,     sqrt(y_base*y_rate), text="$(rate)", color=color, fontsize=11, align=(:right,  :center))
end

function plot_spatial_convergence(df::DataFrame; L=2π)
    mkpath("plots/5-2-convergence")
    markers = [:circle, :rect, :utriangle]
    colors  = [:navy, :firebrick, :darkgreen]

    fig  = Figure(size=(1120, 520), fontsize=16)
    ax_η = Axis(fig[1,1], xlabel="h = L/n", ylabel="L² error (deflection η)",
                xscale=log10, yscale=log10, title="(a) Beam deflection convergence")
    ax_ϕ = Axis(fig[1,2], xlabel="h = L/n", ylabel="L² error (potential ϕ)",
                xscale=log10, yscale=log10, title="(b) Potential-flow convergence")

    for (i, order) in enumerate(sort(unique(Int.(df.orderη))))
        sub = sort(filter(r -> r.orderη == order && r.orderϕ == order, df), :n)
        h   = L ./ Float64.(sub.n)
        e_η = collect(Float64.(sub.e_η_i))
        e_ϕ = collect(Float64.(sub.e_ϕ_i))
        scatterlines!(ax_η, h, e_η, marker=markers[i], markersize=10, color=colors[i], linewidth=2.6, label="p = $order")
        scatterlines!(ax_ϕ, h, e_ϕ, marker=markers[i], markersize=10, color=colors[i], linewidth=2.6, label="p = $order")
        _add_rate_tri_spatial!(ax_η, h, e_η, order+1; color=colors[i])
    end

    axislegend(ax_η, position=:lb)
    axislegend(ax_ϕ, position=:lb)
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


# function write_case_pvd(problem, result; n::Int, order::Int, p=params, step_stride::Int=20, write_all_steps::Bool=false)
#     outdir = joinpath("data", "VTK", "5-2-convergence", @sprintf("n%d_p%d", n, order))
#     mkpath(outdir)

#     trians = S.get_triangulations(problem)
#     Ω = trians[:Ω]
#     Γs = try
#         trians[:Γs]
#     catch
#         trians[:Γη]
#     end

#     fluid_series = joinpath(outdir, "fluid_series")
#     beam_series = joinpath(outdir, "beam_series")
#     wv = exact_wave_functions(p=p)

#     createpvd(joinpath(outdir, "fluid_solution")) do pvd_fluid
#         createpvd(joinpath(outdir, "beam_solution")) do pvd_beam
#             kstep = 0
#             for (t, uh) in result.solution
#                 kstep += 1
#                 write_step = write_all_steps ? ((kstep - 1) % step_stride == 0) : false
#                 write_step = write_step || (kstep == 1)
#                 if !write_step
#                     continue
#                 end

#                 ϕh = uh[result.fmap[:ϕ]]
#                 wh = uh[result.fmap[:w]]

#                 ϕex = x -> wv.ϕ(x, t)
#                 wex = x -> wv.η(x, t)

#                 fluid_file = joinpath(fluid_series, @sprintf("fluid_t%05d", kstep))
#                 beam_file = joinpath(beam_series, @sprintf("beam_t%05d", kstep))
#                 mkpath(dirname(fluid_file))
#                 mkpath(dirname(beam_file))

#                 pvd_fluid[t] = createvtk(
#                     Ω,
#                     fluid_file,
#                     cellfields=[
#                         "phi" => ϕh,
#                         "phi_exact" => ϕex,
#                         "phi_error" => (ϕh - ϕex),
#                     ],
#                 )

#                 pvd_beam[t] = createvtk(
#                     Γs,
#                     beam_file,
#                     cellfields=[
#                         "w" => wh,
#                         "w_exact" => wex,
#                         "w_error" => (wh - wex),
#                     ],
#                 )
#             end
#         end
#     end

#     return outdir
# end

# function save_convergence_csv(df::DataFrame, out_csv::String)
#     mkpath(dirname(out_csv))
#     open(out_csv, "w") do io
#         writedlm(io, ["n" "order" "L2_error_w" "L2_error_phi"], ',')
#         writedlm(io, Matrix(df[:, [:n, :order, :L2_error_w, :L2_error_phi]]), ',')
#     end
#     return out_csv
# end

# function add_rate_triangle_on_curve!(ax, hs, errs, rate; color=:black)
#     length(hs) < 2 && return
#     idx = max(2, length(hs) - 1)
#     x0 = hs[idx]
#     x1 = idx < length(hs) ? sqrt(hs[idx] * hs[idx + 1]) : 0.7 * x0

#     y_base = 1.20 * errs[idx]
#     y_rate = y_base * (x1 / x0)^rate

#     lines!(ax, [x1, x0], [y_base, y_base], color=color, linewidth=1.6)
#     lines!(ax, [x1, x1], [y_rate, y_base], color=color, linewidth=1.6)
#     lines!(ax, [x1, x0], [y_rate, y_base], color=color, linewidth=1.6)

#     text!(
#         ax,
#         sqrt(x0 * x1),
#         1.03 * y_base,
#         text="1",
#         color=color,
#         fontsize=11,
#         align=(:center, :bottom),
#     )

#     text!(
#         ax,
#         x1 / 1.06,
#         sqrt(y_base * y_rate),
#         text="$(rate)",
#         color=color,
#         fontsize=11,
#         align=(:right, :center),
#     )
# end

# function plot_convergence(df::DataFrame; p=params)
#     mkpath("plots/5-2-convergence")

#     fig = Figure(size=(1120, 520), fontsize=16)
#     ax_w = Axis(
#         fig[1, 1],
#         xlabel="h = L / n",
#         ylabel="L² error in deflection",
#         xscale=log10,
#         yscale=log10,
#         title="(a) Deflection convergence",
#     )
#     ax_phi = Axis(
#         fig[1, 2],
#         xlabel="h = L / n",
#         ylabel="L² error in potential",
#         xscale=log10,
#         yscale=log10,
#         title="(b) Potential-flow convergence",
#     )
#     markers = [:circle, :rect, :utriangle, :diamond, :cross]
#     colors = [:navy, :firebrick, :darkgreen, :goldenrod, :black]

#     sorted_ns = sort(p.ns)
#     sorted_h = p.L ./ sorted_ns
#     series_w = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()
#     series_phi = Dict{Int, Tuple{Vector{Float64}, Vector{Float64}}}()

#     for (i, order) in enumerate(p.orders)
#         sub = filter(row -> row.order == order, df)
#         sort!(sub, :n)
#         hvals = collect(Float64.(p.L ./ sub.n))
#         e_w = collect(Float64.(sub.L2_error_w))
#         e_phi = collect(Float64.(sub.L2_error_phi))
#         series_w[order] = (hvals, e_w)
#         series_phi[order] = (hvals, e_phi)
#         scatterlines!(
#             ax_w,
#             hvals,
#             e_w,
#             marker=markers[i],
#             markersize=10,
#             color=colors[i],
#             linewidth=2.6,
#             label="p = $(order)",
#         )
#         scatterlines!(
#             ax_phi,
#             hvals,
#             e_phi,
#             marker=markers[i],
#             markersize=10,
#             color=colors[i],
#             linewidth=2.6,
#             label="p = $(order)",
#         )
#     end

#     for (i, order) in enumerate(sort(p.orders))
#         local_color = colors[i]
#         hs_w, errs_w = series_w[order]
#         add_rate_triangle_on_curve!(ax_w, hs_w, errs_w, order + 1; color=local_color)
#     end

#     first_order = sort(p.orders)[1]
#     hs_phi, errs_phi = series_phi[first_order]
#     add_rate_triangle_on_curve!(ax_phi, hs_phi, errs_phi, p.order_phi + 1; color=:black)

#     xlims!(ax_w, minimum(sorted_h) * 0.9, maximum(sorted_h) * 1.1)
#     xlims!(ax_phi, minimum(sorted_h) * 0.9, maximum(sorted_h) * 1.1)
#     ylims!(ax_w, 0.5 * minimum(df.L2_error_w), 2.0 * maximum(df.L2_error_w))
#     ylims!(ax_phi, 0.5 * minimum(df.L2_error_phi), 2.0 * maximum(df.L2_error_phi))

#     xtick_labels = ["1/$(n)" for n in sorted_ns]
#     ax_w.xticks = (sorted_h, xtick_labels)
#     ax_phi.xticks = (sorted_h, xtick_labels)

#     axislegend(ax_w, position=:rb, framevisible=true, title="Polynomial order")
#     axislegend(ax_phi, position=:rb, framevisible=true, title="Polynomial order")

#     Label(
#         fig[0, :],
#         "Figure 5.2: Periodic-beam spatial convergence in time domain",
#         fontsize=20,
#         font=:bold,
#         tellwidth=false,
#     )

#     pdf_path = "plots/5-2-convergence/fig5_convergence_time.pdf"
#     png_path = "plots/5-2-convergence/fig5_convergence_time.png"
#     save(pdf_path, fig)
#     save(png_path, fig, px_per_unit=300 / 96)

#     return pdf_path, png_path
# end

# function print_summary(df::DataFrame)
#     println("\nConvergence summary (time domain):")
#     println(df)

#     println("\nEstimated orders from adjacent refinements (w error):")
#     for order in unique(df.order)
#         sub = sort(filter(row -> row.order == order, df), :n)
#         hs = params.L ./ sub.n
#         errs = sub.L2_error_w
#         rates = [log(errs[i] / errs[i + 1]) / log(hs[i] / hs[i + 1]) for i in 1:(length(errs) - 1)]
#         @printf("  p = %d -> rates = %s\n", order, string(round.(rates, digits=3)))
#     end
# end

# function run_5_2_1_spatial_convergence(
#     ;
#     ns=params.ns,
#     orders=params.orders,
#     force=false,
#     make_plots=true,
#     save_csv=true,
#     verbose=true,
#     verbose_steps=false,
#     save_pvd=false,
#     pvd_step_stride=20,
#     pvd_all_steps=false,
# )
#     mkpath("data/5-2-convergence")

#     warmup_n = minimum(ns)
#     warmup_order = minimum(orders)
#     warmup_k = params.k
#     warmup_order_phi = params.order_phi
#     warmup_Δt = params.Δt

#     if verbose
#         println("[5-2-1] Stage 1/3: warm-up solve (n=$(warmup_n), order=$(warmup_order), Δt=$(warmup_Δt), tf=$(warmup_Δt))")
#     end
#     warm = run_warmup_case(
#         n=warmup_n,
#         order=warmup_order,
#         k=warmup_k,
#         order_phi=warmup_order_phi,
#         Δt=warmup_Δt,
#         verbose_steps=verbose_steps,
#         stage_label="5-2-1 warmup",
#     )
#     if verbose
#         println("[5-2-1] Warm-up complete: L2_w=$(warm.l2_w), L2_phi=$(warm.l2_ϕ)")
#         println("[5-2-1] Stage 2/3: spatial convergence sweep")
#     end

#     rows = Dict[]
#     for order in orders
#         for n in ns
#             if verbose
#                 println("[5-2-1] Solving case n=$(n), order=$(order)")
#             end
#             cfg = Dict(
#                 :n => n,
#                 :order => order,
#                 :L => params.L,
#                 :H => params.H,
#                 :k => params.k,
#                 :Δt => params.Δt,
#                 :tf => params.tf,
#                 :order_phi => params.order_phi,
#             )

#             if save_pvd
#                 problem, result, t_end = build_time_problem(
#                     n,
#                     order;
#                     p=params,
#                     verbose_steps=verbose_steps,
#                     stage_label="5-2-1 n=$(n) p=$(order)",
#                 )
#                 l2_w, l2_ϕ = compute_errors(problem, result; p=params, t_end=t_end)
#                 write_case_pvd(
#                     problem,
#                     result;
#                     n=n,
#                     order=order,
#                     p=params,
#                     step_stride=pvd_step_stride,
#                     write_all_steps=pvd_all_steps,
#                 )
#                 out = Dict(
#                     :n => n,
#                     :order => order,
#                     :L2_error_w => l2_w,
#                     :L2_error_phi => l2_ϕ,
#                 )
#             else
#                 out, _ = produce_or_load("data/5-2-convergence", cfg; force=force, filename="time_n$(n)_p$(order)") do c
#                     run_case(
#                         c[:n],
#                         c[:order];
#                         p=params,
#                         verbose_steps=verbose_steps,
#                         stage_label="5-2-1 n=$(c[:n]) p=$(c[:order])",
#                     )
#                 end
#             end

#             push!(rows, out)
#         end
#     end

#     df = DataFrame(rows)
#     sort!(df, [:order, :n])

#     if save_csv
#         save_convergence_csv(df, "data/5-2-convergence/convergence_time.csv")
#     end

#     if make_plots
#         with_theme(VLFS_THEME) do
#             plot_convergence(df)
#         end
#     end

#     if verbose
#         println("[5-2-1] Stage 3/3: summary and outputs complete")
#     end

#     if verbose
#         print_summary(df)
#     end

#     return df
# end

# run_convergence_frequency(; kwargs...) = run_5_2_1_spatial_convergence(; kwargs...)

# if abspath(PROGRAM_FILE) == @__FILE__
#     run_5_2_1_spatial_convergence()
# end

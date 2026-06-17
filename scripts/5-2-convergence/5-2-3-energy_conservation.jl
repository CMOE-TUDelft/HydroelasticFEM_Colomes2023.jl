"""
Section 5.2.3: periodic-beam energy conservation
"""

if !isdefined(@__MODULE__, :VLFS_THEME)
    include("../../src/plot_theme.jl")
end

# ─── Plotting helpers ─────────────────────────────────────────────────────────

function plot_energy_conservation(df::DataFrame)
    mkpath("plots/5-2-convergence")

    t = collect(Float64.(df.t_global))
    E0 = df.E_kin_f₀[1] + df.E_pot_f₀[1] + df.E_kin_s₀[1] + df.E_ela_s₀[1]

    fig = Figure(size=(980, 620), fontsize=16)
    ax_E = Axis(
        fig[1, 1],
        xlabel="t",
        ylabel="Normalized energy",
        title="(a) Energy components",
        yscale=log10,
    )
    ax_drift = Axis(
        fig[2, 1],
        xlabel="t",
        ylabel="|(E(t)-E(0))/E(0)|",
        title="(b) Relative total-energy drift",
        yscale=log10,
    )

    E_total_f = df.E_kin_f .+ df.E_pot_f
    E_total_s = df.E_kin_s .+ df.E_ela_s
    E_total = E_total_f .+ E_total_s
    E_total_rel = abs.(E_total .- E0) ./ E0

    lines!(ax_E, t[2:end], E_total_f[2:end] ./ E0, color=:firebrick, linewidth=2.2, label="Fluid energy / E0")
    lines!(ax_E, t[2:end], E_total_s[2:end] ./ E0, color=:navy, linewidth=2.2, label="Structure energy / E0")
    lines!(ax_E, t[2:end], E_total[2:end] ./ E0, color=:black, linewidth=2.4, label="Total energy / E0")
    axislegend(ax_E, position=:rb, framevisible=true)

    lines!(ax_drift, t[2:end], E_total_rel[2:end], color=:darkgreen, linewidth=2.4)

    Label(
        fig[0, :],
        "Section 5.2: Energy conservation test",
        fontsize=20,
        font=:bold,
        tellwidth=false,
    )

    pdf_path = "plots/5-2-convergence/fig5_energy_conservation.pdf"
    png_path = "plots/5-2-convergence/fig5_energy_conservation.png"
    save(pdf_path, fig)
    save(png_path, fig, px_per_unit=300 / 96)
    println("[5-2-3] Energy conservation plot saved to: ", pdf_path, " and ", png_path)

end

function _print_energy_conservation_summary(df::DataFrame)
    E0 = df.E_kin_f₀[1] + df.E_pot_f₀[1] + df.E_kin_s₀[1] + df.E_ela_s₀[1]
    E_final = df.E_kin_f[end] + df.E_pot_f[end] + df.E_kin_s[end] + df.E_ela_s[end]
    drift = (E_final - E0) / E0
    println("\nEnergy conservation summary:")
    println("  Initial total energy E(0) = ", round(E0, sigdigits=4))
    println("  Final total energy E(tf) = ", round(E_final, sigdigits=4))
    println("  Relative energy drift (E(tf)-E(0))/E(0) = ", round(drift, sigdigits=4))
end

# ─── Main runner ────────────────────────────────────────────────────────

function run_5_2_3_energy_conservation(
    ;
    force=false,
    make_plots=true,
    save_csv=true,
    verbose=true,
    verbose_steps=false,
    vtk_output=false,
)

  function run_5_2_3(case::PeriodicBeam_params)
    case_name = savename(case)
    println("-------------")
    println("Case: ", case_name)
    _, _, E_kin_f, E_pot_f, E_kin_s, E_ela_s, E_kin_f₀, E_kin_s₀, E_pot_f₀, E_ela_s₀, t_global = run_periodic_beam(case)
    return Dict(
        "name"   => case.name,
        "n"      => case.n,
        "dt"     => Float64(case.dt),
        "tf"     => Float64(case.tf),
        "orderϕ" => case.orderϕ,
        "orderη" => case.orderη,
        "k"      => case.k,
        "E_kin_f" => E_kin_f,
        "E_pot_f" => E_pot_f,
        "E_kin_s" => E_kin_s,
        "E_ela_s" => E_ela_s,
        "E_kin_f₀" => E_kin_f₀,
        "E_kin_s₀" => E_kin_s₀,
        "E_pot_f₀" => E_pot_f₀,
        "E_ela_s₀" => E_ela_s₀,
        "t_global" => t_global,
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
  path = datadir("5-2-3-periodic-beam-energy-conservation")
  case = PeriodicBeam_params(name="Warm-up", n=n, dt=Δt, tf=T, k=k, orderϕ=order, orderη=order)
  verbose && println("[5-2-3] Warm-up: n=$(n), order=$(order), k=$(k), Δt=$(Δt)")
  produce_or_load(path, case, run_5_2_3; force=force, digits=8)

  # ── Energy conservation test: k=15, small Δt, moderate n and order ────────
  n=32
  order=4
  order_phi=order
  k=15
  Δt=1.0e-3
  ω = sqrt(g * k * tanh(k * H))
  tf=(2 * pi / ω) * 2
  verbose && println("[5-2-3] Energy conservation test: k=$(k), n=$(n), order=$(order), order_phi=$(order_phi), Δt=$(Δt), tf=$(tf)")

  case  = PeriodicBeam_params(
      name="energyConservation",
      n=n, dt=Δt, tf=tf, k=k,
      orderϕ=order_phi, orderη=order,
      verbose_steps=verbose_steps,
      vtk_output=vtk_output,
  )
  verbose && println("[5-2-3] Solving: n=$(n), order=$(order), order_phi=$(order_phi), k=$(k), Δt=$(Δt), tf=$(tf)")
  data, _ = produce_or_load(path, case, run_5_2_3; force=force, digits=8)

  df = DataFrame(data)

  if save_csv
      mkpath("data/5-2-convergence")
      csv_path = "data/5-2-convergence/energy_conservation.csv"
      open(csv_path, "w") do io
          writedlm(io, ["t_global" "E_kin_f" "E_pot_f" "E_kin_s" "E_ela_s" "E_kin_f₀" "E_pot_f₀" "E_kin_s₀" "E_ela_s₀"], ',')
          writedlm(io, [df.t_global df.E_kin_f df.E_pot_f df.E_kin_s df.E_ela_s df.E_kin_f₀ df.E_pot_f₀ df.E_kin_s₀ df.E_ela_s₀], ',')
      end
      verbose && println("[5-2-3] Energy conservation data saved to: ", csv_path)
  end

  if make_plots
    with_theme(VLFS_THEME) do
      plot_energy_conservation(df)
    end
  end

  if verbose
    _print_energy_conservation_summary(df)
  end

end
module APIAdapters

import HydroElasticFEM as HE
import Gridap

const solve_frequency_model = HE.Simulation.simulate
const export_vtk = Gridap.writevtk

"""
    build_beam_frequency_model_old_api(args...; kwargs...)

TODO: Replace old monolithic constructors (`run_periodic_beam`,
`run_Khabakhpasheva_freq_domain`) with a helper that instantiates
`EulerBernoulliBeam`, `PotentialFlow`, and `FreeSurface`, then calls
`build_problem(..., FreqDomainConfig(...))`.
"""
function build_beam_frequency_model_old_api(args...; kwargs...)
  error("TODO: implement beam frequency-domain adapter to HydroElasticFEM entities")
end

"""
    build_beam_time_model_old_api(args...; kwargs...)

TODO: Bridge old time-domain setup (`TransientConstantFEOperator` + Newmark)
to `build_problem(..., TimeDomainConfig(...))` + `simulate(problem, tconfig)`.
"""
function build_beam_time_model_old_api(args...; kwargs...)
  error("TODO: implement beam time-domain adapter to HydroElasticFEM")
end

"""
    build_plate_frequency_model_old_api(args...; kwargs...)

TODO: Port old Yago/MultiGeo plate setup to `KirchhoffLovePlate` entity
construction with `build_kl_tensor` where needed.
"""
function build_plate_frequency_model_old_api(args...; kwargs...)
  error("TODO: implement plate frequency-domain adapter to HydroElasticFEM")
end

"""
    build_elastic_joint_term_old_api(k_r, domain_symbol, normal_symbol)

TODO: Old scripts added joint terms explicitly in weak forms. New API requires
`JointRotationalSpring(domain_symbol, normal_symbol, k_r)` embedded in
`EulerBernoulliBeam(joints=[...])`.
"""
function build_elastic_joint_term_old_api(k_r, domain_symbol, normal_symbol)
  return HE.Physics.JointRotationalSpring(domain_symbol, normal_symbol, k_r)
end

"""
    configure_variable_bathymetry_old_api(args...; kwargs...)

TODO: Missing public-facing API to pass spatially varying bathymetry `H(x)`
for section 5.4 workflows without pre-processing external meshes.
"""
function configure_variable_bathymetry_old_api(args...; kwargs...)
  error("TODO: missing upstream HydroElasticFEM public API for variable bathymetry H(x)")
end

"""
    build_time_integrator_old_api(args...; kwargs...)

TODO: Old scripts configured `Newmark`. New path should map old knobs to
`TimeConfig` (`Δt`, `ρ∞`, initial fields), then rely on simulation internals.
"""
function build_time_integrator_old_api(args...; kwargs...)
  error("TODO: implement Newmark-to-TimeConfig compatibility adapter")
end

"""
    extract_deflection_old_api(result)

TODO: Provide stable extraction helper for structural deflection (`w`/`η`) from
`SimResult.solution` using `result.fmap` symbols.
"""
function extract_deflection_old_api(result)
  error("TODO: implement deflection field extraction adapter from SimResult")
end

"""
    extract_free_surface_phi_old_api(result)

TODO: Provide stable extraction helper for free-surface / potential fields from
`SimResult.solution` via symbol mapping.
"""
function extract_free_surface_phi_old_api(result)
  error("TODO: implement free-surface/potential field extraction adapter")
end

end # module

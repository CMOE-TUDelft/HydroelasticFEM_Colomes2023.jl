module HydroElasticFEM_Colomes2023

export run_tests

include("5_2_convergence_time_domain_main.jl")

function _ensure_5_2_script_modules_loaded()
	scripts_dir = normpath(joinpath(@__DIR__, "..", "scripts", "5-2-convergence"))

	if !isdefined(@__MODULE__, :Section5_2_1_SpatialConvergence)
		include(joinpath(scripts_dir, "5-2-1-spatial_convergence.jl"))
	end
	if !isdefined(@__MODULE__, :Section5_2_2_TimeConvergence)
		include(joinpath(scripts_dir, "5-2-2-time-convergence.jl"))
	end
	if !isdefined(@__MODULE__, :Section5_2_3_EnergyConservation)
		include(joinpath(scripts_dir, "5-2-3-energy_conservation.jl"))
	end
	return nothing
end

function run_tests(
	name::AbstractString="all";
	force::Bool=false,
	make_plots::Bool=false,
	save_csv::Bool=false,
	verbose::Bool=false,
	verbose_steps::Bool=false,
)
	_ensure_5_2_script_modules_loaded()

	run_5_2_1 = () -> Section5_2_1_SpatialConvergence.run_5_2_1_spatial_convergence(
		force=force,
		make_plots=make_plots,
		save_csv=save_csv,
		verbose=verbose,
		verbose_steps=verbose_steps,
	)
	run_5_2_2 = () -> Section5_2_2_TimeConvergence.run_5_2_2_time_convergence(
		force=force,
		make_plots=make_plots,
		save_csv=save_csv,
		verbose=verbose,
		verbose_steps=verbose_steps,
	)
	run_5_2_3 = () -> Section5_2_3_EnergyConservation.run_5_2_3_energy_conservation(
		force=force,
		make_plots=make_plots,
		save_csv=save_csv,
		verbose=verbose,
		verbose_steps=verbose_steps,
	)

	if name == "all"
		return (
			section_5_2_1=run_5_2_1(),
			section_5_2_2=run_5_2_2(),
			section_5_2_3=run_5_2_3(),
		)
	elseif name in ("5-2-1", "5_2_1", "spatial")
		return run_5_2_1()
	elseif name in ("5-2-2", "5_2_2", "time")
		return run_5_2_2()
	elseif name in ("5-2-3", "5_2_3", "energy")
		return run_5_2_3()
	else
		error("Unknown test name: $(name). Use one of: all, 5-2-1, 5-2-2, 5-2-3")
	end
end

end

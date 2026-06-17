module HydroElasticFEM_Colomes2023

using DrWatson
using CairoMakie
using DataFrames
using DataFramesMeta
using DelimitedFiles
using Printf

export run_tests

include("5_2_convergence_time_domain_main.jl")
include("5_3_1_Khabakpasheva_time_domain.jl")

const var"5_2"          = ConvergenceTimeDomain
const PeriodicBeam_params = var"5_2".PeriodicBeam_params
const run_periodic_beam   = var"5_2".run_periodic_beam
const var"5_3_1"        = KhavakpashevaTimeDomain
const Khabakpasheva_time_params = var"5_3_1".Khabakpasheva_time_params
const run_khabakpasheva_time   = var"5_3_1".run_khabakpasheva_time

include("../scripts/5-2-convergence/5-2-1-spatial_convergence.jl")
include("../scripts/5-2-convergence/5-2-2-time-convergence.jl")
include("../scripts/5-2-convergence/5-2-3-energy_conservation.jl")
include("../scripts/5-3-Khabakhpasheva/5-3-1-khabakhpasheva_time.jl")

function run_tests(
	name::AbstractString="all";
	force::Bool=false,
	make_plots::Bool=false,
	save_csv::Bool=false,
	verbose::Bool=false,
	verbose_steps::Bool=false,
	vtk_output::Bool=false,
	test_suit::Bool=false, # If true, only run the warm-up cases for testing purposes
)
	run_5_2_1 = () -> run_5_2_1_spatial_convergence(
		force=force,
		make_plots=make_plots,
		save_csv=save_csv,
		verbose=verbose,
		verbose_steps=verbose_steps,
		vtk_output=vtk_output,
		test_suit=test_suit,
	)
	run_5_2_2 = () -> run_5_2_2_time_convergence(
		force=force,
		make_plots=make_plots,
		save_csv=save_csv,
		verbose=verbose,
		verbose_steps=verbose_steps,
		vtk_output=vtk_output,
		test_suit=test_suit,
	)
	run_5_2_3 = () -> run_5_2_3_energy_conservation(
		force=force,
		make_plots=make_plots,
		save_csv=save_csv,
		verbose=verbose,
		verbose_steps=verbose_steps,
		vtk_output=vtk_output,
		test_suit=test_suit,
	)
	run_5_3_1 = () -> run_5_3_1_khabakpasheva_time(
		force=force,
		make_plots=make_plots,
		save_csv=save_csv,
		verbose=verbose,
		verbose_steps=verbose_steps,
		vtk_output=vtk_output,
		test_suit=test_suit,
	)

	if name == "all"
		return (
			section_5_2_1=run_5_2_1(),
			section_5_2_2=run_5_2_2(),
			section_5_2_3=run_5_2_3(),
			section_5_3_1=run_5_3_1(),
		)
	elseif name in ("5-2-1", "5_2_1", "spatial")
		return run_5_2_1()
	elseif name in ("5-2-2", "5_2_2", "time")
		return run_5_2_2()
	elseif name in ("5-2-3", "5_2_3", "energy")
		return run_5_2_3()
	elseif name in ("5-3-1", "5_3_1", "khabakpasheva")
		return run_5_3_1()
	else
		error("Unknown test name: $(name). Use one of: all, 5-2-1, 5-2-2, 5-2-3, 5-3-1")
	end
end

end

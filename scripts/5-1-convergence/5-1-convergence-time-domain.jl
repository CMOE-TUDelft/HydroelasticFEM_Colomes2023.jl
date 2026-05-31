"""
Compatibility entrypoint for Section 5.1 scripts.

The original monolithic script has been split into:
- 5-1-1-spatial_convergence.jl
- 5-1-2-time-convergence.jl
- 5-1-3-energy_conservation.jl
"""

include("5-1-1-spatial_convergence.jl")
include("5-1-2-time-convergence.jl")
include("5-1-3-energy_conservation.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    run_5_1_1_spatial_convergence()
end

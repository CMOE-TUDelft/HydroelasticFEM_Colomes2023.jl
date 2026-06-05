"""
Smoke tests for HydroElasticFEM_Colomes2023 Section 5.2 scripts.

Goal: verify each script entrypoint runs without throwing.
No value assertions are performed.
"""

using Test
using HydroElasticFEM_Colomes2023

@testset "5.2 Script Smoke Tests (No-Failure)" begin
    # Use the same warm-up-scale inputs for all 5.2 tests.
    p = HydroElasticFEM_Colomes2023.ConvergenceTimeDomain.params
    n_warm = minimum(p.ns)
    order_warm = minimum(p.orders)
    dt_warm = p.Δt
    k_warm = p.k

    mktempdir() do tmp
        cd(tmp) do
            @testset "5-2-1 warm-up scale" begin
                HydroElasticFEM_Colomes2023.run_5_2_1_spatial_convergence(
                    ns=[n_warm],
                    orders=[order_warm],
                    force=true,
                    make_plots=false,
                    save_csv=false,
                    verbose=false,
                    verbose_steps=false,
                    save_pvd=false,
                )
                @test true
            end

            @testset "5-2-2 warm-up scale" begin
                HydroElasticFEM_Colomes2023.run_5_2_2_time_convergence(
                    Δts=[dt_warm],
                    n=n_warm,
                    order=order_warm,
                    order_phi=order_warm,
                    k=k_warm,
                    tf=dt_warm,
                    force=true,
                    make_plots=false,
                    save_csv=false,
                    verbose=false,
                    verbose_steps=false,
                )
                @test true
            end

            @testset "5-2-3 warm-up scale" begin
                HydroElasticFEM_Colomes2023.run_5_2_3_energy_conservation(
                    n=n_warm,
                    order=order_warm,
                    order_phi=order_warm,
                    k=k_warm,
                    Δt=dt_warm,
                    tf=dt_warm,
                    force=true,
                    make_plots=false,
                    save_csv=false,
                    verbose=false,
                    verbose_steps=false,
                )
                @test true
            end
        end
    end
end

"""
Smoke tests for HydroElasticFEM_Colomes2023 Section 5.2 scripts.

Goal: verify each script entrypoint runs without throwing.
No value assertions are performed.
"""

using Test
include("../src/HydroElasticFEM_Colomes2023.jl")

@testset "5.2 Script Smoke Tests (No-Failure)" begin

    mktempdir() do tmp
        cd(tmp) do
            @testset "5-2-1 warm-up scale" begin
                HydroElasticFEM_Colomes2023.run_5_2_1_spatial_convergence(test_suit=true)
                @test true
            end

            @testset "5-2-2 warm-up scale" begin
                HydroElasticFEM_Colomes2023.run_5_2_2_time_convergence(test_suit=true)
                @test true
            end

            @testset "5-2-3 warm-up scale" begin
                HydroElasticFEM_Colomes2023.run_5_2_3_energy_conservation(test_suit=true)
                @test true
            end

            @testset "5-3-1 warm-up scale" begin
                HydroElasticFEM_Colomes2023.run_5_3_1_khabakpasheva_time(test_suit=true)
                @test true
            end
        end
    end
end

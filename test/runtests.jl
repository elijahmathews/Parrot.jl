# 
# Parrot.jl/test/runtests.jl
# 
# By Elijah Mathews (me@elijahmathews.com)
# openpgp4fpr:ac1d3fb1e8a5eb7d14bd587b2932c725055a90d8
# 

using Test, Documenter, Parrot, Random

@testset "Layers" begin

    @testset "Alsing" begin

        Random.seed!(1)

        a = Alsing(8,3)
        r = rand(8)

        @test typeof.([a.W, a.b, a.α, a.β]) == [
            Array{Float32,2},
            Array{Float32,1},
            Array{Float32,1},
            Array{Float32,1},
        ]

        @test a(r) ≈ [0.26666774962631395, 0.32477348838252235, 0.14232596416862664]

    end

end


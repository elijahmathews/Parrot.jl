# 
# Parrot.jl/test/runtests.jl
# 
# By Elijah Mathews (me@elijahmathews.com)
# openpgp4fpr:ac1d3fb1e8a5eb7d14bd587b2932c725055a90d8
# 

using Test, Documenter, Parrot

@testset "Hello World" begin
	message = "Hello world!"
	@test message == "Hello world!"

	x = 2 + 2
	@test x == 4
end


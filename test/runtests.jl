using Test, Documenter, ProspectorML

@testset "Doctests" begin
	doctest(ProspectorML)
end

@testset "Hello World" begin
	message = "Hello world!"
	@test message == "Hello world!"

	x = 2 + 2
	@test x == 4
end

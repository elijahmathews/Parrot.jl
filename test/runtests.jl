#
# ProspectorML.jl/test/runtests.jl
#
# By Elijah Mathews
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#

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
